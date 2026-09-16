#!/usr/bin/env python3
"""Extract frozen CMR-Transformer patient embeddings from SAX cine DICOM data."""

import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CHECKPOINT = PROJECT_ROOT / "epoch=599-step=124200.ckpt"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "patient_manifest.csv"
DEFAULT_OUTPUT = Path("./results/foundation_embeddings")
# 默认短轴动态 cine NIfTI 存放目录 (记录各患者短轴切片 NIfTI 文件)
DEFAULT_NIFTI_ROOT = Path("./data/processed_sax_cine")


SAX_PATTERN = re.compile(
    r"(^|[^a-z])(sa|sax)([^a-z]|$)|short[ _-]*axis|sa[ _-]*function|cine.*sa|^sa(?:ch)?fiesta",
    re.IGNORECASE,
)
CINE_PATTERN = re.compile(
    r"cine|function|fiesta|fisp|trufi|btfe|bffe",
    re.IGNORECASE,
)
LONG_AXIS_PATTERN = re.compile(
    r"[1-4][ _-]*(?:ch|cn)|acn|anch|lvot|long[ _-]*axis|(^|[^a-z])la([^a-z]|$)|(^|[^a-z])(ax|axial)([^a-z]|$)",
    re.IGNORECASE,
)
EXCLUDE_PATTERN = re.compile(
    r"lge|late|delayed|enhance|psir|de[_ -]|ir[_ -]|t1|t2|survey|locali[sz]er|scout|diff|adc|map|perfusion|dynamic|4dflow",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-mode", choices=["nifti", "dicom"], default="nifti")
    parser.add_argument("--nifti-root", type=Path, default=DEFAULT_NIFTI_ROOT)
    parser.add_argument("--cohort", choices=["all", "development", "external_test"], default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--scan-only", action="store_true")
    return parser.parse_args()


def descriptor(dataset):
    return " ".join(
        str(getattr(dataset, key, "") or "")
        for key in ("SeriesDescription", "ProtocolName", "SequenceName")
    ).strip()


def is_sax_cine(dataset):
    text = descriptor(dataset)
    return bool(SAX_PATTERN.search(text)) and not EXCLUDE_PATTERN.search(text)


def is_cine_candidate(dataset):
    text = descriptor(dataset)
    return bool(CINE_PATTERN.search(text)) and not EXCLUDE_PATTERN.search(text)


def slice_coordinate(dataset):
    position = getattr(dataset, "ImagePositionPatient", None)
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    if position is not None and orientation is not None:
        row = np.asarray(orientation[:3], dtype=float)
        column = np.asarray(orientation[3:], dtype=float)
        normal = np.cross(row, column)
        return round(float(np.dot(np.asarray(position, dtype=float), normal)), 3)
    value = getattr(dataset, "SliceLocation", None)
    return round(float(value), 3) if value is not None else 0.0


def numeric_tag(dataset, key):
    value = getattr(dataset, key, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def cluster_slice_coordinate(value, tolerance_mm=0.5):
    return round(value / tolerance_mm) * tolerance_mm


def choose_time_values(frames, min_frames):
    for index in (0, 1, 2):
        values = [frame[index] for frame in frames if frame[index] is not None]
        if len(set(values)) >= min_frames:
            return index
    return None


def scan_patient(patient_path, min_frames):
    series = defaultdict(list)
    descriptions = {}
    header_tags = [
        "SeriesInstanceUID",
        "SeriesDescription",
        "ProtocolName",
        "SequenceName",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "SliceLocation",
        "TriggerTime",
        "TemporalPositionIdentifier",
        "InstanceNumber",
    ]
    series_directories = [
        directory
        for study_directory in patient_path.iterdir()
        if study_directory.is_dir()
        for directory in study_directory.iterdir()
        if directory.is_dir()
    ]
    if series_directories:
        candidate_files = []
        description_tags = ["SeriesDescription", "ProtocolName", "SequenceName"]
        for series_directory in series_directories:
            series_files = [path for path in series_directory.iterdir() if path.is_file()]
            if not series_files:
                continue
            try:
                sample = pydicom.dcmread(
                    series_files[0],
                    stop_before_pixels=True,
                    force=True,
                    specific_tags=description_tags,
                )
            except Exception:
                continue
            if is_cine_candidate(sample):
                candidate_files.extend(series_files)
    else:
        candidate_files = [path for path in patient_path.iterdir() if path.is_file()]

    for file_path in candidate_files:
        if not file_path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(
                file_path,
                stop_before_pixels=True,
                force=True,
                specific_tags=header_tags,
            )
        except Exception:
            continue
        if not is_cine_candidate(dataset):
            continue
        series_uid = str(getattr(dataset, "SeriesInstanceUID", file_path.parent))
        descriptions[series_uid] = descriptor(dataset)
        series[series_uid].append(
            (
                cluster_slice_coordinate(slice_coordinate(dataset)),
                numeric_tag(dataset, "TriggerTime"),
                numeric_tag(dataset, "TemporalPositionIdentifier"),
                numeric_tag(dataset, "InstanceNumber"),
                file_path,
            )
        )

    cine_groups = []
    series_audit = []
    for series_uid, records in series.items():
        slices = defaultdict(list)
        for slice_value, trigger, temporal, instance, file_path in records:
            slices[slice_value].append((trigger, temporal, instance, file_path))
        valid_groups = []
        for slice_value, frames in slices.items():
            time_index = choose_time_values(frames, min_frames)
            if time_index is None:
                continue
            unique_frames = {}
            ordered_frames = sorted(
                frames,
                key=lambda frame: (
                    frame[time_index] is None,
                    frame[time_index] if frame[time_index] is not None else 0,
                ),
            )
            for frame in ordered_frames:
                time_value = frame[time_index]
                file_path = frame[3]
                unique_frames.setdefault(time_value, file_path)
            if len(unique_frames) >= min_frames:
                valid_groups.append(
                    {
                        "series_uid": series_uid,
                        "description": descriptions[series_uid],
                        "slice_coordinate": slice_value,
                        "frames": list(unique_frames.values()),
                    }
                )
        text = descriptions[series_uid]
        explicit_sax = bool(SAX_PATTERN.search(text))
        explicit_long_axis = bool(LONG_AXIS_PATTERN.search(text))
        geometry_sax = len(valid_groups) >= 4 and not explicit_long_axis
        accepted = bool(valid_groups) and (explicit_sax or geometry_sax)
        if accepted:
            cine_groups.extend(valid_groups)
        series_audit.append(
            {
                "description": text,
                "images": len(records),
                "slice_positions": len(slices),
                "valid_cine_slices": len(valid_groups),
                "explicit_sax": explicit_sax,
                "explicit_long_axis": explicit_long_axis,
                "geometry_sax": geometry_sax,
                "accepted": accepted,
            }
        )
    return cine_groups, series_audit


def load_cine(frame_paths):
    frames = []
    for frame_path in frame_paths:
        dataset = pydicom.dcmread(frame_path, force=True)
        frame = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        frames.append(frame * slope + intercept)
    return normalize_cine(np.stack(frames))


def normalize_cine(cine):
    lower, upper = np.percentile(cine, [1, 99])
    cine = np.clip(cine, lower, upper)
    cine = (cine - lower) / max(float(upper - lower), 1e-6)
    return torch.from_numpy(cine.astype(np.float32, copy=False))


def load_nifti_cine(path, min_frames):
    cine = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if cine.ndim != 3 or cine.shape[0] < min_frames:
        raise ValueError(f"NIfTI 不是有效的 (T,H,W) cine: {path}, shape={cine.shape}")
    return normalize_cine(cine)


def prepare_cine(cine):
    if cine.shape[0] != 16:
        frame_indices = torch.linspace(0, cine.shape[0] - 1, 16).round().long()
        cine = cine[frame_indices]
    height, width = cine.shape[-2:]
    scale = 244.0 / min(height, width)
    resized_height = max(224, round(height * scale))
    resized_width = max(224, round(width * scale))
    cine = F.interpolate(
        cine.unsqueeze(1),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    top = (resized_height - 224) // 2
    left = (resized_width - 224) // 2
    cine = cine[:, top : top + 224, left : left + 224]
    return cine.unsqueeze(0).repeat(3, 1, 1, 1)


def load_encoder(checkpoint, device):
    from model_factory import Cardiac_MRI_Encoder, load_contrastive_pretrained_weights

    encoder = Cardiac_MRI_Encoder("mvit", 50, 512, 1, None, 16, "linear", False)
    encoder = load_contrastive_pretrained_weights(encoder, str(checkpoint))
    encoder.to(device).eval()
    return encoder


def encode_patient(encoder, cine_groups, batch_size, device):
    """
    通过基础大模型 (Foundation Model) 对患者的原始 DICOM 图像序列提取特征向量。
    返回患者级别的平均归一化特征(patient_embedding)以及各切片的特征(slice_embeddings)。
    """
    embeddings = []
    for start in range(0, len(cine_groups), batch_size):
        batch_groups = cine_groups[start : start + batch_size]
        videos = torch.stack(
            [
                prepare_cine(
                    load_cine(
                        [
                            group["frames"][index]
                            for index in np.linspace(
                                0, len(group["frames"]) - 1, 16
                            ).round().astype(int)
                        ]
                    )
                )
                for group in batch_groups
            ]
        ).to(device)
        with torch.no_grad():
            embeddings.append(encoder(videos).cpu())
    slice_embeddings = torch.cat(embeddings)
    patient_embedding = F.normalize(slice_embeddings.mean(dim=0), dim=0)
    return patient_embedding.numpy(), slice_embeddings.numpy()


def encode_nifti_patient(encoder, paths, batch_size, min_frames, device):
    """
    通过基础大模型对患者的预处理 NIfTI 图像序列提取特征向量。
    返回患者级别的平均归一化特征(patient_embedding)以及各切片的特征(slice_embeddings)。
    """
    embeddings = []
    for start in range(0, len(paths), batch_size):
        videos = torch.stack(
            [prepare_cine(load_nifti_cine(path, min_frames)) for path in paths[start : start + batch_size]]
        ).to(device)
        with torch.no_grad():
            embeddings.append(encoder(videos).cpu())
    slice_embeddings = torch.cat(embeddings)
    patient_embedding = F.normalize(slice_embeddings.mean(dim=0), dim=0)
    return patient_embedding.numpy(), slice_embeddings.numpy()


def main():
    """
    主程序：加载患者清单，读取 DICOM 或 NIfTI 格式的 SAX cine 影像。
    使用冻结的预训练大模型提取 Embeddings，并将每个患者的特征保存到缓存及最终合并的 npz 文件中。
    """
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    warnings.filterwarnings("ignore", message="Invalid value for VR UI")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if args.cohort != "all":
        manifest = manifest.loc[manifest["cohort"] == args.cohort].copy()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index必须位于[0, num-shards)范围内")
    manifest = manifest.iloc[args.shard_index :: args.num_shards].copy()
    if args.max_patients > 0:
        manifest = manifest.head(args.max_patients).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = None if args.scan_only else load_encoder(args.checkpoint, device)
    patient_embeddings = []
    patient_rows = []
    failures = []
    quality_rows = []
    cache_dir = args.output_dir / "patient_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    nifti_paths = {}
    nifti_series = {}
    if args.input_mode == "nifti":
        slice_qc = pd.read_csv(args.nifti_root / "slice_qc.csv")
        slice_qc["ID"] = slice_qc["ID"].astype(int)
        for key, rows in slice_qc.groupby(["cohort", "ID"]):
            rows = rows.sort_values("slice_index")
            nifti_paths[key] = [Path(path) for path in rows["output_path"]]
            nifti_series[key] = int(rows["series_uid"].nunique())

    for patient_number, row in enumerate(manifest.itertuples(index=False), start=1):
        patient_path = Path(row.image_path)
        cache_path = cache_dir / f"{row.cohort}_{int(row.ID):04d}.npz"
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            patient_embeddings.append(cached["embedding"])
            patient_rows.append(
                {
                    "cohort": row.cohort,
                    "ID": int(row.ID),
                    "event": int(row.event),
                    "sax_series": int(cached["sax_series"]),
                    "sax_slices": int(cached["sax_slices"]),
                }
            )
            print(
                f"[{patient_number}/{len(manifest)}] {row.cohort} ID={row.ID}: cached",
                flush=True,
            )
            continue
        key = (str(row.cohort), int(row.ID))
        if args.input_mode == "nifti":
            cine_groups = nifti_paths.get(key, [])
            series_audit = []
            candidate_series = nifti_series.get(key, 0)
            accepted_series = candidate_series
        else:
            cine_groups, series_audit = scan_patient(patient_path, args.min_frames)
            candidate_series = len(series_audit)
            accepted_series = sum(item["accepted"] for item in series_audit)
            for item in series_audit:
                quality_rows.append(
                    {"cohort": row.cohort, "ID": int(row.ID), **item}
                )
        print(
            f"[{patient_number}/{len(manifest)}] {row.cohort} ID={row.ID}: "
            f"candidates={candidate_series}, accepted={accepted_series}, "
            f"slices={len(cine_groups)}",
            flush=True,
        )
        if not cine_groups:
            failures.append(
                {
                    "cohort": row.cohort,
                    "ID": int(row.ID),
                    "reason": "excluded_no_valid_sax_cine"
                    if args.input_mode == "nifti"
                    else "no_valid_sax_cine",
                }
            )
            continue
        if args.scan_only:
            patient_rows.append(
                {
                    "cohort": row.cohort,
                    "ID": int(row.ID),
                    "event": int(row.event),
                    "sax_series": accepted_series,
                    "sax_slices": len(cine_groups),
                }
            )
            continue
        try:
            if args.input_mode == "nifti":
                patient_embedding, _ = encode_nifti_patient(
                    encoder, cine_groups, args.batch_size, args.min_frames, device
                )
            else:
                patient_embedding, _ = encode_patient(
                    encoder, cine_groups, args.batch_size, device
                )
        except Exception as error:
            failures.append(
                {"cohort": row.cohort, "ID": int(row.ID), "reason": repr(error)}
            )
            continue
        patient_embeddings.append(patient_embedding)
        np.savez_compressed(
            cache_path,
            embedding=patient_embedding,
            sax_series=accepted_series,
            sax_slices=len(cine_groups),
        )
        patient_rows.append(
            {
                "cohort": row.cohort,
                "ID": int(row.ID),
                "event": int(row.event),
                "sax_series": accepted_series,
                "sax_slices": len(cine_groups),
            }
        )

    metadata = pd.DataFrame(patient_rows)
    failures_table = pd.DataFrame(failures, columns=["cohort", "ID", "reason"])
    quality_table = pd.DataFrame(quality_rows)
    metadata.to_csv(args.output_dir / "embedding_metadata.csv", index=False)
    failures_table.to_csv(args.output_dir / "embedding_failures.csv", index=False)
    quality_table.to_csv(args.output_dir / "series_quality_control.csv", index=False)
    if patient_embeddings and not args.scan_only:
        np.savez_compressed(
            args.output_dir / "patient_embeddings.npz",
            embeddings=np.stack(patient_embeddings),
            cohort=metadata["cohort"].to_numpy(),
            patient_id=metadata["ID"].to_numpy(),
            labels=metadata["event"].to_numpy(),
        )
    run_metadata = {
        "checkpoint": str(args.checkpoint),
        "input": "processed noncontrast SAX cine NIfTI"
        if args.input_mode == "nifti"
        else "noncontrast SAX cine DICOM",
        "input_mode": args.input_mode,
        "nifti_root": str(args.nifti_root) if args.input_mode == "nifti" else "",
        "temporal_sampling": "uniform 16 frames",
        "spatial_preprocessing": "short-side resize to 244 then center crop 224",
        "intensity_preprocessing": "1st-99th percentile clip and min-max to [0,1] per slice cine",
        "slice_aggregation": "mean then L2 normalization",
        "patients_requested": len(manifest),
        "patients_completed": len(metadata),
        "patients_failed": len(failures_table),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "device": str(device),
        "scan_only": args.scan_only,
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(run_metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
