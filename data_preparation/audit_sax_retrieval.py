#!/usr/bin/env python3
"""只读扫描全部 DICOM，审计每位患者的 SAX cine 检索完整性。"""

import argparse
import json
import math
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")


DEFAULT_MANIFEST = Path(__file__).resolve().parent / "patient_manifest.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "sax_retrieval_audit"
HEADER_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SeriesNumber",
    "InstanceNumber",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ImageType",
    "Manufacturer",
    "Rows",
    "Columns",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "SliceLocation",
    "TemporalPositionIdentifier",
    "TriggerTime",
    "AcquisitionTime",
    "NumberOfFrames",
]


def parse_args():
    """解析命令行参数，支持分片(shard)并行处理和条件过滤。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cohort", choices=["all", "development", "external_test"], default="all"
    )
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--min-slices", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-patients", type=int, default=0)
    return parser.parse_args()


def normalized_text(*values):
    text = " ".join(str(value or "") for value in values).lower()
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip() + " "


def contains(text, patterns):
    return any(re.search(pattern, text) for pattern in patterns)


def classify_modality(dataset):
    """
    根据 DICOM 头信息 (SeriesDescription, ProtocolName等) 对序列模态进行分类。
    识别出 CINE (电影序列), LGE, T1, T2, PERFUSION, BLACK_BLOOD 等。
    """
    text = normalized_text(
        dataset.get("SeriesDescription", ""),
        dataset.get("ProtocolName", ""),
        dataset.get("SequenceName", ""),
        dataset.get("ImageType", ""),
    )
    if contains(
        text,
        (
            r" localizer ", r" survey ", r" scout ", r" secondary capture ",
            r" screen save ", r" calibration ", r" qflow ", r" flow ",
            r" 3 pl fiesta ", r" realtime fiesta loc ",
        ),
    ):
        return "OTHER", "excluded_non_diagnostic"
    if contains(text, (r" perfusion ", r" first pass ", r" dynamic ", r" dyn stfe ", r" fgre time test ")):
        return "PERFUSION", "perfusion_pattern"
    if contains(text, (r" black blood ", r" double ir ", r" triple ir ", r" stir ", r" haste ")):
        return "BLACK_BLOOD", "black_blood_pattern"
    if contains(text, (r" smart1map ", r" shmolli ", r" molli ", r" t1 map", r" t1mapping ")):
        return "T1", "t1_pattern"
    if contains(text, (r" lge ", r" psir ", r" psmde ", r" mde ", r" delayed ", r" late enhancement ", r" de overview ")):
        return "LGE", "lge_pattern"
    if contains(text, (r" t2 map", r" t2mapping ", r" t2star ", r" t2ssfse ")):
        return "T2", "t2_pattern"
    if contains(text, (r"cine", r"function", r" retro ", r" fiesta ", r" btfe ", r" bffe ", r" trufi ")):
        return "CINE", "cine_pattern"
    return "OTHER", "no_cine_pattern"


def classify_view(dataset):
    """
    根据 DICOM 头信息识别扫描切面视角。
    识别 SAX (短轴), 2CH, 3CH, 4CH (两腔/三腔/四腔) 以及 LAX (长轴)。
    """
    text = normalized_text(dataset.get("SeriesDescription", ""), dataset.get("ProtocolName", ""))
    labels = []
    if contains(text, (r" sax ", r" short axis ", r" shortaxis ", r" sa ", r" sa function ")):
        labels.append("SAX")
    if contains(text, (r" 2ch", r" 2 ch ", r" 2cn", r" two chamber ")):
        labels.append("2CH")
    if contains(text, (r" 3ch", r" 3 ch ", r" 3cn", r" acn ", r" anch ", r" lvot ")):
        labels.append("3CH")
    if contains(text, (r" 4ch", r" 4 ch ", r" 4cn", r" four chamber ")):
        labels.append("4CH")
    if len(set(labels)) > 1:
        return "MULTI"
    if labels:
        return labels[0]
    if contains(text, (r" long axis ", r" lax ")):
        return "LAX"
    if contains(text, (r" axial ", r" ax ", r" tra ")):
        return "AXIAL"
    return "UNKNOWN"


def raw_files(directory):
    ignored = {".py", ".mat", ".nii", ".gz", ".xlsx", ".csv", ".json"}
    return [
        directory / name
        for name in os.listdir(directory)
        if (directory / name).is_file()
        and name.upper() != "DICOMDIR"
        and Path(name).suffix.lower() not in ignored
    ]


def scan_files(patient_path):
    direct = raw_files(patient_path)
    if direct:
        for path in direct:
            yield path, "full", 1
        return
    for root, _, _ in os.walk(patient_path):
        directory = Path(root)
        files = raw_files(directory)
        if not files:
            continue
        sample = None
        for path in files:
            try:
                sample = pydicom.dcmread(
                    path, stop_before_pixels=True, defer_size="1 KB", force=True, specific_tags=HEADER_TAGS
                )
            except Exception:
                continue
            if sample.get("Rows") and sample.get("Columns"):
                break
        if sample is None:
            yield files[0], "representative", len(files)
            continue
        modality, _ = classify_modality(sample)
        view = classify_view(sample)
        if modality == "CINE" or view == "SAX":
            for path in files:
                yield path, "full", 1
        else:
            yield files[0], "representative", len(files)


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_geometry(dataset):
    """
    通过 ImageOrientationPatient 和 ImagePositionPatient 计算图像切片的标准法向量(normal)和截距(location)。
    用于判断多个文件是否同属一个切面位置(即属于同一几何层)。
    """
    orientation = dataset.get("ImageOrientationPatient")
    position = dataset.get("ImagePositionPatient")
    if orientation is not None and position is not None:
        try:
            row = np.asarray(orientation[:3], dtype=float)
            column = np.asarray(orientation[3:], dtype=float)
            normal = np.cross(row, column)
            normal /= max(float(np.linalg.norm(normal)), 1e-8)
            if next((value for value in normal if abs(value) > 1e-6), 1.0) < 0:
                normal = -normal
            scalar = float(np.dot(normal, np.asarray(position, dtype=float)))
            return tuple(np.round(normal, 3)), round(scalar * 2) / 2
        except (TypeError, ValueError):
            pass
    location = as_float(dataset.get("SliceLocation"))
    return None, round(location * 2) / 2 if location is not None else None


def temporal_values(item):
    return item["trigger"], item["temporal"], item["instance"], item["acquisition"]


def angle_degrees(first, second):
    cosine = min(1.0, max(0.0, abs(float(np.dot(first, second)))))
    return math.degrees(math.acos(cosine))


def count_frames(items, min_frames):
    for index in range(4):
        values = [temporal_values(item)[index] for item in items if temporal_values(item)[index] not in (None, "")]
        if len(set(values)) >= min_frames:
            return len(set(values)), ("TriggerTime", "TemporalPositionIdentifier", "InstanceNumber", "AcquisitionTime")[index]
    multiframe = max((item["number_of_frames"] or 0 for item in items), default=0)
    return int(multiframe), "NumberOfFrames" if multiframe else "none"


def scan_patient(row, min_frames, min_slices):
    """
    对单个患者的所有 DICOM 文件进行地毯式扫描和审计。
    1. 根据文件头分类模态与切面。
    2. 将同一位置(根据法向量和位置)的图像分组。
    3. 检查每层的帧数(至少 min_frames) 以及有效的切片数量(至少 min_slices)。
    4. 判断该序列是否能明确认定为 SAX cine，返回判定状态(如 AUTO_CONFIRMED 等)。
    """
    records = {}
    unreadable = 0
    for path, scan_scope, represented_files in scan_files(Path(row.image_path)):
        try:
            dataset = pydicom.dcmread(
                path, stop_before_pixels=True, defer_size="1 KB", force=True, specific_tags=HEADER_TAGS
            )
        except Exception:
            unreadable += 1
            continue
        if not dataset.get("Rows") or not dataset.get("Columns"):
            continue
        series_uid = str(dataset.get("SeriesInstanceUID") or f"MISSING::{path.parent}")
        study_uid = str(dataset.get("StudyInstanceUID") or "MISSING")
        key = study_uid, series_uid
        if key not in records:
            modality, modality_reason = classify_modality(dataset)
            records[key] = {
                "dataset": dataset,
                "modality": modality,
                "modality_reason": modality_reason,
                "view": classify_view(dataset),
                "items": [],
                "sop": set(),
                "duplicates": 0,
                "scan_scopes": set(),
                "represented_files": 0,
            }
        record = records[key]
        record["scan_scopes"].add(scan_scope)
        record["represented_files"] += represented_files
        sop_uid = str(dataset.get("SOPInstanceUID") or path)
        if sop_uid in record["sop"]:
            record["duplicates"] += 1
            continue
        record["sop"].add(sop_uid)
        normal, location = canonical_geometry(dataset)
        record["items"].append(
            {
                "normal": normal,
                "location": location,
                "trigger": as_float(dataset.get("TriggerTime")),
                "temporal": as_float(dataset.get("TemporalPositionIdentifier")),
                "instance": as_float(dataset.get("InstanceNumber")),
                "acquisition": str(dataset.get("AcquisitionTime") or ""),
                "number_of_frames": int(dataset.get("NumberOfFrames") or 0),
            }
        )

    series_rows = []
    for (study_uid, series_uid), record in records.items():
        dataset = record["dataset"]
        layers = defaultdict(list)
        for item in record["items"]:
            key = (item["normal"], item["location"])
            layers[key].append(item)
        valid_layers = []
        frame_sources = set()
        frame_counts = []
        for items in layers.values():
            frame_count, source = count_frames(items, min_frames)
            if frame_count >= min_frames:
                valid_layers.append(items)
                frame_sources.add(source)
                frame_counts.append(frame_count)
        view = record["view"]
        modality = record["modality"]
        normals = [item["normal"] for item in record["items"] if item["normal"] is not None]
        dominant_normal = None
        if normals:
            counts = defaultdict(int)
            for normal in normals:
                counts[normal] += 1
            dominant_normal = max(counts, key=counts.get)
        explicit_sax = modality == "CINE" and view == "SAX" and bool(valid_layers)
        geometry_candidate = modality == "CINE" and view == "UNKNOWN" and len(valid_layers) >= min_slices
        decision = "AUTO_CONFIRMED" if explicit_sax else "GEOMETRY_CANDIDATE" if geometry_candidate else "EXCLUDED"
        series_rows.append(
            {
                "cohort": row.cohort,
                "ID": int(row.ID),
                "study_uid": study_uid,
                "series_uid": series_uid,
                "series_number": str(dataset.get("SeriesNumber") or ""),
                "series_description": str(dataset.get("SeriesDescription") or ""),
                "protocol_name": str(dataset.get("ProtocolName") or ""),
                "sequence_name": str(dataset.get("SequenceName") or ""),
                "manufacturer": str(dataset.get("Manufacturer") or ""),
                "modality": modality,
                "modality_reason": record["modality_reason"],
                "named_view": view,
                "unique_images": len(record["items"]),
                "represented_files": record["represented_files"],
                "scan_scope": "full" if "full" in record["scan_scopes"] else "representative",
                "duplicate_images": record["duplicates"],
                "geometric_layers": len(layers),
                "valid_cine_layers": len(valid_layers),
                "median_frames": float(np.median(frame_counts)) if frame_counts else 0,
                "frame_sources": "|".join(sorted(frame_sources)),
                "normal_x": dominant_normal[0] if dominant_normal else np.nan,
                "normal_y": dominant_normal[1] if dominant_normal else np.nan,
                "normal_z": dominant_normal[2] if dominant_normal else np.nan,
                "sax_reference_angle": np.nan,
                "orientation_margin": np.nan,
                "decision": decision,
            }
        )

    prototypes = defaultdict(list)
    for item in series_rows:
        if item["modality"] != "CINE" or item["named_view"] not in {"SAX", "2CH", "3CH", "4CH"}:
            continue
        if pd.isna(item["normal_x"]):
            continue
        prototypes[item["named_view"]].append(
            (item["normal_x"], item["normal_y"], item["normal_z"])
        )
    for item in series_rows:
        if item["modality"] != "CINE" or item["valid_cine_layers"] < min_slices:
            continue
        if pd.isna(item["normal_x"]) or not prototypes["SAX"]:
            continue
        normal = (item["normal_x"], item["normal_y"], item["normal_z"])
        sax_angle = min(angle_degrees(normal, reference) for reference in prototypes["SAX"])
        competing = [
            angle_degrees(normal, reference)
            for view in ("2CH", "3CH", "4CH")
            for reference in prototypes[view]
        ]
        margin = min(competing) - sax_angle if competing else 90.0 - sax_angle
        item["sax_reference_angle"] = round(sax_angle, 4)
        item["orientation_margin"] = round(margin, 4)
        if item["decision"] != "AUTO_CONFIRMED" and sax_angle <= 10.0 and margin >= 15.0:
            item["decision"] = "GEOMETRY_CONFIRMED"
    for item in series_rows:
        if (
            item["decision"] not in {"AUTO_CONFIRMED", "GEOMETRY_CONFIRMED"}
            and item["modality"] == "CINE"
            and item["valid_cine_layers"] >= 6
            and item["named_view"] != "AXIAL"
        ):
            item["decision"] = "STACK_CONFIRMED"

    cine = [item for item in series_rows if item["modality"] == "CINE"]
    confirmed = [item for item in series_rows if item["decision"] == "AUTO_CONFIRMED"]
    geometry_confirmed = [item for item in series_rows if item["decision"] == "GEOMETRY_CONFIRMED"]
    stack_confirmed = [item for item in series_rows if item["decision"] == "STACK_CONFIRMED"]
    geometry = [item for item in series_rows if item["decision"] == "GEOMETRY_CANDIDATE"]
    confirmed_layers = sum(item["valid_cine_layers"] for item in confirmed)
    if confirmed_layers >= min_slices:
        status = "AUTO_CONFIRMED"
    elif geometry_confirmed:
        status = "GEOMETRY_CONFIRMED"
    elif stack_confirmed:
        status = "STACK_CONFIRMED"
    elif geometry:
        status = "GEOMETRY_CANDIDATE"
    elif cine:
        status = "MANUAL_REVIEW"
    else:
        status = "MISSING_CINE"
    patient_row = {
        "cohort": row.cohort,
        "ID": int(row.ID),
        "status": status,
        "readable_series": len(series_rows),
        "cine_series": len(cine),
        "confirmed_sax_series": len(confirmed),
        "confirmed_sax_layers": confirmed_layers,
        "geometry_candidate_series": len(geometry),
        "geometry_candidate_layers": sum(item["valid_cine_layers"] for item in geometry),
        "geometry_confirmed_series": len(geometry_confirmed),
        "geometry_confirmed_layers": sum(item["valid_cine_layers"] for item in geometry_confirmed),
        "stack_confirmed_series": len(stack_confirmed),
        "stack_confirmed_layers": sum(item["valid_cine_layers"] for item in stack_confirmed),
        "unreadable_files": unreadable,
        "source_path": row.image_path,
    }
    return patient_row, series_rows


def main():
    """
    主程序入口。
    加载包含 DICOM 根路径的 manifest，遍历处理患者，输出该分片的审计状态与库存表。
    """
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index必须位于[0, num-shards)范围内")
    manifest = pd.read_csv(args.manifest)
    if args.cohort != "all":
        manifest = manifest.loc[manifest["cohort"] == args.cohort].copy()
    manifest = manifest.iloc[args.shard_index :: args.num_shards].copy()
    if args.max_patients:
        manifest = manifest.head(args.max_patients)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patients = []
    series = []
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        patient, patient_series = scan_patient(row, args.min_frames, args.min_slices)
        patients.append(patient)
        series.extend(patient_series)
        print(f"[{index}/{len(manifest)}] {row.cohort} ID={row.ID}: {patient['status']}", flush=True)

    suffix = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    patient_table = pd.DataFrame(patients)
    series_table = pd.DataFrame(series)
    patient_table.to_csv(args.output_dir / f"patient_sax_status_{suffix}.csv", index=False)
    series_table.to_csv(args.output_dir / f"series_inventory_{suffix}.csv", index=False)
    summary = {
        "patients": len(patient_table),
        "series": len(series_table),
        "status_counts": patient_table["status"].value_counts().to_dict(),
        "min_frames": args.min_frames,
        "min_slices": args.min_slices,
        "read_only": True,
    }
    (args.output_dir / f"summary_{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
