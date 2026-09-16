#!/usr/bin/env python3
"""依据冻结的 SAX 审计清单，将 DICOM 转换为逐切片 cine NIfTI。"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom


# ==============================================================================
# 模块一：数据预处理 - DICOM 转换为切片级 SAX Cine NIfTI (prepare_sax_nifti.py)
# ==============================================================================

# 默认路径配置 (可由命令行参数覆盖)
DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = Path("./data/patient_manifest.csv")               # 患者清单路径
DEFAULT_AUDIT_DIR = Path("./results/sax_retrieval_audit")           # 序列审计结果目录
DEFAULT_OUTPUT = Path("./data/processed_sax_cine")                  # 处理后的 NIfTI 存放目录

SELECTED_DECISIONS = {
    "AUTO_CONFIRMED",
    "GEOMETRY_CONFIRMED",
    "STACK_CONFIRMED",
    "FAMILY_STACK_CONFIRMED",
    "MANUAL_CONFIRMED",
}
HEADER_TAGS = [
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Rows",
    "Columns",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "SliceLocation",
    "TriggerTime",
    "TemporalPositionIdentifier",
    "InstanceNumber",
    "AcquisitionTime",
    "NumberOfFrames",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cohort", choices=["all", "development", "external_test"], default="all"
    )
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reference_affine():
    return np.asarray(
        [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )


def raw_files(directory):
    ignored = {".py", ".mat", ".nii", ".gz", ".xlsx", ".csv", ".json"}
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return [
        directory / name
        for name in names
        if (directory / name).is_file()
        and name.upper() != "DICOMDIR"
        and Path(name).suffix.lower() not in ignored
    ]


def read_header(path):
    return pydicom.dcmread(
        path,
        stop_before_pixels=True,
        defer_size="1 KB",
        force=True,
        specific_tags=HEADER_TAGS,
    )


def selected_files(patient_path, selected_uids):
    direct = raw_files(patient_path)
    if direct:
        yield from direct
        return
    for root, _, _ in os.walk(patient_path):
        directory = Path(root)
        files = raw_files(directory)
        if not files:
            continue
        sample = None
        for path in files:
            try:
                sample = read_header(path)
            except Exception:
                continue
            if sample.get("Rows") and sample.get("Columns"):
                break
        if sample is None:
            continue
        uid = str(sample.get("SeriesInstanceUID") or "")
        if uid in selected_uids:
            yield from files


def canonical_coordinate(dataset):
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
            coordinate = float(np.dot(normal, np.asarray(position, dtype=float)))
            return round(coordinate * 2) / 2
        except (TypeError, ValueError):
            pass
    try:
        return round(float(dataset.get("SliceLocation")) * 2) / 2
    except (TypeError, ValueError):
        return 0.0


def numeric_value(dataset, keyword):
    try:
        return float(dataset.get(keyword))
    except (TypeError, ValueError):
        return None


def time_value(record, index):
    return record[("trigger", "temporal", "instance", "acquisition")[index]]


def ordered_unique_frames(records, min_frames):
    sources = ("TriggerTime", "TemporalPositionIdentifier", "InstanceNumber", "AcquisitionTime")
    for index, source in enumerate(sources):
        valid = [time_value(record, index) for record in records]
        valid = [value for value in valid if value not in (None, "")]
        if len(set(valid)) < min_frames:
            continue
        unique = {}
        ordered = sorted(
            records,
            key=lambda item: (
                time_value(item, index) in (None, ""),
                time_value(item, index) if time_value(item, index) not in (None, "") else 0,
            ),
        )
        for record in ordered:
            value = time_value(record, index)
            if value not in (None, ""):
                unique.setdefault(value, record["path"])
        if len(unique) >= min_frames:
            return list(unique.values()), source
    return [], "none"


def collect_cine_groups(patient_path, selected_uids, min_frames):
    """
    在指定患者的 DICOM 目录下收集属于选中序列(selected_uids)的短轴电影(cine)数据。
    按几何切面位置(coordinate)将图像分组，并检查是否满足最低帧数要求。
    """
    records = defaultdict(list)
    seen_sop = set()
    unreadable = 0
    for path in selected_files(patient_path, selected_uids):
        try:
            dataset = read_header(path)
        except Exception:
            unreadable += 1
            continue
        if not dataset.get("Rows") or not dataset.get("Columns"):
            continue
        uid = str(dataset.get("SeriesInstanceUID") or "")
        if uid not in selected_uids:
            continue
        sop = str(dataset.get("SOPInstanceUID") or path)
        if sop in seen_sop:
            continue
        seen_sop.add(sop)
        records[(uid, canonical_coordinate(dataset))].append(
            {
                "path": path,
                "trigger": numeric_value(dataset, "TriggerTime"),
                "temporal": numeric_value(dataset, "TemporalPositionIdentifier"),
                "instance": numeric_value(dataset, "InstanceNumber"),
                "acquisition": str(dataset.get("AcquisitionTime") or ""),
                "number_of_frames": int(dataset.get("NumberOfFrames") or 0),
            }
        )

    groups = []
    found_uids = set()
    for (uid, coordinate), layer_records in records.items():
        multiframe = max(layer_records, key=lambda item: item["number_of_frames"])
        if multiframe["number_of_frames"] >= min_frames:
            frame_paths = [multiframe["path"]]
            frame_source = "NumberOfFrames"
            frame_count = multiframe["number_of_frames"]
            is_multiframe = True
        else:
            frame_paths, frame_source = ordered_unique_frames(layer_records, min_frames)
            frame_count = len(frame_paths)
            is_multiframe = False
        if frame_count < min_frames:
            continue
        found_uids.add(uid)
        groups.append(
            {
                "series_uid": uid,
                "slice_coordinate": coordinate,
                "frame_paths": frame_paths,
                "frame_count": frame_count,
                "frame_source": frame_source,
                "is_multiframe": is_multiframe,
            }
        )
    return groups, found_uids, unreadable


def choose_unique_layers(groups):
    chosen = {}
    for group in groups:
        coordinate = float(group["slice_coordinate"])
        current = chosen.get(coordinate)
        if current is None or group["frame_count"] > current["frame_count"]:
            chosen[coordinate] = group
    return [chosen[coordinate] for coordinate in sorted(chosen)]


def scaled_pixels(dataset):
    image = np.asarray(dataset.pixel_array, dtype=np.float32)
    slope = float(dataset.get("RescaleSlope", 1.0) or 1.0)
    intercept = float(dataset.get("RescaleIntercept", 0.0) or 0.0)
    return image * slope + intercept


def load_group(group):
    if group["is_multiframe"]:
        cine = scaled_pixels(pydicom.dcmread(group["frame_paths"][0], force=True))
        if cine.ndim != 3:
            raise ValueError(f"多帧 DICOM 维度异常: {cine.shape}")
        return cine.astype(np.float32, copy=False)
    frames = [scaled_pixels(pydicom.dcmread(path, force=True)) for path in group["frame_paths"]]
    if any(frame.ndim != 2 for frame in frames):
        raise ValueError("单帧 DICOM 中出现非二维像素数据")
    return np.stack(frames).astype(np.float32, copy=False)


def save_group(group, output_path):
    """
    将同一几何切面的 DICOM 帧序列保存为 3D NIfTI 文件 (*.nii.gz)。
    保留原始 DICOM 的 rescale slope/intercept 强度缩放。
    """
    cine = load_group(group)
    image = nib.Nifti1Image(cine, reference_affine())
    image.header.set_data_dtype(np.float32)
    image.header["descrip"] = b"SAX cine layout=T,H,W; raw rescaled intensity"
    temporary_path = output_path.with_name(output_path.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(image, temporary_path)
    temporary_path.replace(output_path)
    return tuple(int(value) for value in cine.shape)


def validate_inputs(manifest, patients, inventory):
    manifest_keys = set(map(tuple, manifest[["cohort", "ID"]].to_numpy()))
    patient_keys = set(map(tuple, patients[["cohort", "ID"]].to_numpy()))
    if manifest_keys != patient_keys:
        raise ValueError(
            f"审计患者与清单不一致: missing={len(manifest_keys - patient_keys)}, "
            f"unexpected={len(patient_keys - manifest_keys)}"
        )
    selected = inventory.loc[inventory["decision"].isin(SELECTED_DECISIONS)]
    selected_keys = set(map(tuple, selected[["cohort", "ID"]].drop_duplicates().to_numpy()))
    expected_keys = set(
        map(
            tuple,
            patients.loc[patients["status"].isin(SELECTED_DECISIONS), ["cohort", "ID"]].to_numpy(),
        )
    )
    if selected_keys != expected_keys:
        raise ValueError("患者状态与选中序列不一致，请重新运行 merge_sax_audit.py")


def main():
    """
    主程序：根据已锁定的 SAX 序列清单(audit 结果)，将对应的原始 DICOM
    并行安全地转换为 NIfTI 格式，并输出每位患者与每个切片的质控(QC)日志。
    """
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index 必须位于 [0, num-shards) 范围内")
    warnings.filterwarnings("ignore", message="Invalid value for VR UI")
    manifest = pd.read_csv(args.manifest)
    patients = pd.read_csv(args.audit_dir / "patient_sax_status.csv")
    inventory = pd.read_csv(
        args.audit_dir / "series_inventory.csv", dtype={"series_uid": str}, low_memory=False
    )
    for table in (manifest, patients, inventory):
        table["ID"] = table["ID"].astype(int)
    validate_inputs(manifest, patients, inventory)

    manifest = manifest.merge(
        patients[["cohort", "ID", "status"]], on=["cohort", "ID"], validate="one_to_one"
    )
    if args.cohort != "all":
        manifest = manifest.loc[manifest["cohort"].eq(args.cohort)].copy()
    manifest = manifest.iloc[args.shard_index :: args.num_shards].copy()
    if args.max_patients > 0:
        manifest = manifest.head(args.max_patients).copy()

    patient_rows = []
    slice_rows = []
    file_entries = {"development": [], "external_test": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        cohort = str(row.cohort)
        patient_id = int(row.ID)
        selected = inventory.loc[
            inventory["cohort"].eq(cohort)
            & inventory["ID"].eq(patient_id)
            & inventory["decision"].isin(SELECTED_DECISIONS)
        ].copy()
        selected_uids = set(selected["series_uid"])
        if not selected_uids:
            patient_rows.append(
                {
                    "cohort": cohort,
                    "ID": patient_id,
                    "event": int(row.event),
                    "audit_status": row.status,
                    "conversion_status": "skipped_unresolved",
                    "selected_series": 0,
                    "found_series": 0,
                    "missing_series": "",
                    "saved_slices": 0,
                    "frame_counts": "[]",
                    "spatial_shapes": "[]",
                    "unreadable_files": 0,
                    "reason": "审计未确认完整 SAX cine",
                    "source_path": str(row.image_path),
                }
            )
            print(
                f"[{number}/{len(manifest)}] {cohort} ID={patient_id}: skipped ({row.status})",
                flush=True,
            )
            continue

        groups, found_uids, unreadable = collect_cine_groups(
            Path(row.image_path), selected_uids, args.min_frames
        )
        groups = choose_unique_layers(groups)
        image_dir = args.output_dir / cohort / "images" / "cmr"
        image_dir.mkdir(parents=True, exist_ok=True)
        shapes = []
        error = ""
        for slice_index, group in enumerate(groups, start=1):
            output_path = image_dir / f"patient{patient_id:04d}_slice{slice_index:02d}.nii.gz"
            try:
                if args.overwrite or not output_path.exists():
                    shape = save_group(group, output_path)
                else:
                    shape = tuple(int(value) for value in nib.load(output_path).shape)
                shapes.append(shape)
                file_entries[cohort].append(str(output_path.resolve()))
                slice_rows.append(
                    {
                        "cohort": cohort,
                        "ID": patient_id,
                        "slice_index": slice_index,
                        "series_uid": group["series_uid"],
                        "slice_coordinate": group["slice_coordinate"],
                        "frame_count": shape[0],
                        "height": shape[1],
                        "width": shape[2],
                        "frame_source": group["frame_source"],
                        "output_path": str(output_path.resolve()),
                    }
                )
            except Exception as exception:
                error = repr(exception)
                break

        missing_uids = sorted(selected_uids - found_uids)
        if error:
            conversion_status = "failed"
        elif not shapes:
            conversion_status = "failed"
            error = "no_valid_selected_sax_cine"
        elif missing_uids:
            conversion_status = "failed"
            error = "selected_series_missing_or_invalid"
        else:
            conversion_status = "ok"
        patient_rows.append(
            {
                "cohort": cohort,
                "ID": patient_id,
                "event": int(row.event),
                "audit_status": row.status,
                "conversion_status": conversion_status,
                "selected_series": len(selected_uids),
                "found_series": len(found_uids),
                "missing_series": "|".join(missing_uids),
                "saved_slices": len(shapes),
                "frame_counts": json.dumps([shape[0] for shape in shapes]),
                "spatial_shapes": json.dumps([list(shape[1:]) for shape in shapes]),
                "unreadable_files": unreadable,
                "reason": error,
                "source_path": str(row.image_path),
            }
        )
        print(
            f"[{number}/{len(manifest)}] {cohort} ID={patient_id}: "
            f"status={conversion_status}, series={len(found_uids)}/{len(selected_uids)}, "
            f"slices={len(shapes)}",
            flush=True,
        )

    suffix = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    patient_table = pd.DataFrame(patient_rows)
    slice_table = pd.DataFrame(slice_rows)
    patient_table.to_csv(args.output_dir / f"patient_qc_{suffix}.csv", index=False)
    slice_table.to_csv(args.output_dir / f"slice_qc_{suffix}.csv", index=False)
    for cohort, entries in file_entries.items():
        if entries:
            (args.output_dir / cohort / f"cine_files_{suffix}.txt").write_text(
                "\n".join(entries) + "\n", encoding="utf-8"
            )
    summary = {
        "layout": "T,H,W",
        "intensity": "raw DICOM pixels after rescale slope/intercept",
        "selection_source": str((args.audit_dir / "series_inventory.csv").resolve()),
        "patients_requested": len(manifest),
        "patients_completed": int(patient_table["conversion_status"].eq("ok").sum()),
        "patients_skipped_unresolved": int(
            patient_table["conversion_status"].eq("skipped_unresolved").sum()
        ),
        "patients_failed": int(patient_table["conversion_status"].eq("failed").sum()),
        "nifti_files": int(patient_table["saved_slices"].sum()),
        "min_frames": args.min_frames,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (args.output_dir / f"run_metadata_{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
