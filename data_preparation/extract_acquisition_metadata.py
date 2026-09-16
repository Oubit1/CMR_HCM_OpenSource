#!/usr/bin/env python3
"""Extract auditable patient-level acquisition metadata from selected SAX DICOM series."""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm


warnings.filterwarnings("ignore", module="pydicom")

# ==============================================================================
# 模块一：数据预处理 - 提取扫描设备与技术采集参数 (extract_acquisition_metadata.py)
# ==============================================================================

# 默认路径配置 (可由命令行参数覆盖)
DEFAULT_PROCESSED_ROOT = Path("./data/processed_sax_cine")          # 处理后的短轴 cine NIfTI 根目录
DEFAULT_OUTPUT = Path("./results/acquisition_metadata.csv")         # 提取的设备与参数元数据 CSV

TAGS = [
    "SeriesInstanceUID",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "TemporalResolution",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
    "Rows",
    "Columns",
    "InstitutionName",
    "StationName",
    "SoftwareVersions",
    "SeriesDescription",
    "ProtocolName",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def first_float(value):
    if value is None:
        return np.nan
    try:
        if isinstance(value, (list, tuple)) or hasattr(value, "__iter__"):
            return float(list(value)[0])
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def clean_text(value):
    text = str(value or "").strip()
    return text if text else np.nan


def read_header(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pydicom.dcmread(
                path,
                stop_before_pixels=True,
                force=True,
                specific_tags=TAGS,
            )
    except Exception:
        return None


def directory_samples(source_path):
    for root, _, filenames in os.walk(source_path):
        if not filenames:
            continue
        path = Path(root) / sorted(filenames)[0]
        if path.is_file():
            yield path


def find_series_header(source_path, series_uid):
    attempted = 0
    sampled = set()
    for path in directory_samples(source_path):
        sampled.add(path)
        attempted += 1
        dataset = read_header(path)
        if dataset is None:
            continue
        if str(getattr(dataset, "SeriesInstanceUID", "")) == series_uid:
            return dataset, path, attempted
    for root, _, filenames in os.walk(source_path):
        for filename in sorted(filenames):
            path = Path(root) / filename
            if path in sampled or not path.is_file():
                continue
            attempted += 1
            dataset = read_header(path)
            if dataset is None:
                continue
            if str(getattr(dataset, "SeriesInstanceUID", "")) == series_uid:
                return dataset, path, attempted
    return None, None, attempted


def metadata_row(patient, slices):
    dominant_uid = str(slices["series_uid"].value_counts().index[0])
    selected = slices.loc[slices["series_uid"].astype(str) == dominant_uid]
    dataset, path, attempted = find_series_header(
        Path(patient["source_path"]), dominant_uid
    )
    base = {
        "cohort": patient["cohort"],
        "ID": int(patient["ID"]),
        "event": int(patient["event"]),
        "source_path": patient["source_path"],
        "series_uid": dominant_uid,
        "representative_dicom": str(path) if path else np.nan,
        "headers_attempted": attempted,
        "metadata_status": "ok" if dataset is not None else "series_not_found",
        "sax_slices": int(len(slices)),
        "dominant_series_slices": int(len(selected)),
        "median_frames": float(selected["frame_count"].median()),
        "median_height": float(selected["height"].median()),
        "median_width": float(selected["width"].median()),
    }
    if dataset is None:
        return base
    pixel_spacing = getattr(dataset, "PixelSpacing", None)
    base.update(
        {
            "manufacturer": clean_text(getattr(dataset, "Manufacturer", None)),
            "manufacturer_model": clean_text(
                getattr(dataset, "ManufacturerModelName", None)
            ),
            "institution": clean_text(getattr(dataset, "InstitutionName", None)),
            "station_name": clean_text(getattr(dataset, "StationName", None)),
            "software_versions": clean_text(
                getattr(dataset, "SoftwareVersions", None)
            ),
            "series_description": clean_text(
                getattr(dataset, "SeriesDescription", None)
            ),
            "protocol_name": clean_text(getattr(dataset, "ProtocolName", None)),
            "field_strength_t": as_float(
                getattr(dataset, "MagneticFieldStrength", None)
            ),
            "pixel_spacing_row_mm": first_float(pixel_spacing),
            "pixel_spacing_col_mm": (
                as_float(pixel_spacing[1])
                if pixel_spacing is not None and len(pixel_spacing) > 1
                else first_float(pixel_spacing)
            ),
            "slice_thickness_mm": as_float(
                getattr(dataset, "SliceThickness", None)
            ),
            "spacing_between_slices_mm": as_float(
                getattr(dataset, "SpacingBetweenSlices", None)
            ),
            "temporal_resolution_ms": as_float(
                getattr(dataset, "TemporalResolution", None)
            ),
            "repetition_time_ms": as_float(
                getattr(dataset, "RepetitionTime", None)
            ),
            "echo_time_ms": as_float(getattr(dataset, "EchoTime", None)),
            "flip_angle_deg": as_float(getattr(dataset, "FlipAngle", None)),
            "rows": as_float(getattr(dataset, "Rows", None)),
            "columns": as_float(getattr(dataset, "Columns", None)),
        }
    )
    return base


def main():
    args = parse_args()
    patient_qc = pd.read_csv(args.processed_root / "patient_qc.csv")
    slice_qc = pd.read_csv(args.processed_root / "slice_qc.csv")
    patient_qc = patient_qc.loc[patient_qc["conversion_status"] == "ok"].copy()
    existing = pd.DataFrame()
    if args.resume and args.output.exists():
        existing = pd.read_csv(args.output)
        completed_rows = existing.loc[existing["metadata_status"] == "ok"]
        completed = set(
            zip(completed_rows["cohort"], completed_rows["ID"].astype(int))
        )
        existing = completed_rows.copy()
        patient_qc = patient_qc.loc[
            ~patient_qc.apply(
                lambda row: (row["cohort"], int(row["ID"])) in completed, axis=1
            )
        ]
    if args.max_patients:
        patient_qc = patient_qc.head(args.max_patients)
    grouped = {
        key: table
        for key, table in slice_qc.groupby(["cohort", "ID"], sort=False)
    }
    rows = []
    for _, patient in tqdm(
        patient_qc.iterrows(), total=len(patient_qc), desc="DICOM metadata"
    ):
        key = (patient["cohort"], int(patient["ID"]))
        if key not in grouped:
            rows.append(
                {
                    "cohort": key[0],
                    "ID": key[1],
                    "event": int(patient["event"]),
                    "source_path": patient["source_path"],
                    "metadata_status": "no_converted_slices",
                }
            )
            continue
        rows.append(metadata_row(patient, grouped[key]))
    result = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    result = result.sort_values(["cohort", "ID"]).drop_duplicates(
        ["cohort", "ID"], keep="last"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = {
        "patients": len(result),
        "metadata_ok": int((result["metadata_status"] == "ok").sum()),
        "metadata_failed": int((result["metadata_status"] != "ok").sum()),
        "field_strength_counts": result["field_strength_t"]
        .value_counts(dropna=False)
        .to_dict(),
        "manufacturer_counts": result["manufacturer"]
        .value_counts(dropna=False)
        .to_dict(),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
