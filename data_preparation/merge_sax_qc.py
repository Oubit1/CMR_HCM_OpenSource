#!/usr/bin/env python3
"""合并 SAX NIfTI 转换分片并执行最终一致性检查。"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def key_set(table, columns):
    return set(map(tuple, table[columns].to_numpy()))


def main():
    """
    主程序：合并多个并行的 NIfTI 转换分片 (shards) 的结果。
    执行严格的一致性检查，包括：是否覆盖全集、是否有重复生成、是否有未完成的临时文件、以及切片与患者层面的统计是否对齐。
    """
    args = parse_args()
    patient_files = sorted(args.output_dir.glob("patient_qc_shard_*.csv"))
    slice_files = sorted(args.output_dir.glob("slice_qc_shard_*.csv"))
    metadata_files = sorted(args.output_dir.glob("run_metadata_shard_*.json"))
    if not patient_files or len(patient_files) != len(slice_files):
        raise RuntimeError("患者 QC 与切片 QC 分片数量不完整")

    patients = pd.concat([pd.read_csv(path) for path in patient_files], ignore_index=True)
    slices = pd.concat([pd.read_csv(path) for path in slice_files], ignore_index=True)
    manifest = pd.read_csv(args.manifest)
    for table in (patients, slices, manifest):
        table["ID"] = table["ID"].astype(int)

    if patients.duplicated(["cohort", "ID"]).any():
        raise RuntimeError("患者 QC 中存在重复患者")
    if slices.duplicated(["cohort", "ID", "slice_index"]).any():
        raise RuntimeError("切片 QC 中存在重复切片")
    if key_set(patients, ["cohort", "ID"]) != key_set(manifest, ["cohort", "ID"]):
        raise RuntimeError("患者 QC 未完整覆盖 manifest")

    expected_paths = {str(Path(path).resolve()) for path in slices["output_path"]}
    actual_paths = {
        str(path.resolve())
        for path in args.output_dir.glob("*/images/cmr/*.nii.gz")
        if not path.name.endswith(".tmp.nii.gz")
    }
    temporary_paths = list(args.output_dir.glob("*/images/cmr/*.tmp.nii.gz"))
    if expected_paths != actual_paths:
        raise RuntimeError(
            f"NIfTI 与切片 QC 不一致: missing={len(expected_paths - actual_paths)}, "
            f"unexpected={len(actual_paths - expected_paths)}"
        )
    if temporary_paths:
        raise RuntimeError(f"发现未完成的临时 NIfTI: {len(temporary_paths)}")
    if int(patients["saved_slices"].sum()) != len(slices):
        raise RuntimeError("患者切片计数与切片 QC 行数不一致")
    if not patients["conversion_status"].isin(["ok", "skipped_unresolved"]).all():
        raise RuntimeError("仍有转换失败患者")
    if (slices["frame_count"] < 12).any():
        raise RuntimeError("存在少于 12 帧的 NIfTI")

    patients = patients.sort_values(["cohort", "ID"]).reset_index(drop=True)
    slices = slices.sort_values(["cohort", "ID", "slice_index"]).reset_index(drop=True)
    patients.to_csv(args.output_dir / "patient_qc.csv", index=False)
    slices.to_csv(args.output_dir / "slice_qc.csv", index=False)
    summary = {
        "patient_shards": len(patient_files),
        "metadata_shards": len(metadata_files),
        "patients": len(patients),
        "patients_completed": int(patients["conversion_status"].eq("ok").sum()),
        "patients_skipped_unresolved": int(
            patients["conversion_status"].eq("skipped_unresolved").sum()
        ),
        "patients_failed": int(patients["conversion_status"].eq("failed").sum()),
        "nifti_files": len(actual_paths),
        "frame_count_min": int(slices["frame_count"].min()),
        "frame_count_max": int(slices["frame_count"].max()),
        "manifest_coverage_complete": True,
        "duplicate_patients": 0,
        "duplicate_slices": 0,
        "missing_files": 0,
        "temporary_files": 0,
    }
    (args.output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
