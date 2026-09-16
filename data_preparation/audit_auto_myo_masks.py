#!/usr/bin/env python3
"""Audit automatically generated myocardium masks without reference annotations."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


# ==============================================================================
# 模块一：数据预处理 - 自动心肌分割掩膜完整性与体积质控审计 (audit_auto_myo_masks.py)
# ==============================================================================

# 默认路径配置 (可由命令行参数覆盖)
DEFAULT_IMAGE_ROOT = Path("./data/processed_sax_cine")              # 处理后的短轴 cine NIfTI 根目录
DEFAULT_MASK_ROOT = Path("./data/auto_myo_masks")                   # 自动心肌分割生成的掩膜目录
DEFAULT_OUTPUT = Path("./results/auto_myo_masks_qc")                # 质控审计输出结果目录



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def time_first(array):
    array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"expected 3-D data, got {array.shape}")
    minimum = min(array.shape)
    candidates = [index for index, size in enumerate(array.shape) if size == minimum]
    if len(candidates) != 1:
        raise ValueError(f"cannot infer time axis from {array.shape}")
    return np.moveaxis(array, candidates[0], 0)


def temporal_dice(mask):
    if len(mask) < 2:
        return np.nan
    first, second = mask[:-1], mask[1:]
    intersection = np.logical_and(first, second).sum(axis=(1, 2))
    denominator = first.sum(axis=(1, 2)) + second.sum(axis=(1, 2))
    return float(np.median((2 * intersection + 1) / (denominator + 1)))


def audit_pair(image_path, mask_path, relative):
    image = time_first(np.asanyarray(nib.load(image_path).dataobj))
    mask = time_first(np.asanyarray(nib.load(mask_path).dataobj)) > 0
    if image.shape != mask.shape:
        raise ValueError(f"shape mismatch: image={image.shape}, mask={mask.shape}")
    areas = mask.sum(axis=(1, 2))
    fractions = mask.mean(axis=(1, 2))
    components = np.asarray(
        [ndimage.label(frame)[1] for frame in mask], dtype=int
    )
    empty_frames = int((areas == 0).sum())
    median_fraction = float(np.median(fractions))
    median_temporal_dice = temporal_dice(mask)
    median_components = float(np.median(components))
    flags = []
    if empty_frames:
        flags.append("empty_frame")
    if median_fraction < 0.002:
        flags.append("very_small_mask")
    if median_fraction > 0.30:
        flags.append("very_large_mask")
    if median_temporal_dice < 0.50:
        flags.append("low_temporal_consistency")
    if median_components > 2:
        flags.append("fragmented_mask")
    parts = Path(relative).parts
    cohort = parts[0] if parts else "unknown"
    name = Path(relative).name
    patient = name.split("_slice")[0]
    return {
        "cohort": cohort,
        "patient": patient,
        "relative_path": relative,
        "frames": len(mask),
        "height": mask.shape[1],
        "width": mask.shape[2],
        "empty_frames": empty_frames,
        "median_area_pixels": float(np.median(areas)),
        "minimum_area_pixels": int(areas.min()),
        "maximum_area_pixels": int(areas.max()),
        "median_area_fraction": median_fraction,
        "median_connected_components": median_components,
        "maximum_connected_components": int(components.max()),
        "median_consecutive_frame_dice": median_temporal_dice,
        "qc_flags": ";".join(flags),
        "qc_pass": not flags,
    }


def main():
    args = parse_args()
    masks = sorted(
        path
        for path in args.mask_root.rglob("*")
        if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))
    )
    if not masks:
        raise FileNotFoundError(f"no masks found under {args.mask_root}")
    rows, failures = [], []
    for mask_path in tqdm(masks, desc="Mask QC"):
        relative = mask_path.relative_to(args.mask_root)
        image_path = args.image_root / relative
        try:
            rows.append(audit_pair(image_path, mask_path, str(relative)))
        except Exception as error:
            failures.append(
                {
                    "relative_path": str(relative),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "error": repr(error),
                }
            )
    result = pd.DataFrame(rows)
    failure_table = pd.DataFrame(failures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "mask_volume_qc.csv", index=False)
    failure_table.to_csv(args.output_dir / "mask_qc_failures.csv", index=False)
    patient = (
        result.groupby(["cohort", "patient"])
        .agg(
            slices=("relative_path", "size"),
            failed_slices=("qc_pass", lambda values: int((~values).sum())),
            empty_frames=("empty_frames", "sum"),
            minimum_temporal_dice=("median_consecutive_frame_dice", "min"),
            median_area_fraction=("median_area_fraction", "median"),
        )
        .reset_index()
    )
    patient["qc_pass"] = patient["failed_slices"] == 0
    patient.to_csv(args.output_dir / "mask_patient_qc.csv", index=False)
    summary = {
        "mask_volumes": len(result),
        "patients": len(patient),
        "volume_qc_pass": int(result["qc_pass"].sum()),
        "volume_qc_flagged": int((~result["qc_pass"]).sum()),
        "patient_qc_pass": int(patient["qc_pass"].sum()),
        "patient_qc_flagged": int((~patient["qc_pass"]).sum()),
        "read_failures": len(failures),
        "empty_frames": int(result["empty_frames"].sum()),
        "median_temporal_dice": float(
            result["median_consecutive_frame_dice"].median()
        ),
        "median_mask_area_fraction": float(result["median_area_fraction"].median()),
        "limitations": (
            "These reference-free checks detect gross failures only. They do not "
            "establish Dice accuracy against manual myocardium ROIs."
        ),
    }
    (args.output_dir / "mask_qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
