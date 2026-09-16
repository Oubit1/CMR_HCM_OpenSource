#!/usr/bin/env python3
"""Simulate contrast-sparing operating points using development-locked thresholds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = EXPERIMENT_ROOT / "results/final_full_comparison_icc080_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-predictions",
        type=Path,
        default=DEFAULT_RESULTS / "development_predictions.csv",
    )
    parser.add_argument(
        "--external-predictions",
        type=Path,
        default=DEFAULT_RESULTS / "external_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS / "clinical_impact",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "CMR",
            "Rscore+CMR",
            "Rscore+CMR_XGBoost",
            "Foundation_frozen",
            "Foundation_frozen+CMR",
            "Foundation_full_transfer",
            "Foundation_full_transfer+CMR",
            "Foundation_full_transfer+Rscore+CMR",
        ],
    )
    parser.add_argument(
        "--sensitivity-targets", nargs="+", type=float, default=[0.95, 0.975, 0.99]
    )
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_high_sensitivity_threshold(labels, probabilities, target):
    _, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    eligible = np.isfinite(thresholds) & (true_positive_rate >= target)
    if not eligible.any():
        return float(np.nextafter(probabilities.min(), -np.inf))
    return float(np.max(thresholds[eligible]))


def metrics(labels, probabilities, threshold):
    predicted_positive = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(
        labels, predicted_positive, labels=[0, 1]
    ).ravel()
    divide = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    avoided = tn + fn
    return {
        "n": len(labels),
        "event_n": int(labels.sum()),
        "sensitivity": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "npv": divide(tn, tn + fn),
        "contrast_avoided_fraction": divide(avoided, len(labels)),
        "contrast_avoided_per_1000": 1000 * divide(avoided, len(labels)),
        "missed_lge_per_1000_patients": 1000 * divide(fn, len(labels)),
        "missed_lge_per_1000_lge_positive": 1000 * divide(fn, tp + fn),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def bootstrap_intervals(labels, probabilities, threshold, bootstraps, seed):
    rng = np.random.default_rng(seed)
    values = []
    indices = np.arange(len(labels))
    for _ in range(bootstraps):
        sample = rng.choice(indices, size=len(indices), replace=True)
        sample_labels = labels[sample]
        if np.unique(sample_labels).size < 2:
            continue
        values.append(metrics(sample_labels, probabilities[sample], threshold))
    interval_metrics = [
        "sensitivity",
        "specificity",
        "npv",
        "contrast_avoided_per_1000",
        "missed_lge_per_1000_patients",
    ]
    intervals = {}
    for name in interval_metrics:
        samples = np.asarray([row[name] for row in values], dtype=float)
        intervals[f"{name}_ci_lower"] = float(np.nanpercentile(samples, 2.5))
        intervals[f"{name}_ci_upper"] = float(np.nanpercentile(samples, 97.5))
    return intervals


def main():
    args = parse_args()
    development = pd.read_csv(args.development_predictions)
    external = pd.read_csv(args.external_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_index, model in enumerate(args.models):
        column = f"{model}_probability"
        if column not in development or column not in external:
            print(f"Skipping unavailable model: {model}")
            continue
        development_labels = development["event"].to_numpy(int)
        development_probability = development[column].to_numpy(float)
        external_labels = external["event"].to_numpy(int)
        external_probability = external[column].to_numpy(float)
        for target_index, target in enumerate(args.sensitivity_targets):
            threshold = select_high_sensitivity_threshold(
                development_labels, development_probability, target
            )
            for dataset_index, (dataset, labels, probability) in enumerate(
                [
                    ("Development OOF", development_labels, development_probability),
                    ("External test", external_labels, external_probability),
                ]
            ):
                point = metrics(labels, probability, threshold)
                intervals = bootstrap_intervals(
                    labels,
                    probability,
                    threshold,
                    args.bootstraps,
                    args.seed + model_index * 100 + target_index * 10 + dataset_index,
                )
                rows.append(
                    {
                        "model": model,
                        "target_development_sensitivity": target,
                        "locked_threshold": threshold,
                        "dataset": dataset,
                        **point,
                        **intervals,
                    }
                )
    output = pd.DataFrame(rows)
    output.to_csv(args.output_dir / "clinical_operating_points.csv", index=False)
    external = output.loc[output["dataset"] == "External test"].copy()
    external.to_csv(args.output_dir / "external_clinical_impact.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "development_predictions": str(args.development_predictions),
                "external_predictions": str(args.external_predictions),
                "models": args.models,
                "sensitivity_targets": args.sensitivity_targets,
                "bootstraps": args.bootstraps,
                "seed": args.seed,
                "interpretation": "Hypothetical triage simulation; predicted-negative patients are counted as avoiding contrast. Thresholds are selected only in development OOF predictions.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if external.empty:
        raise RuntimeError("No requested model was available")
    best = external.loc[external["target_development_sensitivity"] == 0.95].sort_values(
        ["missed_lge_per_1000_patients", "contrast_avoided_per_1000"],
        ascending=[True, False],
    )
    lines = [
        "# Clinical Operating-Point Analysis",
        "",
        "Thresholds were selected exclusively from development OOF predictions and then locked for external evaluation.",
        "This is a hypothetical contrast-sparing simulation rather than a management recommendation.",
        "",
        "## External test at development sensitivity target 0.95",
        "",
    ]
    for row in best.to_dict("records"):
        lines.append(
            f"- {row['model']}: sensitivity {row['sensitivity']:.3f}, specificity {row['specificity']:.3f}, "
            f"contrast avoided {row['contrast_avoided_per_1000']:.1f}/1000, "
            f"missed LGE {row['missed_lge_per_1000_patients']:.1f}/1000."
        )
    (args.output_dir / "CLINICAL_IMPACT_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
