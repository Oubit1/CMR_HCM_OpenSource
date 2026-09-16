#!/usr/bin/env python3
"""Compare full-cine and repeated-ED inputs on identical patients."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-cine-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results/foundation_attention_mil_full_transfer_v1",
    )
    parser.add_argument(
        "--ed-repeat-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results/foundation_attention_mil_ed_repeat_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results/temporal_ablation_v1",
    )
    parser.add_argument("--bootstraps", type=int, default=2000)
    return parser.parse_args()


def load_predictions(directory, dataset, prefix):
    filename = (
        "development_oof_predictions.csv"
        if dataset == "development"
        else "external_test_predictions.csv"
    )
    table = pd.read_csv(directory / "Foundation" / filename)
    rename = {
        "probability": f"{prefix}_probability",
        "Foundation+CMR_probability": f"{prefix}+CMR_probability",
        "Foundation+Rscore+CMR_probability": f"{prefix}+Rscore+CMR_probability",
    }
    return table[["ID", "event", *rename]].rename(columns=rename)


def merge_predictions(args, dataset):
    full = load_predictions(args.full_cine_dir, dataset, "Full_cine")
    ed = load_predictions(args.ed_repeat_dir, dataset, "ED_repeat")
    merged = full.merge(ed, on=["ID", "event"], validate="one_to_one")
    expected = 1050 if dataset == "development" else 309
    if len(merged) != expected:
        raise RuntimeError(f"Expected {expected} {dataset} patients, got {len(merged)}")
    return merged.sort_values("ID").reset_index(drop=True)


def thresholds(directory, prefix):
    table = pd.read_csv(directory / "model_comparison.csv")
    output = {}
    for row in table.to_dict("records"):
        suffix = row["arm"].removeprefix("Foundation")
        output[f"{prefix}{suffix}"] = float(row["external_threshold"])
    return output


def operating_metrics(labels, probabilities, threshold):
    prediction = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    divide = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    return {
        "auc": roc_auc_score(labels, probabilities),
        "brier": brier_score_loss(labels, probabilities),
        "threshold": threshold,
        "sensitivity": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "npv": divide(tn, tn + fn),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def main():
    args = parse_args()
    if not (args.ed_repeat_dir / "model_comparison.csv").exists():
        raise FileNotFoundError("ED-repeat experiment has not completed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development = merge_predictions(args, "development")
    external = merge_predictions(args, "external")
    development_path = args.output_dir / "development_predictions.csv"
    external_path = args.output_dir / "external_predictions.csv"
    development.to_csv(development_path, index=False)
    external.to_csv(external_path, index=False)

    threshold_map = {
        **thresholds(args.full_cine_dir, "Full_cine"),
        **thresholds(args.ed_repeat_dir, "ED_repeat"),
    }
    rows = []
    for dataset, table in [
        ("Development OOF", development),
        ("External test", external),
    ]:
        labels = table["event"].to_numpy(int)
        for column in [name for name in table if name.endswith("_probability")]:
            model = column.removesuffix("_probability")
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "n": len(table),
                    **operating_metrics(
                        labels,
                        table[column].to_numpy(float),
                        threshold_map[model],
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(args.output_dir / "operating_metrics.csv", index=False)

    statistics_dir = args.output_dir / "additional_statistics"
    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_ROOT / "run_additional_statistics.py"),
            "--development-predictions",
            str(development_path),
            "--external-predictions",
            str(external_path),
            "--output-dir",
            str(statistics_dir),
            "--bootstraps",
            str(args.bootstraps),
        ],
        check=True,
    )
    delong = pd.read_csv(statistics_dir / "delong_pairwise.csv")
    key_pairs = []
    for suffix in ["", "+CMR", "+Rscore+CMR"]:
        full_name = f"Full_cine{suffix}"
        ed_name = f"ED_repeat{suffix}"
        selected = delong.loc[
            (
                (delong["Reference model"] == full_name)
                & (delong["Comparison model"] == ed_name)
            )
            | (
                (delong["Reference model"] == ed_name)
                & (delong["Comparison model"] == full_name)
            )
        ].copy()
        key_pairs.append(selected)
    key_pairs = pd.concat(key_pairs, ignore_index=True)
    key_pairs.to_csv(args.output_dir / "key_paired_delong.csv", index=False)

    metrics = pd.DataFrame(rows)
    lines = [
        "# Full-Cine vs Repeated-ED Temporal Ablation",
        "",
        "The same patients, architecture, hyperparameters, and validation policy were used; only temporal input differed.",
        "",
    ]
    for dataset in ["Development OOF", "External test"]:
        lines.append(f"## {dataset}")
        for suffix in ["", "+CMR", "+Rscore+CMR"]:
            full = metrics.loc[
                (metrics["dataset"] == dataset)
                & (metrics["model"] == f"Full_cine{suffix}")
            ].iloc[0]
            ed = metrics.loc[
                (metrics["dataset"] == dataset)
                & (metrics["model"] == f"ED_repeat{suffix}")
            ].iloc[0]
            pair = key_pairs.loc[
                (key_pairs["Dataset"] == dataset)
                & key_pairs["Reference model"].isin([f"Full_cine{suffix}", f"ED_repeat{suffix}"])
                & key_pairs["Comparison model"].isin([f"Full_cine{suffix}", f"ED_repeat{suffix}"])
            ].iloc[0]
            lines.append(
                f"- {suffix.removeprefix('+') or 'Foundation only'}: full-cine AUC {full.auc:.4f}, "
                f"ED-repeat AUC {ed.auc:.4f}, paired DeLong P={pair['P value']:.4g}."
            )
        lines.append("")
    (args.output_dir / "TEMPORAL_ABLATION_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "full_cine_dir": str(args.full_cine_dir),
                "ed_repeat_dir": str(args.ed_repeat_dir),
                "comparison_policy": "identical complete-case patients; full cine versus first/ED phase repeated to 16 frames",
                "bootstraps": args.bootstraps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
