#!/usr/bin/env python3
"""Evaluate model performance across available demographic and acquisition subgroups."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_RESULTS = EXPERIMENT_ROOT / "results/final_full_comparison_icc080_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--acquisition-metadata",
        type=Path,
        default=EXPERIMENT_ROOT
        / "results/reviewer_supplement_v1/acquisition_metadata.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS / "subgroup_analysis"
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
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-group-size", type=int, default=30)
    return parser.parse_args()


def operating_metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    divide = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    return {
        "auc": roc_auc_score(labels, probabilities),
        "sensitivity": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "npv": divide(tn, tn + fn),
    }


def bootstrap_auc(labels, probabilities, bootstraps, rng):
    estimates = []
    indices = np.arange(len(labels))
    for _ in range(bootstraps):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if np.unique(labels[sample]).size < 2:
            continue
        estimates.append(roc_auc_score(labels[sample], probabilities[sample]))
    return np.asarray(estimates)


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def prepare_dataset(predictions, clinical, acquisition, cohort):
    table = predictions.merge(
        clinical[["ID", "sex", "age"]], on="ID", how="left", validate="one_to_one"
    )
    metadata = acquisition.loc[acquisition["cohort"] == cohort].copy()
    table = table.merge(
        metadata[["ID", "field_strength_t", "sax_slices"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )
    table["sex_group"] = table["sex"].map({0: "Female", 1: "Male"})
    table["age_group"] = np.where(table["age"] < 55, "Age <55 years", "Age ≥55 years")
    table["field_strength_group"] = table["field_strength_t"].map(
        {1.5: "1.5 T", 3.0: "3.0 T"}
    )
    table["sax_slice_group"] = np.where(
        table["sax_slices"] < 10, "<10 SAX slices", "≥10 SAX slices"
    )
    return table


def main():
    args = parse_args()
    development = pd.read_csv(args.results_dir / "development_predictions.csv")
    external = pd.read_csv(args.results_dir / "external_predictions.csv")
    train_clinical = pd.read_excel(PROJECT_ROOT / "CMR parameter/train.xlsx")
    external_clinical = pd.read_excel(PROJECT_ROOT / "CMR parameter/external.xlsx")
    acquisition = pd.read_csv(args.acquisition_metadata)
    run_config = json.loads((args.results_dir / "run_config.json").read_text())
    thresholds = run_config["thresholds"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "Development OOF": prepare_dataset(
            development, train_clinical, acquisition, "development"
        ),
        "External test": prepare_dataset(
            external, external_clinical, acquisition, "external_test"
        ),
    }
    subgroup_columns = {
        "Sex": "sex_group",
        "Age": "age_group",
        "Field strength": "field_strength_group",
        "SAX slice coverage": "sax_slice_group",
    }
    metric_rows = []
    difference_rows = []
    rng = np.random.default_rng(args.seed)
    for dataset_name, table in datasets.items():
        for model in args.models:
            probability_column = f"{model}_probability"
            if probability_column not in table or model not in thresholds:
                continue
            threshold = float(thresholds[model])
            for subgroup_name, subgroup_column in subgroup_columns.items():
                valid = table.dropna(subset=[subgroup_column, "event", probability_column])
                groups = []
                for level, subset in valid.groupby(subgroup_column, sort=True):
                    labels = subset["event"].to_numpy(int)
                    probabilities = subset[probability_column].to_numpy(float)
                    if (
                        len(subset) < args.minimum_group_size
                        or np.unique(labels).size < 2
                        or labels.sum() < 5
                        or (len(labels) - labels.sum()) < 5
                    ):
                        continue
                    point = operating_metrics(labels, probabilities, threshold)
                    auc_samples = bootstrap_auc(labels, probabilities, args.bootstraps, rng)
                    metric_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "subgroup": subgroup_name,
                            "level": level,
                            "n": len(subset),
                            "event_n": int(labels.sum()),
                            "event_prevalence": float(labels.mean()),
                            "threshold": threshold,
                            **point,
                            "auc_ci_lower": float(np.percentile(auc_samples, 2.5)),
                            "auc_ci_upper": float(np.percentile(auc_samples, 97.5)),
                        }
                    )
                    groups.append((level, labels, probabilities, auc_samples))
                if len(groups) == 2:
                    first, second = groups
                    count = min(len(first[3]), len(second[3]))
                    differences = first[3][:count] - second[3][:count]
                    point_difference = roc_auc_score(first[1], first[2]) - roc_auc_score(
                        second[1], second[2]
                    )
                    p_value = min(
                        1.0,
                        2
                        * min(
                            float(np.mean(differences <= 0)),
                            float(np.mean(differences >= 0)),
                        ),
                    )
                    difference_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "subgroup": subgroup_name,
                            "level_1": first[0],
                            "level_2": second[0],
                            "auc_difference_level_1_minus_level_2": point_difference,
                            "ci_lower": float(np.percentile(differences, 2.5)),
                            "ci_upper": float(np.percentile(differences, 97.5)),
                            "p_value": p_value,
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    differences = pd.DataFrame(difference_rows)
    if not differences.empty:
        differences["holm_adjusted_p"] = holm_adjust(differences["p_value"])
    metrics.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    differences.to_csv(args.output_dir / "subgroup_auc_differences.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "results_dir": str(args.results_dir),
                "acquisition_metadata": str(args.acquisition_metadata),
                "models": args.models,
                "age_cutoff_years": 55,
                "sax_slice_cutoff": 10,
                "bootstraps": args.bootstraps,
                "minimum_group_size": args.minimum_group_size,
                "seed": args.seed,
                "threshold_policy": "development-locked model-specific thresholds",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    significant = (
        differences.loc[differences["holm_adjusted_p"] < 0.05]
        if not differences.empty
        else differences.copy()
    )
    lines = [
        "# Subgroup Performance Analysis",
        "",
        "Available demographic and acquisition subgroups were evaluated using model-specific development-locked thresholds.",
        f"A total of {len(metrics)} subgroup-specific metric rows and {len(differences)} two-group AUC comparisons were generated.",
        f"Holm-significant AUC differences: {len(significant)}.",
        "",
        "Subgroup estimates are exploratory and should not be interpreted as proof of equivalence.",
    ]
    (args.output_dir / "SUBGROUP_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
