#!/usr/bin/env python3
"""Generate paired discrimination, calibration and decision-curve analyses."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta, norm
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


DEFAULT_RESULTS = (
    Path(__file__).resolve().parent / "results/foundation_comparison_nifti_v2"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS / "additional_statistics"
DEFAULT_INCREMENTAL_PAIRS = [
    ("CMR", "Rscore+CMR"),
    ("CMR", "Foundation+CMR"),
    ("Foundation", "Foundation+CMR"),
    ("Rscore+CMR", "Foundation+Rscore+CMR"),
    ("Foundation+CMR", "Foundation+Rscore+CMR"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-predictions",
        type=Path,
        default=DEFAULT_RESULTS / "development_oof_predictions.csv",
    )
    parser.add_argument(
        "--external-predictions",
        type=Path,
        default=DEFAULT_RESULTS / "external_test_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--dca-min", type=float, default=0.01)
    parser.add_argument("--dca-max", type=float, default=0.80)
    parser.add_argument("--dca-step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def model_name(probability_column):
    return probability_column.removesuffix("_probability")


def load_predictions(path):
    table = pd.read_csv(path)
    if {"Model", "Row index", "True label", "OOF probability"}.issubset(table.columns):
        model_order = table["Model"].drop_duplicates().tolist()
        averaged = (
            table.groupby(["Model", "Row index"], sort=False)
            .agg(event=("True label", "first"), probability=("OOF probability", "mean"))
            .reset_index()
        )
        labels = averaged.groupby("Row index")["event"].nunique()
        if (labels != 1).any():
            raise ValueError(f"{path} 的重复OOF标签不一致")
        wide = averaged.pivot(index="Row index", columns="Model", values="probability")
        events = averaged.groupby("Row index")["event"].first()
        table = pd.DataFrame({"ID": events.index, "event": events.values})
        for name in model_order:
            table[f"{name}_probability"] = wide.loc[events.index, name].to_numpy()

    probability_columns = [
        column for column in table.columns if column.endswith("_probability")
    ]
    if not probability_columns:
        raise ValueError(f"{path} 中没有 *_probability 列")
    required = ["event", *probability_columns]
    if table[required].isna().any().any():
        raise ValueError(f"{path} 中存在缺失标签或概率")
    labels = table["event"].astype(int).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{path} 必须同时包含0和1标签")
    probabilities = {
        model_name(column): table[column].astype(float).to_numpy()
        for column in probability_columns
    }
    for name, values in probabilities.items():
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{path} 的 {name} 概率无效")
    return labels, probabilities


def compute_midrank(values):
    order = np.argsort(values)
    sorted_values = values[order]
    count = len(values)
    midranks = np.empty(count, dtype=float)
    start = 0
    while start < count:
        end = start
        while end < count and sorted_values[end] == sorted_values[start]:
            end += 1
        midranks[start:end] = 0.5 * (start + end - 1)
        start = end
    result = np.empty(count, dtype=float)
    result[order] = midranks + 1.0
    return result


def fast_delong(sorted_predictions, positive_count):
    """
    高效的 DeLong 检验实现，用于计算多个分类器的 AUC 及它们的协方差矩阵。
    """
    model_count, sample_count = sorted_predictions.shape
    negative_count = sample_count - positive_count
    positive = sorted_predictions[:, :positive_count]
    negative = sorted_predictions[:, positive_count:]
    positive_ranks = np.empty_like(positive)
    negative_ranks = np.empty_like(negative)
    pooled_ranks = np.empty_like(sorted_predictions)
    for index in range(model_count):
        positive_ranks[index] = compute_midrank(positive[index])
        negative_ranks[index] = compute_midrank(negative[index])
        pooled_ranks[index] = compute_midrank(sorted_predictions[index])
    aucs = (
        pooled_ranks[:, :positive_count].sum(axis=1)
        / positive_count
        / negative_count
        - (positive_count + 1.0) / 2.0 / negative_count
    )
    positive_components = (
        pooled_ranks[:, :positive_count] - positive_ranks
    ) / negative_count
    negative_components = 1.0 - (
        pooled_ranks[:, positive_count:] - negative_ranks
    ) / positive_count
    covariance = np.atleast_2d(np.cov(positive_components)) / positive_count
    covariance += np.atleast_2d(np.cov(negative_components)) / negative_count
    return aucs, covariance


def delong_estimates(labels, probabilities):
    names = list(probabilities)
    order = np.argsort(-labels)
    matrix = np.vstack([probabilities[name] for name in names])[:, order]
    aucs, covariance = fast_delong(matrix, int(labels.sum()))
    return names, aucs, covariance


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def delong_tables(dataset, labels, probabilities):
    names, aucs, covariance = delong_estimates(labels, probabilities)
    auc_rows = []
    for index, name in enumerate(names):
        standard_error = float(np.sqrt(max(covariance[index, index], 0.0)))
        auc_rows.append(
            {
                "Dataset": dataset,
                "Model": name,
                "AUC": aucs[index],
                "SE": standard_error,
                "CI lower": max(0.0, aucs[index] - 1.959964 * standard_error),
                "CI upper": min(1.0, aucs[index] + 1.959964 * standard_error),
            }
        )

    pair_rows = []
    for first, second in combinations(range(len(names)), 2):
        difference = float(aucs[second] - aucs[first])
        variance = float(
            covariance[first, first]
            + covariance[second, second]
            - 2.0 * covariance[first, second]
        )
        standard_error = np.sqrt(max(variance, 0.0))
        if standard_error == 0:
            z_value = 0.0 if difference == 0 else np.sign(difference) * np.inf
            p_value = 1.0 if difference == 0 else 0.0
        else:
            z_value = difference / standard_error
            p_value = 2.0 * norm.sf(abs(z_value))
        pair_rows.append(
            {
                "Dataset": dataset,
                "Reference model": names[first],
                "Comparison model": names[second],
                "Reference AUC": aucs[first],
                "Comparison AUC": aucs[second],
                "Delta AUC": difference,
                "Delta CI lower": difference - 1.959964 * standard_error,
                "Delta CI upper": difference + 1.959964 * standard_error,
                "Z": z_value,
                "P value": p_value,
            }
        )
    adjusted = holm_adjust([row["P value"] for row in pair_rows])
    for row, adjusted_value in zip(pair_rows, adjusted):
        row["Holm-adjusted P value"] = adjusted_value
        row["Significant after Holm"] = bool(adjusted_value < 0.05)
    return auc_rows, pair_rows


def calibration_fit(labels, probabilities):
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    design = np.column_stack([np.ones(len(clipped)), logit(clipped)])
    coefficients = np.array([0.0, 1.0])
    for _ in range(100):
        fitted = expit(design @ coefficients)
        weights = np.clip(fitted * (1.0 - fitted), 1e-9, None)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (labels - fitted)
        try:
            step = np.linalg.solve(information + np.eye(2) * 1e-10, score)
        except np.linalg.LinAlgError:
            return np.array([np.nan, np.nan])
        coefficients += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return coefficients


def stratified_bootstrap_indices(labels, iterations, seed):
    random = np.random.default_rng(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    for _ in range(iterations):
        yield np.concatenate(
            [
                random.choice(negative, len(negative), replace=True),
                random.choice(positive, len(positive), replace=True),
            ]
        )


def calibration_points(labels, probabilities, bins):
    quantiles = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 3:
        quantiles = np.linspace(0.0, 1.0, bins + 1)
    groups = np.digitize(probabilities, quantiles[1:-1], right=True)
    rows = []
    for group in np.unique(groups):
        selected = groups == group
        count = int(selected.sum())
        events = int(labels[selected].sum())
        lower = 0.0 if events == 0 else beta.ppf(0.025, events, count - events + 1)
        upper = 1.0 if events == count else beta.ppf(0.975, events + 1, count - events)
        rows.append(
            {
                "Bin": int(group + 1),
                "N": count,
                "Mean predicted": float(probabilities[selected].mean()),
                "Observed": float(labels[selected].mean()),
                "Observed CI lower": float(lower),
                "Observed CI upper": float(upper),
            }
        )
    return rows


def expected_calibration_error(labels, probabilities, bins):
    rows = calibration_points(labels, probabilities, bins)
    return sum(
        row["N"] / len(labels) * abs(row["Observed"] - row["Mean predicted"])
        for row in rows
    )


def calibration_tables(dataset, labels, probabilities, bins, bootstraps, seed):
    metric_rows = []
    point_rows = []
    for model_index, (name, values) in enumerate(probabilities.items()):
        coefficients = calibration_fit(labels, values)
        bootstrap_values = []
        for indices in stratified_bootstrap_indices(
            labels, bootstraps, seed + model_index * 1000
        ):
            estimate = calibration_fit(labels[indices], values[indices])
            if np.isfinite(estimate).all():
                bootstrap_values.append(estimate)
        bootstrap_values = np.asarray(bootstrap_values)
        interval = np.percentile(bootstrap_values, [2.5, 97.5], axis=0)
        metric_rows.append(
            {
                "Dataset": dataset,
                "Model": name,
                "N": len(labels),
                "Events": int(labels.sum()),
                "AUC": roc_auc_score(labels, values),
                "Brier": brier_score_loss(labels, values),
                "Calibration intercept": coefficients[0],
                "Intercept CI lower": interval[0, 0],
                "Intercept CI upper": interval[1, 0],
                "Calibration slope": coefficients[1],
                "Slope CI lower": interval[0, 1],
                "Slope CI upper": interval[1, 1],
                "ECE": expected_calibration_error(labels, values, bins),
            }
        )
        for row in calibration_points(labels, values, bins):
            point_rows.append({"Dataset": dataset, "Model": name, **row})
    return metric_rows, point_rows


def continuous_nri_idi(labels, old_probability, new_probability):
    event = labels == 1
    nonevent = labels == 0
    event_nri = np.mean(new_probability[event] > old_probability[event]) - np.mean(
        new_probability[event] < old_probability[event]
    )
    nonevent_nri = np.mean(
        new_probability[nonevent] < old_probability[nonevent]
    ) - np.mean(new_probability[nonevent] > old_probability[nonevent])
    idi = (
        np.mean(new_probability[event])
        - np.mean(new_probability[nonevent])
        - np.mean(old_probability[event])
        + np.mean(old_probability[nonevent])
    )
    return np.array([event_nri, nonevent_nri, event_nri + nonevent_nri, idi])


def two_sided_bootstrap_p(values):
    below = (np.sum(values <= 0) + 1) / (len(values) + 1)
    above = (np.sum(values >= 0) + 1) / (len(values) + 1)
    return min(1.0, 2.0 * min(below, above))


def incremental_table(dataset, labels, probabilities, pairs, bootstraps, seed):
    names, aucs, covariance = delong_estimates(labels, probabilities)
    name_to_index = {name: index for index, name in enumerate(names)}
    rows = []
    for pair_index, (reference, comparison) in enumerate(pairs):
        if reference not in probabilities or comparison not in probabilities:
            continue
        first = name_to_index[reference]
        second = name_to_index[comparison]
        variance = (
            covariance[first, first]
            + covariance[second, second]
            - 2.0 * covariance[first, second]
        )
        auc_difference = float(aucs[second] - aucs[first])
        auc_se = np.sqrt(max(float(variance), 0.0))
        delong_p = 1.0 if auc_se == 0 and auc_difference == 0 else (
            0.0 if auc_se == 0 else 2.0 * norm.sf(abs(auc_difference / auc_se))
        )
        estimates = continuous_nri_idi(
            labels, probabilities[reference], probabilities[comparison]
        )
        bootstrap_values = []
        for indices in stratified_bootstrap_indices(
            labels, bootstraps, seed + pair_index * 1000
        ):
            bootstrap_values.append(
                continuous_nri_idi(
                    labels[indices],
                    probabilities[reference][indices],
                    probabilities[comparison][indices],
                )
            )
        bootstrap_values = np.asarray(bootstrap_values)
        intervals = np.percentile(bootstrap_values, [2.5, 97.5], axis=0)
        rows.append(
            {
                "Dataset": dataset,
                "Reference model": reference,
                "Comparison model": comparison,
                "Delta AUC": auc_difference,
                "Delta AUC CI lower": auc_difference - 1.959964 * auc_se,
                "Delta AUC CI upper": auc_difference + 1.959964 * auc_se,
                "DeLong P value": delong_p,
                "Event NRI": estimates[0],
                "Event NRI CI lower": intervals[0, 0],
                "Event NRI CI upper": intervals[1, 0],
                "Nonevent NRI": estimates[1],
                "Nonevent NRI CI lower": intervals[0, 1],
                "Nonevent NRI CI upper": intervals[1, 1],
                "Continuous NRI": estimates[2],
                "NRI CI lower": intervals[0, 2],
                "NRI CI upper": intervals[1, 2],
                "NRI bootstrap P value": two_sided_bootstrap_p(bootstrap_values[:, 2]),
                "IDI": estimates[3],
                "IDI CI lower": intervals[0, 3],
                "IDI CI upper": intervals[1, 3],
                "IDI bootstrap P value": two_sided_bootstrap_p(bootstrap_values[:, 3]),
            }
        )
    return rows


def decision_curve_table(dataset, labels, probabilities, thresholds):
    rows = []
    prevalence = labels.mean()
    sample_count = len(labels)
    for threshold in thresholds:
        odds = threshold / (1.0 - threshold)
        rows.extend(
            [
                {
                    "Dataset": dataset,
                    "Model": "Treat none",
                    "Threshold": threshold,
                    "Net benefit": 0.0,
                },
                {
                    "Dataset": dataset,
                    "Model": "Treat all",
                    "Threshold": threshold,
                    "Net benefit": prevalence - (1.0 - prevalence) * odds,
                },
            ]
        )
        for name, values in probabilities.items():
            predicted = values >= threshold
            true_positive = np.sum(predicted & (labels == 1))
            false_positive = np.sum(predicted & (labels == 0))
            net_benefit = true_positive / sample_count - false_positive / sample_count * odds
            rows.append(
                {
                    "Dataset": dataset,
                    "Model": name,
                    "Threshold": threshold,
                    "Net benefit": net_benefit,
                }
            )
    return rows


def roc_points(dataset, labels, probabilities):
    rows = []
    for name, values in probabilities.items():
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, values)
        for fpr, tpr, threshold in zip(false_positive_rate, true_positive_rate, thresholds):
            rows.append(
                {
                    "Dataset": dataset,
                    "Model": name,
                    "FPR": fpr,
                    "TPR": tpr,
                    "Threshold": threshold,
                }
            )
    return rows


def plot_calibration(dataset, point_table, model_order, output_path):
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for axis, name in zip(axes.flat, model_order):
        selected = point_table.loc[point_table["Model"] == name].sort_values("Bin")
        lower = selected["Observed"] - selected["Observed CI lower"]
        upper = selected["Observed CI upper"] - selected["Observed"]
        axis.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1)
        axis.errorbar(
            selected["Mean predicted"],
            selected["Observed"],
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=1.5,
            capsize=2,
        )
        axis.set_title(name)
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(model_order):]:
        axis.axis("off")
    figure.supxlabel("Mean predicted probability")
    figure.supylabel("Observed event rate")
    figure.suptitle(f"Calibration — {dataset}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc(dataset, labels, probabilities, output_path):
    figure, axis = plt.subplots(figsize=(7, 6))
    for name, values in probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, values)
        auc = roc_auc_score(labels, values)
        axis.plot(false_positive_rate, true_positive_rate, label=f"{name} ({auc:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="0.55", label="Chance")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC — {dataset}")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_decision_curve(dataset, table, model_order, output_path):
    figure, axis = plt.subplots(figsize=(8, 6))
    for name in [*model_order, "Treat all", "Treat none"]:
        selected = table.loc[table["Model"] == name].sort_values("Threshold")
        style = "--" if name in {"Treat all", "Treat none"} else "-"
        color = "black" if name == "Treat none" else ("0.5" if name == "Treat all" else None)
        axis.plot(
            selected["Threshold"],
            selected["Net benefit"],
            linestyle=style,
            color=color,
            label=name,
        )
    axis.set(
        xlabel="Threshold probability",
        ylabel="Net benefit",
        title=f"Decision curve — {dataset}",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_summary(output_dir, calibration, incremental):
    external_calibration = calibration.loc[calibration["Dataset"] == "External test"]
    external_incremental = (
        incremental.loc[incremental["Dataset"] == "External test"]
        if "Dataset" in incremental.columns
        else incremental
    )
    lines = [
        "# Additional Statistical Analyses",
        "",
        "The tables use paired patient-level predictions. DeLong P values are two-sided;",
        "the all-pairs table also reports Holm multiplicity correction. Calibration",
        "intercept/slope, continuous NRI and IDI confidence intervals use stratified",
        "bootstrap resampling. No external-test recalibration was performed.",
        "",
        "## External calibration",
        "",
        "| Model | Brier | Intercept | Slope | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in external_calibration.to_dict("records"):
        lines.append(
            f"| {row['Model']} | {row['Brier']:.4f} | "
            f"{row['Calibration intercept']:.3f} | "
            f"{row['Calibration slope']:.3f} | {row['ECE']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## External incremental comparisons",
            "",
            "| Reference | Comparison | Delta AUC | DeLong P | Continuous NRI | IDI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in external_incremental.to_dict("records"):
        lines.append(
            f"| {row['Reference model']} | {row['Comparison model']} | "
            f"{row['Delta AUC']:.4f} | {row['DeLong P value']:.4g} | "
            f"{row['Continuous NRI']:.4f} | {row['IDI']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `auc_estimates.csv` and `delong_pairwise.csv`: discrimination and paired tests.",
            "- `calibration_metrics.csv` and `calibration_curve_points.csv`: calibration results.",
            "- `incremental_value.csv`: Delta AUC, continuous NRI and IDI.",
            "- `decision_curve_points.csv`: model, treat-all and treat-none net benefit.",
            "- `*.png`: publication-resolution ROC, calibration and decision-curve figures.",
        ]
    )
    (output_dir / "STATISTICAL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """
    主程序：生成配对区分度 (DeLong AUC比较)、校准度 (Calibration Curve)
    以及临床决策曲线 (Decision Curve Analysis, DCA) 等附加统计量。
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "Development OOF": load_predictions(args.development_predictions),
        "External test": load_predictions(args.external_predictions),
    }
    development_models = list(datasets["Development OOF"][1])
    external_models = list(datasets["External test"][1])
    if development_models != external_models:
        raise ValueError("开发集和外部测试集的模型列不一致")

    auc_rows = []
    delong_rows = []
    calibration_rows = []
    calibration_point_rows = []
    incremental_rows = []
    decision_rows = []
    roc_rows = []
    thresholds = np.arange(args.dca_min, args.dca_max + args.dca_step / 2, args.dca_step)

    for dataset_index, (dataset, (labels, probabilities)) in enumerate(datasets.items()):
        dataset_auc, dataset_delong = delong_tables(dataset, labels, probabilities)
        auc_rows.extend(dataset_auc)
        delong_rows.extend(dataset_delong)
        metrics, points = calibration_tables(
            dataset,
            labels,
            probabilities,
            args.calibration_bins,
            args.bootstraps,
            args.seed + dataset_index * 10000,
        )
        calibration_rows.extend(metrics)
        calibration_point_rows.extend(points)
        incremental_rows.extend(
            incremental_table(
                dataset,
                labels,
                probabilities,
                DEFAULT_INCREMENTAL_PAIRS,
                args.bootstraps,
                args.seed + dataset_index * 20000,
            )
        )
        decision_rows.extend(decision_curve_table(dataset, labels, probabilities, thresholds))
        roc_rows.extend(roc_points(dataset, labels, probabilities))

    auc_table = pd.DataFrame(auc_rows)
    delong_table = pd.DataFrame(delong_rows)
    calibration_table = pd.DataFrame(calibration_rows)
    calibration_points_table = pd.DataFrame(calibration_point_rows)
    incremental_table_output = pd.DataFrame(incremental_rows)
    if incremental_table_output.empty:
        incremental_table_output = pd.DataFrame(
            columns=[
                "Dataset",
                "Reference model",
                "Comparison model",
                "Delta AUC",
                "DeLong P value",
                "Continuous NRI",
                "IDI",
            ]
        )
    decision_table = pd.DataFrame(decision_rows)
    roc_table = pd.DataFrame(roc_rows)
    auc_table.to_csv(args.output_dir / "auc_estimates.csv", index=False)
    delong_table.to_csv(args.output_dir / "delong_pairwise.csv", index=False)
    calibration_table.to_csv(args.output_dir / "calibration_metrics.csv", index=False)
    calibration_points_table.to_csv(
        args.output_dir / "calibration_curve_points.csv", index=False
    )
    incremental_table_output.to_csv(args.output_dir / "incremental_value.csv", index=False)
    decision_table.to_csv(args.output_dir / "decision_curve_points.csv", index=False)
    roc_table.to_csv(args.output_dir / "roc_curve_points.csv", index=False)

    for dataset, (labels, probabilities) in datasets.items():
        slug = "development" if dataset == "Development OOF" else "external"
        selected_calibration = calibration_points_table.loc[
            calibration_points_table["Dataset"] == dataset
        ]
        selected_decision = decision_table.loc[decision_table["Dataset"] == dataset]
        plot_calibration(
            dataset,
            selected_calibration,
            development_models,
            args.output_dir / f"calibration_{slug}.png",
        )
        plot_roc(dataset, labels, probabilities, args.output_dir / f"roc_{slug}.png")
        plot_decision_curve(
            dataset,
            selected_decision,
            development_models,
            args.output_dir / f"decision_curve_{slug}.png",
        )

    config = {
        "development_predictions": str(args.development_predictions.resolve()),
        "external_predictions": str(args.external_predictions.resolve()),
        "models": development_models,
        "bootstraps": args.bootstraps,
        "calibration_bins": args.calibration_bins,
        "decision_curve_thresholds": [args.dca_min, args.dca_max, args.dca_step],
        "seed": args.seed,
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(args.output_dir, calibration_table, incremental_table_output)
    print(f"Additional statistics saved to {args.output_dir}")


if __name__ == "__main__":
    main()
