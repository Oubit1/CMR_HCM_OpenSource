#!/usr/bin/env python3
"""Combine foundation-representation ablations and five-classifier controls."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_additional_statistics import delong_tables


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_FOUNDATION = EXPERIMENT_ROOT / "results/foundation_comparison_nifti_v2"
DEFAULT_CLASSIFIERS = EXPERIMENT_ROOT / "results/tabular_five_models_rebuilt_v1"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/complete_comparison_v1"

FOUNDATION_LABELS = {
    "CMR": "CMR (Elastic-net)",
    "Rscore+CMR": "Rscore+CMR (Elastic-net)",
    "Foundation": "Foundation (Elastic-net)",
    "Foundation+CMR": "Foundation+CMR (Elastic-net)",
    "Foundation+Rscore+CMR": "Foundation+Rscore+CMR (Elastic-net)",
}
CLASSIFIER_ORDER = ["LightGBM", "SVM", "XGBoost", "CatBoost", "Bagging"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation-dir", type=Path, default=DEFAULT_FOUNDATION)
    parser.add_argument("--classifier-dir", type=Path, default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_auc_intervals(statistics_dir, dataset):
    table = pd.read_csv(statistics_dir / "auc_estimates.csv")
    return table.loc[table["Dataset"] == dataset].set_index("Model")


def foundation_external_table(directory):
    metrics = pd.read_csv(directory / "model_comparison.csv")
    metrics = metrics.loc[metrics["Dataset"] == "External test"].copy()
    intervals = load_auc_intervals(directory / "additional_statistics", "External test")
    rows = []
    for row in metrics.to_dict("records"):
        interval = intervals.loc[row["Model"]]
        rows.append(
            {
                "Experiment family": "Representation ablation",
                "Model": FOUNDATION_LABELS[row["Model"]],
                "Input": row["Model"],
                "Classifier": "Elastic-net logistic regression",
                "N": 309,
                "AUC": row["AUC"],
                "AUC CI lower": interval["CI lower"],
                "AUC CI upper": interval["CI upper"],
                "Sensitivity": row["Sensitivity"],
                "Specificity": row["Specificity"],
                "PPV": row["PPV"],
                "NPV": row["NPV"],
                "Accuracy": row["Accuracy"],
                "F1": row["F1"],
                "Brier": row["Brier"],
                "Locked threshold": row["Locked threshold"],
            }
        )
    return pd.DataFrame(rows)


def classifier_external_table(directory):
    metrics = pd.read_csv(directory / "external_test_metrics.csv")
    intervals = load_auc_intervals(directory / "additional_statistics", "External test")
    rows = []
    for classifier in CLASSIFIER_ORDER:
        row = metrics.loc[metrics["Model"] == classifier].iloc[0]
        interval = intervals.loc[classifier]
        rows.append(
            {
                "Experiment family": "Rscore+CMR classifier benchmark",
                "Model": f"Rscore+CMR ({classifier})",
                "Input": "Rscore+CMR",
                "Classifier": classifier,
                "N": 309,
                "AUC": row["AUC"],
                "AUC CI lower": interval["CI lower"],
                "AUC CI upper": interval["CI upper"],
                "Sensitivity": row["Sensitivity"],
                "Specificity": row["Specificity"],
                "PPV": row["PPV"],
                "NPV": row["NPV"],
                "Accuracy": row["Accuracy"],
                "F1": row["F1 Score"],
                "Brier": row["Brier Score"],
                "Locked threshold": row["Locked threshold"],
            }
        )
    return pd.DataFrame(rows)


def development_table(foundation_dir, classifier_dir):
    foundation = pd.read_csv(foundation_dir / "model_comparison.csv")
    foundation = foundation.loc[foundation["Dataset"] == "Development OOF"].copy()
    foundation_rows = []
    for row in foundation.to_dict("records"):
        foundation_rows.append(
            {
                "Experiment family": "Representation ablation",
                "Model": FOUNDATION_LABELS[row["Model"]],
                "N": 1050,
                "AUC": row["AUC"],
                "Sensitivity": row["Sensitivity"],
                "Specificity": row["Specificity"],
                "NPV": row["NPV"],
                "Brier": row["Brier"],
            }
        )

    summary = pd.read_csv(classifier_dir / "Repeated_nested_CV_summary.csv")
    estimates = summary.pivot(index="Model", columns="Metric", values="Estimate")
    classifier_rows = []
    for classifier in CLASSIFIER_ORDER:
        row = estimates.loc[classifier]
        classifier_rows.append(
            {
                "Experiment family": "Rscore+CMR classifier benchmark",
                "Model": f"Rscore+CMR ({classifier})",
                "N": 1059,
                "AUC": row["AUC"],
                "Sensitivity": row["Sensitivity"],
                "Specificity": row["Specificity"],
                "NPV": row["NPV"],
                "Brier": row["Brier Score"],
            }
        )
    return pd.DataFrame([*foundation_rows, *classifier_rows])


def combined_external_predictions(foundation_dir, classifier_dir):
    foundation = pd.read_csv(foundation_dir / "external_test_predictions.csv")
    classifiers = pd.read_csv(classifier_dir / "external_test_predictions.csv")
    foundation_columns = {"ID": "ID", "event": "event_foundation"}
    for source, label in FOUNDATION_LABELS.items():
        foundation_columns[f"{source}_probability"] = f"{label}_probability"
    foundation = foundation[list(foundation_columns)].rename(columns=foundation_columns)

    classifier_columns = {"ID": "ID", "event": "event_classifiers"}
    for classifier in CLASSIFIER_ORDER:
        classifier_columns[f"{classifier}_probability"] = (
            f"Rscore+CMR ({classifier})_probability"
        )
    classifiers = classifiers[list(classifier_columns)].rename(columns=classifier_columns)
    combined = foundation.merge(classifiers, on="ID", how="inner", validate="one_to_one")
    if len(combined) != 309:
        raise ValueError(f"外部测试集匹配后应为309例，实际为{len(combined)}例")
    if not np.array_equal(combined["event_foundation"], combined["event_classifiers"]):
        raise ValueError("两组实验的外部测试标签不一致")
    combined = combined.rename(columns={"event_foundation": "event"}).drop(
        columns="event_classifiers"
    )
    return combined


def complete_delong_table(predictions):
    probabilities = {
        column.removesuffix("_probability"): predictions[column].to_numpy(float)
        for column in predictions.columns
        if column.endswith("_probability")
    }
    _, rows = delong_tables("External test", predictions["event"].to_numpy(int), probabilities)
    return pd.DataFrame(rows)


def comparison_lookup(table, reference, comparison):
    direct = table.loc[
        (table["Reference model"] == reference)
        & (table["Comparison model"] == comparison)
    ]
    if len(direct):
        return direct.iloc[0].to_dict()
    reverse = table.loc[
        (table["Reference model"] == comparison)
        & (table["Comparison model"] == reference)
    ]
    if not len(reverse):
        raise KeyError(f"未找到 {reference} 与 {comparison} 的DeLong结果")
    row = reverse.iloc[0].to_dict()
    row["Reference model"], row["Comparison model"] = reference, comparison
    row["Reference AUC"], row["Comparison AUC"] = row["Comparison AUC"], row["Reference AUC"]
    row["Delta AUC"] *= -1
    old_lower = row["Delta CI lower"]
    row["Delta CI lower"] = -row["Delta CI upper"]
    row["Delta CI upper"] = -old_lower
    row["Z"] *= -1
    return row


def key_delong_table(all_pairs):
    comparisons = []
    neural_models = [
        "Foundation (Elastic-net)",
        "Foundation+CMR (Elastic-net)",
        "Foundation+Rscore+CMR (Elastic-net)",
    ]
    traditional_models = [f"Rscore+CMR ({name})" for name in CLASSIFIER_ORDER]
    for neural_model in neural_models:
        for traditional_model in traditional_models:
            comparisons.append(comparison_lookup(all_pairs, neural_model, traditional_model))
    comparisons.append(
        comparison_lookup(
            all_pairs,
            "Rscore+CMR (Elastic-net)",
            "Foundation+Rscore+CMR (Elastic-net)",
        )
    )
    return pd.DataFrame(comparisons)


def plot_auc_forest(table, output_path):
    ordered = table.iloc[::-1].reset_index(drop=True)
    family_colors = {
        "Representation ablation": "#2c7fb8",
        "Rscore+CMR classifier benchmark": "#d95f0e",
    }
    positions = np.arange(len(ordered))
    lower = ordered["AUC"] - ordered["AUC CI lower"]
    upper = ordered["AUC CI upper"] - ordered["AUC"]
    figure, axis = plt.subplots(figsize=(10, 7))
    for family, color in family_colors.items():
        selected = ordered["Experiment family"] == family
        axis.errorbar(
            ordered.loc[selected, "AUC"],
            positions[selected],
            xerr=np.vstack([lower[selected], upper[selected]]),
            fmt="none",
            ecolor=color,
            capsize=3,
            linewidth=1.5,
        )
        axis.scatter(
            ordered.loc[selected, "AUC"],
            positions[selected],
            color=color,
            s=55,
            zorder=3,
            label=family,
        )
    axis.set_yticks(positions, ordered["Model"])
    axis.set_xlim(0.60, 0.96)
    axis.set_xlabel("External-test AUC (95% DeLong CI)")
    axis.set_title("Frozen foundation representation versus radiomics-based controls")
    axis.grid(axis="x", alpha=0.25)
    axis.axvline(0.5, color="0.6", linestyle="--", linewidth=1)
    axis.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_operating_points(table, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    family_styles = {
        "Representation ablation": ("#2c7fb8", "o"),
        "Rscore+CMR classifier benchmark": ("#d95f0e", "s"),
    }
    short_names = {
        "CMR (Elastic-net)": "CMR EN",
        "Rscore+CMR (Elastic-net)": "Rscore+CMR EN",
        "Foundation (Elastic-net)": "Foundation EN",
        "Foundation+CMR (Elastic-net)": "Foundation+CMR EN",
        "Foundation+Rscore+CMR (Elastic-net)": "Full EN",
        "Rscore+CMR (LightGBM)": "LightGBM",
        "Rscore+CMR (SVM)": "SVM",
        "Rscore+CMR (XGBoost)": "XGBoost",
        "Rscore+CMR (CatBoost)": "CatBoost",
        "Rscore+CMR (Bagging)": "Bagging",
    }
    offsets = {
        "CMR (Elastic-net)": (4, -12),
        "Rscore+CMR (Elastic-net)": (4, -12),
        "Foundation (Elastic-net)": (4, -12),
        "Foundation+CMR (Elastic-net)": (4, 6),
        "Foundation+Rscore+CMR (Elastic-net)": (4, 7),
        "Rscore+CMR (LightGBM)": (-42, 7),
        "Rscore+CMR (SVM)": (4, 4),
        "Rscore+CMR (XGBoost)": (-25, 8),
        "Rscore+CMR (CatBoost)": (4, -13),
        "Rscore+CMR (Bagging)": (-35, -13),
    }
    for axis_index, axis in enumerate(axes):
        for family, selected in table.groupby("Experiment family", sort=False):
            color, marker = family_styles[family]
            axis.scatter(
                selected["Specificity"],
                selected["Sensitivity"],
                s=70,
                color=color,
                marker=marker,
                label=family,
            )
            for row in selected.to_dict("records"):
                if axis_index == 1 and row["Model"] == "Rscore+CMR (SVM)":
                    continue
                axis.annotate(
                    short_names[row["Model"]],
                    (row["Specificity"], row["Sensitivity"]),
                    xytext=offsets[row["Model"]],
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.axhline(
            0.95,
            color="0.45",
            linestyle="--",
            linewidth=1,
            label="Sensitivity target" if axis_index == 0 else None,
        )
        axis.grid(alpha=0.2)
        axis.set_xlabel("External-test specificity at locked threshold")
    axes[0].set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), title="All models")
    axes[0].set_ylabel("External-test sensitivity at locked threshold")
    axes[1].set(xlim=(0.25, 0.80), ylim=(0.88, 0.98), title="High-sensitivity region")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=8, loc="lower center", ncol=3)
    figure.suptitle("Locked operating points")
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.16)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_report(output_dir, external, key_delong):
    """
    撰写包含核心指标和显著性发现的总结报告 (COMPLETE_COMPARISON_CN.md)。
    用于一览所有模型的性能差异。
    """
    indexed = external.set_index("Model")
    selected = indexed.loc["Rscore+CMR (XGBoost)"]
    catboost = indexed.loc["Rscore+CMR (CatBoost)"]
    foundation = indexed.loc["Foundation (Elastic-net)"]
    foundation_cmr = indexed.loc["Foundation+CMR (Elastic-net)"]
    full = indexed.loc["Foundation+Rscore+CMR (Elastic-net)"]
    radiomics = indexed.loc["Rscore+CMR (Elastic-net)"]
    full_vs_radiomics = comparison_lookup(
        key_delong,
        "Rscore+CMR (Elastic-net)",
        "Foundation+Rscore+CMR (Elastic-net)",
    )
    foundation_cmr_vs_xgb = comparison_lookup(
        key_delong,
        "Foundation+CMR (Elastic-net)",
        "Rscore+CMR (XGBoost)",
    )
    lines = [
        "# 神经网络与五种传统分类器完整对照",
        "",
        "## 对照口径",
        "",
        "- 神经网络结果是冻结预训练 cardiac-CMR encoder 后提取的患者级 embedding，",
        "  再连接 Elastic-net Logistic Regression；当前没有进行端到端微调。",
        "- 表征消融实验使用 1,050 例完整开发集和 309 例外部测试集。",
        "- 五分类器实验使用全部 1,059 例开发集和同一组 309 例外部测试集，",
        "  输入统一为重建 OOF Rscore + 四项 CMR 参数。",
        "- 因开发集样本数和分类器头不同，开发集 OOF 结果应分组解释；外部测试集",
        "  患者完全一致，可进行患者配对 DeLong 比较。",
        "",
        "## 外部测试完整结果",
        "",
        "| 实验 | AUC | 敏感度 | 特异度 | NPV | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in external.to_dict("records"):
        lines.append(
            f"| {row['Model']} | {row['AUC']:.4f} | {row['Sensitivity']:.4f} | "
            f"{row['Specificity']:.4f} | {row['NPV']:.4f} | {row['Brier']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 鲜明结论",
            "",
            f"1. Foundation 单独使用时表现最弱：外部 AUC {foundation['AUC']:.3f}、"
            f"特异度 {foundation['Specificity']:.3f}，说明当前冻结表征不能单独替代影像组学。",
            f"2. 加入 CMR 后 Foundation AUC 提升至 {foundation_cmr['AUC']:.3f}，但仍低于"
            f" XGBoost 的 {selected['AUC']:.3f}；配对差值为 "
            f"{foundation_cmr_vs_xgb['Delta AUC']:+.3f}，P={foundation_cmr_vs_xgb['P value']:.3g}。",
            f"3. Rscore+CMR 的 Elastic-net 外部 AUC 为 {radiomics['AUC']:.3f}；加入 Foundation"
            f" 后为 {full['AUC']:.3f}，增益仅 {full_vs_radiomics['Delta AUC']:+.3f}，"
            f"P={full_vs_radiomics['P value']:.3g}，没有显著增量价值。",
            f"4. 完整模型的敏感度最高（{full['Sensitivity']:.3f}），但特异度降至"
            f" {full['Specificity']:.3f}；高敏感度是以明显增加假阳性为代价。",
            f"5. 五分类器中按开发集预设规则选择 XGBoost；其外部 AUC {selected['AUC']:.3f}、"
            f"敏感度 {selected['Sensitivity']:.3f}、特异度 {selected['Specificity']:.3f}。"
            f" CatBoost 外部点估计 AUC 略高（{catboost['AUC']:.3f}），但不能据此事后改选模型。",
            "6. SVM 外部 AUC 尚可，但锁定阈值迁移失败，敏感度接近零；因此不能仅凭 AUC"
            "判断模型是否适合规则排除场景。",
            "",
            "## 文件",
            "",
            "- `complete_external_comparison.csv`：10组外部测试指标及AUC置信区间。",
            "- `complete_development_comparison.csv`：两组开发集OOF结果。",
            "- `complete_external_delong_pairwise.csv`：全部45组患者配对DeLong检验。",
            "- `key_cross_family_delong.csv`：神经网络与传统分类器的重点比较。",
            "- `external_auc_forest.png`：统一AUC森林图。",
            "- `external_operating_points.png`：锁定阈值下敏感度-特异度对照图。",
        ]
    )
    (output_dir / "COMPLETE_COMPARISON_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    """
    主程序：汇总基础模型表征消融实验和五种传统分类器实验的结果。
    生成整合的外部比较表、开发集对比表以及跨类配对 DeLong 检验和相关可视化图表。
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    foundation = foundation_external_table(args.foundation_dir)
    classifiers = classifier_external_table(args.classifier_dir)
    external = pd.concat([foundation, classifiers], ignore_index=True)
    development = development_table(args.foundation_dir, args.classifier_dir)
    predictions = combined_external_predictions(args.foundation_dir, args.classifier_dir)
    delong = complete_delong_table(predictions)
    key_delong = key_delong_table(delong)

    external.to_csv(args.output_dir / "complete_external_comparison.csv", index=False)
    development.to_csv(args.output_dir / "complete_development_comparison.csv", index=False)
    predictions.to_csv(args.output_dir / "combined_external_predictions.csv", index=False)
    delong.to_csv(args.output_dir / "complete_external_delong_pairwise.csv", index=False)
    key_delong.to_csv(args.output_dir / "key_cross_family_delong.csv", index=False)
    plot_auc_forest(external, args.output_dir / "external_auc_forest.png")
    plot_operating_points(external, args.output_dir / "external_operating_points.png")
    write_report(args.output_dir, external, key_delong)
    print(external.to_string(index=False))


if __name__ == "__main__":
    main()
