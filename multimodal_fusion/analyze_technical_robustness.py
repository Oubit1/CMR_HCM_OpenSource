#!/usr/bin/env python3
"""Assess associations and model robustness across CMR acquisition factors."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import brier_score_loss, roc_auc_score

from run_additional_statistics import holm_adjust


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_METADATA = (
    EXPERIMENT_ROOT / "results/reviewer_supplement_v1/acquisition_metadata.csv"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/reviewer_supplement_v1/technical_robustness"
CONTINUOUS_FACTORS = [
    "pixel_spacing_row_mm",
    "slice_thickness_mm",
    "spacing_between_slices_mm",
    "repetition_time_ms",
    "echo_time_ms",
    "flip_angle_deg",
    "median_frames",
    "sax_slices",
    "rows",
    "columns",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rebuilt-rscore-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results/rscore_oof",
    )
    parser.add_argument(
        "--comparison-predictions",
        type=Path,
        default=EXPERIMENT_ROOT
        / "results/complete_comparison_v1/combined_external_predictions.csv",
    )
    return parser.parse_args()


def hc3_ols(data, outcome, continuous, categorical, adjustment):
    columns = [outcome, *continuous, *categorical, *adjustment]
    table = data[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    design_parts = [pd.DataFrame({"Intercept": np.ones(len(table))}, index=table.index)]
    for column in continuous:
        standard_deviation = table[column].std(ddof=0)
        if standard_deviation <= 0:
            continue
        design_parts.append(
            pd.DataFrame(
                {
                    column: (table[column] - table[column].mean())
                    / standard_deviation
                },
                index=table.index,
            )
        )
    for column in categorical:
        categories = sorted(table[column].astype(str).unique())
        if len(categories) < 2:
            continue
        encoded = pd.get_dummies(
            table[column].astype(str), prefix=column, drop_first=True, dtype=float
        )
        design_parts.append(encoded)
    for column in adjustment:
        design_parts.append(
            pd.DataFrame({column: table[column].astype(float)}, index=table.index)
        )
    design = pd.concat(design_parts, axis=1)
    matrix = design.to_numpy(dtype=float)
    response = table[outcome].to_numpy(dtype=float)
    inverse = np.linalg.pinv(matrix.T @ matrix)
    coefficients = inverse @ matrix.T @ response
    fitted = matrix @ coefficients
    residual = response - fitted
    leverage = np.sum((matrix @ inverse) * matrix, axis=1)
    adjusted_residual = residual / np.clip(1.0 - leverage, 1e-6, None)
    meat = matrix.T @ (matrix * np.square(adjusted_residual)[:, None])
    covariance = inverse @ meat @ inverse
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    degrees_freedom = max(len(response) - matrix.shape[1], 1)
    statistics = coefficients / np.clip(standard_errors, 1e-12, None)
    p_values = 2.0 * t.sf(np.abs(statistics), degrees_freedom)
    total_sum_squares = np.square(response - response.mean()).sum()
    r_squared = 1.0 - np.square(residual).sum() / total_sum_squares
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (len(response) - 1) / degrees_freedom
    rows = []
    for name, coefficient, standard_error, statistic, p_value in zip(
        design.columns, coefficients, standard_errors, statistics, p_values
    ):
        rows.append(
            {
                "term": name,
                "coefficient": coefficient,
                "robust_se_hc3": standard_error,
                "ci_lower": coefficient - 1.959964 * standard_error,
                "ci_upper": coefficient + 1.959964 * standard_error,
                "t": statistic,
                "p_value": p_value,
                "n": len(response),
                "r_squared": r_squared,
                "adjusted_r_squared": adjusted_r_squared,
            }
        )
    result = pd.DataFrame(rows)
    tested = ~result["term"].isin(["Intercept", *adjustment])
    result["holm_p_value"] = np.nan
    if tested.any():
        result.loc[tested, "holm_p_value"] = holm_adjust(
            result.loc[tested, "p_value"].to_numpy()
        )
    result["significant_after_holm"] = result["holm_p_value"] < 0.05
    return result, table, design


def attach_rscores(metadata, rebuilt_rscore_dir):
    manifest = pd.read_csv(EXPERIMENT_ROOT / "patient_manifest.csv")
    result = metadata.merge(
        manifest[["cohort", "ID", "Rscore"]],
        on=["cohort", "ID"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"Rscore": "legacy_rscore"})
    development = pd.read_csv(
        rebuilt_rscore_dir / "development_rscore_oof.csv"
    )[["ID", "Rscore_OOF"]]
    external = pd.read_csv(rebuilt_rscore_dir / "external_rscore.csv")[
        ["ID", "Rscore"]
    ].rename(columns={"Rscore": "Rscore_external"})
    result = result.merge(
        development.rename(columns={"Rscore_OOF": "rebuilt_rscore"}),
        on="ID",
        how="left",
    )
    external_map = external.set_index("ID")["Rscore_external"]
    external_rows = result["cohort"] == "external_test"
    result.loc[external_rows, "rebuilt_rscore"] = result.loc[
        external_rows, "ID"
    ].map(external_map)
    return result


def acquisition_summary(data, output_dir):
    rows = []
    for cohort, table in data.groupby("cohort"):
        for variable in ["field_strength_t", *CONTINUOUS_FACTORS]:
            values = table[variable].dropna()
            rows.append(
                {
                    "cohort": cohort,
                    "variable": variable,
                    "n": len(values),
                    "missing": int(table[variable].isna().sum()),
                    "mean": values.mean(),
                    "standard_deviation": values.std(),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "acquisition_factor_summary.csv", index=False)
    field_counts = (
        data.assign(field_strength_t=data["field_strength_t"].astype(str))
        .groupby(["cohort", "field_strength_t"], dropna=False)
        .agg(n=("ID", "size"), events=("event", "sum"))
        .reset_index()
    )
    field_counts.to_csv(output_dir / "field_strength_counts.csv", index=False)
    return summary, field_counts


def rscore_regressions(data, output_dir):
    development = data.loc[data["cohort"] == "development"].copy()
    available = [
        column
        for column in CONTINUOUS_FACTORS
        if development[column].notna().mean() >= 0.80
        and development[column].nunique(dropna=True) > 1
    ]
    categorical = []
    if development["field_strength_t"].notna().mean() >= 0.80:
        development["field_strength_group"] = development["field_strength_t"].map(
            lambda value: f"{value:g}T" if pd.notna(value) else np.nan
        )
        categorical.append("field_strength_group")
    outputs = {}
    for outcome in ["legacy_rscore", "rebuilt_rscore"]:
        result, complete, design = hc3_ols(
            development,
            outcome,
            available,
            categorical,
            adjustment=["event"],
        )
        result.insert(0, "outcome", outcome)
        result.to_csv(output_dir / f"{outcome}_technical_regression.csv", index=False)
        outputs[outcome] = {
            "n": len(complete),
            "predictors": design.columns.tolist(),
            "adjusted_r_squared": float(result["adjusted_r_squared"].iloc[0]),
            "significant_technical_terms_after_holm": result.loc[
                result["significant_after_holm"], "term"
            ].tolist(),
        }
    return outputs


def subgroup_metrics(metadata, output_dir, comparison_predictions):
    predictions = pd.read_csv(comparison_predictions)
    external = metadata.loc[metadata["cohort"] == "external_test"].merge(
        predictions, on=["ID", "event"], validate="one_to_one"
    )
    external["field_strength_group"] = external["field_strength_t"].map(
        lambda value: f"{value:g}T" if pd.notna(value) else "Missing"
    )
    external["manufacturer_group"] = (
        external["manufacturer"]
        .astype(str)
        .str.upper()
        .replace({"SIEMENS": "Siemens", "GE MEDICAL SYSTEMS": "GE", "NAN": "Missing"})
    )
    external["inplane_spacing_group"] = pd.qcut(
        external["pixel_spacing_row_mm"],
        q=2,
        labels=["Lower in-plane spacing", "Higher in-plane spacing"],
        duplicates="drop",
    ).astype(str)
    rows = []
    probability_columns = [
        column for column in predictions.columns if column.endswith("_probability")
    ]
    for subgroup_variable in [
        "field_strength_group",
        "manufacturer_group",
        "inplane_spacing_group",
    ]:
        for subgroup, table in external.groupby(subgroup_variable, dropna=False):
            labels = table["event"].to_numpy()
            for column in probability_columns:
                if len(np.unique(labels)) < 2:
                    auc = np.nan
                else:
                    auc = roc_auc_score(labels, table[column])
                rows.append(
                    {
                        "subgroup_variable": subgroup_variable,
                        "subgroup": subgroup,
                        "model": column.removesuffix("_probability"),
                        "n": len(table),
                        "events": int(labels.sum()),
                        "nonevents": int((labels == 0).sum()),
                        "auc": auc,
                        "brier": brier_score_loss(labels, table[column]),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "external_model_subgroup_metrics.csv", index=False)
    return result


def write_summary(output_dir, metadata, field_counts, regressions, subgroup):
    lines = [
        "# 技术因素与亚组稳健性分析",
        "",
        f"- 成功提取 {int((metadata['metadata_status'] == 'ok').sum())}/{len(metadata)} 例的所选SAX序列DICOM参数。",
        "- 开发集厂商/型号字段均被匿名化为 `Ano`；外部测试集保留了部分厂商信息，故仅能在外部队列进行探索性厂商亚组分析。",
        "- Rscore技术因素回归在开发集完成，并将LGE事件状态作为生物学调整变量；连续技术变量按1个标准差标准化，使用HC3稳健标准误和Holm多重校正。",
    ]
    for row in field_counts.to_dict("records"):
        lines.append(
            f"- {row['cohort']}，{row['field_strength_t']}：n={row['n']}，事件={row['events']}。"
        )
    for outcome, result in regressions.items():
        terms = result["significant_technical_terms_after_holm"]
        lines.append(
            f"- {outcome}：完整病例n={result['n']}，校正后R²={result['adjusted_r_squared']:.3f}，"
            f"Holm校正后显著技术项={terms if terms else '无'}。"
        )
    valid_subgroups = subgroup.loc[
        (subgroup["events"] >= 10) & (subgroup["nonevents"] >= 10)
    ]
    lines.append(
        f"- 外部测试中满足阳性和阴性均≥10例的模型×亚组结果共 {len(valid_subgroups)} 条；详见CSV。"
    )
    lines.extend(
        [
            "",
            "注意：场强、厂商和空间分辨率亚组属于补充稳健性分析，不应据此重新选择模型或阈值。开发集均为3.0T，无法在开发集内估计场强效应；层间距和层数与Rscore的关联也可能混有解剖覆盖范围因素，不能直接解释为扫描仪偏倚。",
        ]
    )
    (output_dir / "TECHNICAL_ROBUSTNESS_SUMMARY_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    metadata = attach_rscores(metadata, args.rebuilt_rscore_dir)
    metadata.to_csv(args.output_dir / "analysis_dataset.csv", index=False)
    _, field_counts = acquisition_summary(metadata, args.output_dir)
    regressions = rscore_regressions(metadata, args.output_dir)
    subgroup = subgroup_metrics(metadata, args.output_dir, args.comparison_predictions)
    write_summary(args.output_dir, metadata, field_counts, regressions, subgroup)
    run_config = {
        "metadata": str(args.metadata),
        "rebuilt_rscore_dir": str(args.rebuilt_rscore_dir),
        "comparison_predictions": str(args.comparison_predictions),
        "continuous_factors": CONTINUOUS_FACTORS,
        "rscore_regression_adjustment": ["event"],
        "standard_errors": "HC3",
        "multiple_testing": "Holm within each Rscore regression",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
