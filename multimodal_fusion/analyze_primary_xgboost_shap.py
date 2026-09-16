#!/usr/bin/env python3
"""Generate exact TreeSHAP summaries for the locked primary XGBoost model."""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
PREDICTORS = ["Rscore", "MWT", "maxLAV", "LVGRS", "LVGLS"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results/tabular_five_models_icc080_5x5_v1",
    )
    parser.add_argument(
        "--rscore-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results/rscore_oof_icc080",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results/primary_xgboost_shap_icc080_v1",
    )
    return parser.parse_args()


def datasets(rscore_dir):
    development = pd.read_excel(PROJECT_ROOT / "CMR parameter/train.xlsx")
    external = pd.read_excel(PROJECT_ROOT / "CMR parameter/external.xlsx")
    radiomics = pd.read_excel(PROJECT_ROOT / "Radiomics/Radiomics_train.xlsx")
    bundle = joblib.load(rscore_dir / "final_rscore_model.joblib")
    locked_score = bundle["model"].decision_function(radiomics[bundle["features"]])
    development["Rscore"] = pd.Series(
        locked_score, index=radiomics["ID"].astype(int)
    ).reindex(development["ID"].astype(int)).to_numpy()
    external_score = pd.read_csv(rscore_dir / "external_rscore.csv").set_index("ID")["Rscore"]
    external["Rscore"] = external["ID"].map(external_score)
    return {"development": development, "external_test": external}


def tree_shap(model, values):
    booster = model.get_booster()
    matrix = xgb.DMatrix(values, feature_names=PREDICTORS)
    contributions = booster.predict(matrix, pred_contribs=True)
    if contributions.shape[1] != len(PREDICTORS) + 1:
        raise ValueError("Unexpected TreeSHAP contribution shape")
    return contributions[:, :-1], contributions[:, -1]


def beeswarm(values, contributions, title, path):
    importance = np.abs(contributions).mean(axis=0)
    order = np.argsort(importance)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    random = np.random.default_rng(42)
    for position, feature_index in enumerate(order):
        feature_values = values[:, feature_index]
        normalized = (feature_values - np.nanpercentile(feature_values, 5)) / max(
            np.nanpercentile(feature_values, 95) - np.nanpercentile(feature_values, 5),
            1e-9,
        )
        jitter = random.normal(0, 0.08, len(values))
        axis.scatter(
            contributions[:, feature_index],
            np.full(len(values), position) + jitter,
            c=np.clip(normalized, 0, 1),
            cmap="coolwarm",
            s=10,
            alpha=0.65,
            edgecolors="none",
        )
    axis.axvline(0, color="0.5", linewidth=1)
    axis.set_yticks(np.arange(len(order)), [PREDICTORS[index] for index in order])
    axis.set(xlabel="TreeSHAP contribution to model log odds", title=title)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.classifier_dir / "final_models/XGBoost.joblib")
    summary_rows = []
    for cohort, table in datasets(args.rscore_dir).items():
        values = table[PREDICTORS].to_numpy(float)
        contributions, base_value = tree_shap(model, values)
        output = table[["ID", "event"]].copy()
        for index, feature in enumerate(PREDICTORS):
            output[f"{feature}_value"] = values[:, index]
            output[f"{feature}_shap"] = contributions[:, index]
            correlation, p_value = spearmanr(values[:, index], contributions[:, index])
            summary_rows.append({
                "cohort": cohort,
                "feature": feature,
                "mean_absolute_shap": np.abs(contributions[:, index]).mean(),
                "median_shap": np.median(contributions[:, index]),
                "value_shap_spearman_rho": correlation,
                "value_shap_p_value": p_value,
                "mean_base_value": base_value.mean(),
            })
        output.to_csv(args.output_dir / f"{cohort}_shap_values.csv", index=False)
        beeswarm(
            values,
            contributions,
            f"Locked XGBoost TreeSHAP: {cohort.replace('_', ' ')}",
            args.output_dir / f"{cohort}_shap_beeswarm.png",
        )
    summary = pd.DataFrame(summary_rows)
    summary["importance_rank"] = summary.groupby("cohort")["mean_absolute_shap"].rank(
        method="first", ascending=False
    )
    summary.to_csv(args.output_dir / "global_shap_summary.csv", index=False)
    print(summary.sort_values(["cohort", "importance_rank"]).to_string(index=False))


if __name__ == "__main__":
    main()
