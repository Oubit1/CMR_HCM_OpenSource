#!/usr/bin/env python3
"""Compare conventional, radiomics and frozen foundation-model representations."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "patient_manifest.csv"
DEFAULT_EMBEDDINGS = EXPERIMENT_ROOT / "results/foundation_embeddings/patient_embeddings.npz"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/foundation_comparison"
DEFAULT_RSCORE_DIR = EXPERIMENT_ROOT / "results/rscore_oof"
CMR_COLUMNS = ["MWT", "LAs", "LVGRS", "LVGLS"]
TARGET_SENSITIVITY = 0.95


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--search-iterations", type=int, default=30)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--exclude-missing-embeddings", action="store_true")
    parser.add_argument(
        "--rscore-mode", choices=["rebuilt", "provided"], default="rebuilt"
    )
    parser.add_argument("--rscore-dir", type=Path, default=DEFAULT_RSCORE_DIR)
    return parser.parse_args()


def select_threshold(labels, probabilities):
    """
    根据灵敏度约束（TARGET_SENSITIVITY=0.95）选取最佳截断值(Threshold)。
    要求灵敏度 >= 0.95 时，使特异度最高。
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, probabilities, drop_intermediate=False
    )
    eligible = np.flatnonzero(true_positive_rate >= TARGET_SENSITIVITY)
    specificity = 1.0 - false_positive_rate
    best_specificity = specificity[eligible].max()
    candidates = eligible[np.isclose(specificity[eligible], best_specificity)]
    finite = candidates[np.isfinite(thresholds[candidates])]
    index = finite[np.argmax(thresholds[finite])] if len(finite) else candidates[0]
    return float(np.clip(thresholds[index], 0.0, 1.0))


def calculate_metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    safe_divide = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    return {
        "AUC": roc_auc_score(labels, probabilities),
        "Accuracy": accuracy_score(labels, predictions),
        "Sensitivity": safe_divide(true_positive, true_positive + false_negative),
        "Specificity": safe_divide(true_negative, true_negative + false_positive),
        "PPV": safe_divide(true_positive, true_positive + false_positive),
        "NPV": safe_divide(true_negative, true_negative + false_negative),
        "F1": f1_score(labels, predictions, zero_division=0),
        "Brier": brier_score_loss(labels, probabilities),
    }


def make_model(seed):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=5000,
                    tol=1e-3,
                    random_state=seed,
                ),
            ),
        ]
    )


def tune_model(features, labels, folds, iterations, seed, jobs):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        make_model(seed),
        {
            "model__C": loguniform(1e-3, 100),
            "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
        n_iter=iterations,
        scoring="roc_auc",
        refit=True,
        cv=cv,
        random_state=seed,
        n_jobs=jobs,
        error_score="raise",
    )
    search.fit(features, labels)
    return search.best_estimator_, search.best_params_


def load_embedding_table(path):
    data = np.load(path, allow_pickle=True)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    table = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{index:03d}" for index in range(embeddings.shape[1])],
    )
    table.insert(0, "ID", data["patient_id"].astype(int))
    table.insert(1, "cohort", data["cohort"].astype(str))
    return table


def build_feature_sets(manifest, embedding_table, exclude_missing):
    missing = manifest.merge(
        embedding_table[["cohort", "ID"]], on=["cohort", "ID"], how="left", indicator=True
    )
    missing = missing.loc[missing["_merge"] == "left_only", ["cohort", "ID", "event"]].copy()
    missing["reason"] = "excluded_no_valid_sax_cine"
    if len(missing) and not exclude_missing:
        raise ValueError(f"缺少患者embedding，共{len(missing)}例")
    merged = manifest.merge(
        embedding_table,
        on=["cohort", "ID"],
        how="inner",
        validate="one_to_one",
    )
    embedding_columns = [column for column in merged.columns if column.startswith("embedding_")]
    feature_sets = {
        "CMR": CMR_COLUMNS,
        "Rscore+CMR": ["Rscore", *CMR_COLUMNS],
        "Foundation": embedding_columns,
        "Foundation+CMR": [*embedding_columns, *CMR_COLUMNS],
        "Foundation+Rscore+CMR": [*embedding_columns, "Rscore", *CMR_COLUMNS],
    }
    return merged, feature_sets, missing


def apply_rebuilt_rscore(development, external, rscore_dir):
    development_oof = pd.read_csv(rscore_dir / "development_rscore_oof.csv")
    external_rscore = pd.read_csv(rscore_dir / "external_rscore.csv")
    model_bundle = joblib.load(rscore_dir / "final_rscore_model.joblib")
    radiomics_development = pd.read_excel(PROJECT_ROOT / "Radiomics/Radiomics_train.xlsx")

    locked_score = model_bundle["model"].decision_function(
        radiomics_development[model_bundle["features"]]
    )
    oof_map = development_oof.set_index("ID")["Rscore_OOF"]
    locked_map = pd.Series(
        locked_score,
        index=radiomics_development["ID"].astype(int),
    )
    external_map = external_rscore.set_index("ID")["Rscore"]

    development_oof_data = development.copy()
    development_final_data = development.copy()
    external_data = external.copy()
    development_oof_data["Rscore"] = development_oof_data["ID"].map(oof_map)
    development_final_data["Rscore"] = development_final_data["ID"].map(locked_map)
    external_data["Rscore"] = external_data["ID"].map(external_map)

    for name, table in {
        "development OOF": development_oof_data,
        "development final": development_final_data,
        "external test": external_data,
    }.items():
        if table["Rscore"].isna().any():
            raise ValueError(f"{name} 中存在无法匹配的重建Rscore")
    return development_oof_data, development_final_data, external_data


def bootstrap_intervals(labels, probabilities, threshold, iterations, seed):
    random = np.random.RandomState(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    values = []
    for _ in range(iterations):
        indices = np.concatenate(
            [
                random.choice(negative, len(negative), replace=True),
                random.choice(positive, len(positive), replace=True),
            ]
        )
        values.append(calculate_metrics(labels[indices], probabilities[indices], threshold))
    intervals = {}
    for metric in values[0]:
        lower, upper = np.percentile([row[metric] for row in values], [2.5, 97.5])
        intervals[metric] = [float(lower), float(upper)]
    return intervals


def main():
    """
    主程序：对比传统的临床参数/影像组学(Rscore+CMR)和 Foundation Model 提取的 Embeddings。
    分别使用不同的特征组合跑随机搜索超参数，采用嵌套交叉验证计算 OOF 指标。
    并在外部验证集上报告各项性能指标。
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    embeddings = load_embedding_table(args.embeddings)
    data, feature_sets, excluded = build_feature_sets(
        manifest, embeddings, args.exclude_missing_embeddings
    )
    excluded.to_csv(args.output_dir / "excluded_cases.csv", index=False)
    development = data.loc[data["cohort"] == "development"].reset_index(drop=True)
    external = data.loc[data["cohort"] == "external_test"].reset_index(drop=True)
    development_final = development.copy()
    if args.rscore_mode == "rebuilt":
        development, development_final, external = apply_rebuilt_rscore(
            development, external, args.rscore_dir
        )
    development_labels = development["event"].astype(int).to_numpy()
    external_labels = external["event"].astype(int).to_numpy()
    outer_cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    summary_rows = []
    development_predictions = development[["ID", "event"]].copy()
    external_predictions = external[["ID", "event"]].copy()

    for model_index, (model_name, feature_columns) in enumerate(feature_sets.items()):
        print(f"Running {model_name} ({len(feature_columns)} features)", flush=True)
        development_features = development[feature_columns].astype(float)
        final_development_features = development_final[feature_columns].astype(float)
        external_features = external[feature_columns].astype(float)
        oof_probability = np.full(len(development), np.nan)

        for fold, (train_index, validation_index) in enumerate(
            outer_cv.split(development_features, development_labels), start=1
        ):
            best_model, _ = tune_model(
                development_features.iloc[train_index],
                development_labels[train_index],
                args.inner_folds,
                args.search_iterations,
                args.seed + model_index * 100 + fold,
                args.jobs,
            )
            oof_probability[validation_index] = best_model.predict_proba(
                development_features.iloc[validation_index]
            )[:, 1]

        locked_threshold = select_threshold(development_labels, oof_probability)
        development_metrics = calculate_metrics(
            development_labels, oof_probability, locked_threshold
        )
        final_model, best_parameters = tune_model(
            final_development_features,
            development_labels,
            args.inner_folds,
            args.search_iterations,
            args.seed + model_index * 1000,
            args.jobs,
        )
        final_model.fit(final_development_features, development_labels)
        external_probability = final_model.predict_proba(external_features)[:, 1]
        external_metrics = calculate_metrics(
            external_labels, external_probability, locked_threshold
        )
        external_intervals = bootstrap_intervals(
            external_labels,
            external_probability,
            locked_threshold,
            args.bootstraps,
            args.seed + model_index,
        )

        for dataset_name, metrics in (
            ("Development OOF", development_metrics),
            ("External test", external_metrics),
        ):
            summary_rows.append(
                {
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Features": len(feature_columns),
                    "Locked threshold": locked_threshold,
                    "Best parameters": json.dumps(best_parameters, sort_keys=True),
                    **metrics,
                    "Bootstrap 95% CI": json.dumps(external_intervals)
                    if dataset_name == "External test"
                    else "",
                }
            )
        development_predictions[f"{model_name}_probability"] = oof_probability
        external_predictions[f"{model_name}_probability"] = external_probability
        joblib.dump(final_model, model_dir / f"{model_name.replace('+', '_')}.joblib")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "model_comparison.csv", index=False)
    development_predictions.to_csv(
        args.output_dir / "development_oof_predictions.csv", index=False
    )
    external_predictions.to_csv(
        args.output_dir / "external_test_predictions.csv", index=False
    )
    run_config = {
        "manifest": str(args.manifest.resolve()),
        "embeddings": str(args.embeddings.resolve()),
        "development_patients": len(development),
        "external_test_patients": len(external),
        "excluded_patients": len(excluded),
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "search_iterations": args.search_iterations,
        "bootstraps": args.bootstraps,
        "seed": args.seed,
        "complete_case_analysis": args.exclude_missing_embeddings,
        "rscore_mode": args.rscore_mode,
        "rscore_dir": str(args.rscore_dir.resolve()) if args.rscore_mode == "rebuilt" else None,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
