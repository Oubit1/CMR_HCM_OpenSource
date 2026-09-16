#!/usr/bin/env python3
"""Rebuild a leakage-controlled radiomics score from the available feature tables."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/rscore_oof"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--icc-threshold", type=float, default=0.75)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--c-grid-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_data():
    development = pd.read_excel(PROJECT_ROOT / "Radiomics/Radiomics_train.xlsx")
    external = pd.read_excel(PROJECT_ROOT / "Radiomics/Radiomics_external.xlsx")
    icc = pd.read_csv(PROJECT_ROOT / "Radiomics/ICC/ICC_analysis_results.csv")
    return development, external, icc

#数据有效性校验
def validate_tables(development, external):
    if development["ID"].duplicated().any() or external["ID"].duplicated().any():
        raise ValueError("Radiomics表中存在重复ID")
    development_features = set(development.columns) - {"ID", "event"}
    external_features = set(external.columns) - {"ID", "event"}
    if development_features != external_features:
        raise ValueError("开发集与外部测试集的radiomics特征列不一致")
    if development.isna().any().any() or external.isna().any().any():
        raise ValueError("Radiomics表中存在缺失值")


def stable_features(icc, available_features, threshold):
    selected = icc.loc[
        (icc["ICC_Intra"] >= threshold) & (icc["ICC_Inter"] >= threshold),
        "Feature",
    ].tolist()
    return [feature for feature in selected if feature in available_features]

#fold内相关性去冗余
def remove_redundant_features(data, threshold):
    correlation = data.corr(method="pearson").abs().fillna(0.0)
    np.fill_diagonal(correlation.values, 0.0)
    mean_correlation = correlation.mean(axis=1)
    priority = sorted(data.columns, key=lambda name: (mean_correlation[name], name))
    retained = []
    blocked = set()
    for feature in priority:
        if feature in blocked:
            continue
        retained.append(feature)
        correlated = correlation.columns[correlation.loc[feature] > threshold]
        blocked.update(correlated.tolist())
    return retained

#标准化+LASSO·
def make_pipeline(c_value, seed):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lasso",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=c_value,
                    max_iter=3000,
                    tol=1e-3,
                    random_state=seed,
                ),
            ),
        ]
    )

#选择最优C（One-Standard-Error规则）
def select_c_one_standard_error(data, labels, c_grid, folds, seed, jobs):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for c_value in c_grid:
        scores = -cross_val_score(
            make_pipeline(c_value, seed),
            data,
            labels,
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=jobs,
        )
        rows.append(
            {
                "C": float(c_value),
                "mean_log_loss": float(scores.mean()),
                "standard_error": float(scores.std(ddof=1) / np.sqrt(len(scores))),
            }
        )
    table = pd.DataFrame(rows).sort_values("C")
    minimum_index = table["mean_log_loss"].idxmin()
    cutoff = (
        table.loc[minimum_index, "mean_log_loss"]
        + table.loc[minimum_index, "standard_error"]
    )
    selected_c = table.loc[table["mean_log_loss"] <= cutoff, "C"].min()
    table["selected_one_se"] = np.isclose(table["C"], selected_c)
    return float(selected_c), table

# LASSO回归特征筛选和模型训练
def fit_rscore(data, labels, stable, args, seed):
    """
    核心评分构建逻辑：
    1. 根据 Pearson 相关系数去除冗余的高相关特征。
    2. 使用带有内层交叉验证的 Elastic Net (L1正则化 Logistic 回归) 寻找最优超参数 C (One-Standard-Error 规则)。
    3. 训练最终的 L1 惩罚模型并提取非零系数特征构建 Rscore。
    """
    retained = remove_redundant_features(
        data[stable], args.correlation_threshold
    )
    grid_size = 4 if args.smoke else args.c_grid_size
    inner_folds = 3 if args.smoke else args.inner_folds
    c_grid = np.logspace(-3, 1, grid_size)
    selected_c, cv_table = select_c_one_standard_error(
        data[retained], labels, c_grid, inner_folds, seed, args.jobs
    )
    model = make_pipeline(selected_c, seed)
    model.fit(data[retained], labels)
    coefficients = model.named_steps["lasso"].coef_.ravel()
    nonzero = [
        feature for feature, coefficient in zip(retained, coefficients)
        if not np.isclose(coefficient, 0.0)
    ]
    return model, retained, nonzero, cv_table


#严格OOF及外部锁定流程
def main():
    """
    主程序：重新构建严格防止数据泄漏的 Rscore (影像组学评分)。
    执行严格的 Outer Fold 交叉验证，在每一折内独立进行相关性特征筛选和 L1 正则化降维。
    最后利用所有的开发集数据拟合出一个锁定参数的最终模型用于预测外部测试集。
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development, external, icc = load_data()
    validate_tables(development, external)

    feature_columns = [column for column in development.columns if column not in {"ID", "event"}]
    stable = stable_features(icc, set(feature_columns), args.icc_threshold)
    if not stable:
        raise RuntimeError("ICC筛选后没有可用特征")

    outer_folds = 3 if args.smoke else args.outer_folds
    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=args.seed
    )
    labels = development["event"].astype(int).to_numpy()
    oof_score = np.full(len(development), np.nan)
    oof_probability = np.full(len(development), np.nan)
    fold_rows = []

    for fold, (train_index, validation_index) in enumerate(
        outer_cv.split(development, labels), start=1
    ):
        print(f"Rscore outer fold {fold}/{outer_folds}", flush=True)
        model, retained, nonzero, _ = fit_rscore(
            development.iloc[train_index],
            labels[train_index],
            stable,
            args,
            args.seed + fold,
        )
        validation = development.iloc[validation_index]
        oof_score[validation_index] = model.decision_function(validation[retained])
        oof_probability[validation_index] = model.predict_proba(validation[retained])[:, 1]
        fold_rows.append(
            {
                "fold": fold,
                "training_n": len(train_index),
                "validation_n": len(validation_index),
                "icc_features": len(stable),
                "correlation_retained": len(retained),
                "lasso_nonzero": len(nonzero),
                "selected_features": json.dumps(nonzero, ensure_ascii=False),
            }
        )

    if np.isnan(oof_probability).any():
        raise RuntimeError("OOF Rscore生成不完整")

    final_model, retained, nonzero, cv_table = fit_rscore(
        development,
        labels,
        stable,
        args,
        args.seed + 1000,
    )
    external_labels = external["event"].astype(int).to_numpy()
    external_score = final_model.decision_function(external[retained])
    external_probability = final_model.predict_proba(external[retained])[:, 1]

    pd.DataFrame(
        {
            "ID": development["ID"].astype(int),
            "event": labels,
            "Rscore_OOF": oof_score,
            "Rscore_probability_OOF": oof_probability,
        }
    ).to_csv(args.output_dir / "development_rscore_oof.csv", index=False)
    pd.DataFrame(
        {
            "ID": external["ID"].astype(int),
            "event": external_labels,
            "Rscore": external_score,
            "Rscore_probability": external_probability,
        }
    ).to_csv(args.output_dir / "external_rscore.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_feature_selection.csv", index=False)
    cv_table.to_csv(args.output_dir / "final_lasso_cv_path.csv", index=False)
    joblib.dump(
        {"model": final_model, "features": retained},
        args.output_dir / "final_rscore_model.joblib",
    )

    scaler = final_model.named_steps["scaler"]
    coefficients = final_model.named_steps["lasso"].coef_.ravel()
    metadata = {
        "development_n": len(development),
        "external_test_n": len(external),
        "icc_threshold": args.icc_threshold,
        "icc_features": len(stable),
        "correlation_threshold": args.correlation_threshold,
        "correlation_retained": len(retained),
        "lasso_nonzero": len(nonzero),
        "selected_C": final_model.named_steps["lasso"].C,
        "selected_features": nonzero,
        "standardized_coefficients": {
            feature: float(coefficient)
            for feature, coefficient in zip(retained, coefficients)
            if not np.isclose(coefficient, 0.0)
        },
        "scaler_mean": dict(zip(retained, scaler.mean_.astype(float))),
        "scaler_scale": dict(zip(retained, scaler.scale_.astype(float))),
        "intercept": float(final_model.named_steps["lasso"].intercept_[0]),
        "development_oof_auc": float(roc_auc_score(labels, oof_probability)),
        "external_test_auc": float(roc_auc_score(external_labels, external_probability)),
    }
    with (args.output_dir / "rscore_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "development_n": metadata["development_n"],
                "external_test_n": metadata["external_test_n"],
                "icc_features": metadata["icc_features"],
                "correlation_retained": metadata["correlation_retained"],
                "lasso_nonzero": metadata["lasso_nonzero"],
                "selected_C": metadata["selected_C"],
                "development_oof_auc": metadata["development_oof_auc"],
                "external_test_auc": metadata["external_test_auc"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
