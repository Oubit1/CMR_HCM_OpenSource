# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict


# 默认路径配置 (可由命令行参数覆盖，支持相对路径)
# 请将数据放置在 ./data/CMR_parameter/ 或通过命令行传入实际路径
DATA_DIR = Path("./data/CMR_parameter")
TRAIN_FILE = DATA_DIR / "train.xlsx"        # 开发集表格路径 (含 MWT, LAs, LVGRS, LVGLS, Rscore, event)
EXTERNAL_FILE = DATA_DIR / "external.xlsx"  # 外部独立测试集表格路径
RESULT_DIR = Path("./results/catboost_model") # 模型评估与权重保存路径

LABEL = "event"
FEATURES_WITH = ["Rscore", "MWT", "LAs", "LVGRS", "LVGLS"]
FEATURES_WITHOUT = ["MWT", "LAs", "LVGRS", "LVGLS"]
TARGET_SENSITIVITY = 0.95
CV_FOLDS = 5
N_ITER = 30
SEED = 42
N_JOBS = -1


def specificity_at_target(y_true, probability):
    fpr, tpr, _ = roc_curve(y_true, probability, drop_intermediate=False)
    eligible = tpr >= TARGET_SENSITIVITY
    return float(np.max(1 - fpr[eligible])) if eligible.any() else 0.0


def specificity_scorer(estimator, X, y):
    return specificity_at_target(y, estimator.predict_proba(X)[:, 1])


def select_refit(cv_results):
    table = pd.DataFrame({
        "index": np.arange(len(cv_results["params"])),
        "specificity": cv_results["mean_test_specificity"],
        "auc": cv_results["mean_test_auc"],
        "neg_brier": cv_results["mean_test_neg_brier"],
    })
    return int(table.sort_values(
        ["specificity", "auc", "neg_brier"], ascending=False
    ).iloc[0]["index"])


def select_threshold(y_true, probability):
    fpr, tpr, thresholds = roc_curve(
        y_true, probability, drop_intermediate=False
    )
    eligible = np.flatnonzero(tpr >= TARGET_SENSITIVITY)
    specificity = 1 - fpr
    best = specificity[eligible].max()
    candidates = eligible[np.isclose(specificity[eligible], best)]
    finite = candidates[np.isfinite(thresholds[candidates])]
    index = finite[np.argmax(thresholds[finite])] if len(finite) else candidates[0]
    return float(thresholds[index])


def calculate_metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y_true, probability),
        "Brier score": brier_score_loss(y_true, probability),
        "Threshold": threshold,
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
        "PPV": tp / (tp + fp),
        "NPV": tn / (tn + fn),
        "Accuracy": accuracy_score(y_true, prediction),
        "F1 score": f1_score(y_true, prediction),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def make_model(seed):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        bootstrap_type="Bernoulli",
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )


def json_parameters(parameters):
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in parameters.items()
    }


def fit_model(data, external_data, features, cv, parameter_space, model_file):
    X = data[features].astype(float)
    y = data[LABEL].astype(int).to_numpy()
    X_external = external_data[features].astype(float)
    y_external = external_data[LABEL].astype(int).to_numpy()

    search = RandomizedSearchCV(
        estimator=make_model(SEED),
        param_distributions=parameter_space,
        n_iter=N_ITER,
        scoring={
            "specificity": specificity_scorer,
            "auc": "roc_auc",
            "neg_brier": "neg_brier_score",
        },
        refit=select_refit,
        cv=cv,
        random_state=SEED,
        n_jobs=N_JOBS,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X, y)

    locked_model = clone(search.best_estimator_)
    oof_probability = cross_val_predict(
        locked_model, X, y, cv=cv, method="predict_proba", n_jobs=N_JOBS
    )[:, 1]
    threshold = select_threshold(y, oof_probability)
    oof_prediction = (oof_probability >= threshold).astype(int)

    final_model = clone(search.best_estimator_).set_params(thread_count=-1)
    final_model.fit(X, y)
    final_model.save_model(str(RESULT_DIR / model_file))

    external_probability = final_model.predict_proba(X_external)[:, 1]
    external_prediction = (external_probability >= threshold).astype(int)

    tuning = pd.DataFrame(search.cv_results_)[[
        "rank_test_specificity",
        "mean_test_specificity",
        "mean_test_auc",
        "mean_test_neg_brier",
        "params",
    ]].sort_values("rank_test_specificity")
    tuning["mean_test_brier"] = -tuning.pop("mean_test_neg_brier")
    tuning["params"] = tuning["params"].apply(
        lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, default=float)
    )

    return {
        "features": features,
        "parameters": json_parameters(search.best_params_),
        "threshold": threshold,
        "oof_probability": oof_probability,
        "oof_prediction": oof_prediction,
        "external_probability": external_probability,
        "external_prediction": external_prediction,
        "oof_metrics": calculate_metrics(y, oof_probability, threshold),
        "external_metrics": calculate_metrics(
            y_external, external_probability, threshold
        ),
        "importance": final_model.get_feature_importance(),
        "tuning": tuning,
    }


def main():
    np.random.seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(TRAIN_FILE)
    external_data = pd.read_excel(EXTERNAL_FILE)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    parameter_space = {
        "iterations": randint(250, 901),
        "learning_rate": loguniform(0.005, 0.08),
        "depth": randint(3, 8),
        "l2_leaf_reg": loguniform(0.5, 20),
        "random_strength": uniform(0, 3),
        "border_count": [64, 128, 254],
        "subsample": uniform(0.60, 0.40),
        "scale_pos_weight": [1.0],
    }

    with_rscore = fit_model(
        data,
        external_data,
        FEATURES_WITH,
        cv,
        parameter_space,
        "CatBoost_final.cbm",
    )
    without_rscore = fit_model(
        data,
        external_data,
        FEATURES_WITHOUT,
        cv,
        parameter_space,
        "CatBoost_without_Rscore.cbm",
    )

    identifiers = [column for column in ["ID", "name"] if column in data.columns]
    training_predictions = data[identifiers].copy()
    training_predictions["true_label"] = data[LABEL].astype(int).to_numpy()
    training_predictions["predicted_probability"] = with_rscore["oof_probability"]
    training_predictions["predicted_label"] = with_rscore["oof_prediction"]
    training_predictions["predicted_probability_without_Rscore"] = without_rscore["oof_probability"]
    training_predictions["predicted_label_without_Rscore"] = without_rscore["oof_prediction"]

    external_identifiers = [
        column for column in ["ID", "name"] if column in external_data.columns
    ]
    external_predictions = external_data[external_identifiers].copy()
    external_predictions["true_label"] = external_data[LABEL].astype(int).to_numpy()
    external_predictions["predicted_probability"] = with_rscore["external_probability"]
    external_predictions["predicted_label"] = with_rscore["external_prediction"]
    external_predictions["predicted_probability_without_Rscore"] = without_rscore["external_probability"]
    external_predictions["predicted_label_without_Rscore"] = without_rscore["external_prediction"]

    training_predictions.to_csv(
        RESULT_DIR / "prediction_results_training.csv", index=False, encoding="utf-8-sig"
    )
    external_predictions.to_csv(
        RESULT_DIR / "prediction_results_external.csv", index=False, encoding="utf-8-sig"
    )

    model_rows = []
    performance_rows = []
    importance_rows = []
    tuning_rows = []
    for model_name, result in [
        ("With Rscore", with_rscore),
        ("Without Rscore", without_rscore),
    ]:
        model_rows.append({
            "Model": model_name,
            "Predictors": ", ".join(result["features"]),
            "Locked threshold": result["threshold"],
            "Best parameters": json.dumps(
                result["parameters"], ensure_ascii=False, sort_keys=True
            ),
        })
        performance_rows.extend([
            {"Model": model_name, "Dataset": "Training OOF", **result["oof_metrics"]},
            {"Model": model_name, "Dataset": "External validation", **result["external_metrics"]},
        ])
        importance_rows.extend([
            {"Model": model_name, "Feature": feature, "Importance": importance}
            for feature, importance in zip(result["features"], result["importance"])
        ])
        table = result["tuning"].copy()
        table.insert(0, "Model", model_name)
        tuning_rows.append(table)

    model_table = pd.DataFrame(model_rows)
    performance = pd.DataFrame(performance_rows)
    importance = pd.DataFrame(importance_rows).sort_values(
        ["Model", "Importance"], ascending=[True, False]
    )
    tuning = pd.concat(tuning_rows, ignore_index=True)

    with pd.ExcelWriter(
        RESULT_DIR / "CatBoost_final_results.xlsx", engine="openpyxl"
    ) as writer:
        model_table.to_excel(writer, sheet_name="Final models", index=False)
        performance.to_excel(writer, sheet_name="Performance", index=False)
        training_predictions.to_excel(writer, sheet_name="Training OOF", index=False)
        external_predictions.to_excel(writer, sheet_name="External predictions", index=False)
        importance.to_excel(writer, sheet_name="Feature importance", index=False)
        tuning.to_excel(writer, sheet_name="Final tuning", index=False)

    metadata = {
        "outcome": LABEL,
        "target_sensitivity": TARGET_SENSITIVITY,
        "cv_folds": CV_FOLDS,
        "seed": SEED,
        "with_Rscore": {
            "features": with_rscore["features"],
            "locked_threshold": with_rscore["threshold"],
            "best_parameters": with_rscore["parameters"],
        },
        "without_Rscore": {
            "features": without_rscore["features"],
            "locked_threshold": without_rscore["threshold"],
            "best_parameters": without_rscore["parameters"],
        },
    }
    with open(
        RESULT_DIR / "CatBoost_final_metadata.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(performance[[
        "Model", "Dataset", "AUC", "Sensitivity", "Specificity", "NPV", "Brier score"
    ]].to_string(index=False))
    print("\nLocked thresholds:")
    print(f"With Rscore: {with_rscore['threshold']:.6f}")
    print(f"Without Rscore: {without_rscore['threshold']:.6f}")
    print(f"\nResults: {RESULT_DIR}")


if __name__ == "__main__":
    main()
