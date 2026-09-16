#!/usr/bin/env python3
"""Compare Rscore definitions built with ICC thresholds of 0.75 and 0.80."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_additional_statistics import delong_tables


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--icc075-dir", type=Path, default=EXPERIMENT_ROOT / "results/rscore_oof"
    )
    parser.add_argument(
        "--icc080-dir", type=Path, default=EXPERIMENT_ROOT / "results/rscore_oof_icc080"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results/reviewer_supplement_v1/rscore_icc_sensitivity",
    )
    return parser.parse_args()


def load_predictions(directory, dataset):
    if dataset == "Development OOF":
        path = directory / "development_rscore_oof.csv"
        column = "Rscore_probability_OOF"
    else:
        path = directory / "external_rscore.csv"
        column = "Rscore_probability"
    table = pd.read_csv(path).sort_values("ID").reset_index(drop=True)
    return table[["ID", "event"]], table[column].to_numpy(float)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    auc_rows, pair_rows = [], []
    prediction_outputs = []
    for dataset in ["Development OOF", "External test"]:
        labels075, probability075 = load_predictions(args.icc075_dir, dataset)
        labels080, probability080 = load_predictions(args.icc080_dir, dataset)
        if not labels075.equals(labels080):
            raise ValueError(f"Patient alignment differs for {dataset}")
        probabilities = {"ICC>=0.75": probability075, "ICC>=0.80": probability080}
        estimates, pairs = delong_tables(
            dataset, labels075["event"].to_numpy(int), probabilities
        )
        auc_rows.extend(estimates)
        pair_rows.extend(pairs)
        output = labels075.copy()
        output["dataset"] = dataset
        output["ICC075_probability"] = probability075
        output["ICC080_probability"] = probability080
        prediction_outputs.append(output)
    pd.DataFrame(auc_rows).to_csv(args.output_dir / "auc_estimates.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.output_dir / "paired_delong.csv", index=False)
    pd.concat(prediction_outputs, ignore_index=True).to_csv(
        args.output_dir / "paired_predictions.csv", index=False
    )
    metadata = []
    for label, directory in [("ICC>=0.75", args.icc075_dir), ("ICC>=0.80", args.icc080_dir)]:
        values = json.loads((directory / "rscore_metadata.json").read_text())
        metadata.append({
            "definition": label,
            "icc_features": values["icc_features"],
            "correlation_retained": values["correlation_retained"],
            "lasso_nonzero": values["lasso_nonzero"],
            "development_oof_auc": values["development_oof_auc"],
            "external_test_auc": values["external_test_auc"],
        })
    pd.DataFrame(metadata).to_csv(args.output_dir / "definition_summary.csv", index=False)
    print(pd.DataFrame(metadata).to_string(index=False))
    print(pd.DataFrame(pair_rows).to_string(index=False))


if __name__ == "__main__":
    main()
