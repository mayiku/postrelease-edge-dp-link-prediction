"""Summarize DPLP's native key_results.pickle files into one CSV.

This script does not rerun DPLP and never modifies its result directories.
Pass exactly one native result directory per dataset.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import statistics
from pathlib import Path
from typing import Any


FIELDS = [
    "dataset",
    "runs",
    "mean_test_auc_pct",
    "sd_test_auc_pct",
    "median_test_auc_pct",
    "mean_best_valid_auc_pct",
    "num_hops",
    "max_node_degree",
    "batch_size",
    "train_samples",
    "target_epsilon",
    "sigma",
    "sensitivity",
    "run_test_auc_pct",
]


def load_result(result_dir: Path) -> dict[str, Any]:
    result_path = result_dir / "key_results.pickle"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    with result_path.open("rb") as handle:
        result = pickle.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"{result_path} does not contain a result dictionary")
    return result


def summarize(result_dir: Path) -> dict[str, Any]:
    result = load_result(result_dir)
    runs = result.get("all_runs", {})
    test_values = [float(value) for value in runs.get("final_test", [])]
    valid_values = [float(value) for value in runs.get("highest_val", [])]
    if not test_values:
        raise ValueError(f"{result_dir} contains no per-run test AUC values")
    if len(test_values) != len(valid_values):
        raise ValueError(f"{result_dir} has mismatched test/validation run counts")

    return {
        "dataset": result["dataset"],
        "runs": len(test_values),
        "mean_test_auc_pct": statistics.mean(test_values),
        "sd_test_auc_pct": (
            statistics.stdev(test_values) if len(test_values) > 1 else 0.0
        ),
        "median_test_auc_pct": statistics.median(test_values),
        "mean_best_valid_auc_pct": statistics.mean(valid_values),
        "num_hops": result["num_hop"],
        "max_node_degree": result["max_node_degree"],
        "batch_size": result["batch_size"],
        "train_samples": result["train_samples"],
        "target_epsilon": result.get("target_epsilon", result.get("epsilon")),
        "sigma": result["sigma"],
        "sensitivity": result.get("sens"),
        "run_test_auc_pct": ";".join(f"{value:.6f}" for value in test_values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        action="append",
        required=True,
        type=Path,
        help="native DPLP result directory containing key_results.pickle",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [summarize(result_dir) for result_dir in args.result_dir]
    datasets = [row["dataset"] for row in rows]
    if len(set(datasets)) != len(datasets):
        raise ValueError(f"duplicate datasets in result directories: {datasets}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["dataset"]))


if __name__ == "__main__":
    main()

