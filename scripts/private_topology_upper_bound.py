#!/usr/bin/env python
"""Nonprivate upper bound using the same public and topology features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postrelease_lp import (  # noqa: E402
    feature_candidates,
    graph_edges,
    load_graph_data,
    metrics,
    pair_features_for,
    public_projected_features,
    split_bucket,
    split_name,
    summarize,
    topology_features,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-salt", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13000])
    parser.add_argument("--target-candidates", type=int, default=20)
    parser.add_argument("--incidence-cap", type=int, default=16)
    parser.add_argument("--topology-dimension", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph, raw_features = load_graph_data(args.data)
    edges = graph_edges(graph)
    pairs = feature_candidates(raw_features, args.target_candidates)
    workload = {
        name: [
            pair
            for pair in pairs
            if split_name(split_bucket(pair, args.split_salt)) == name
        ]
        for name in ("train", "valid", "test")
    }
    labels = {
        name: np.asarray(
            [pair in edges for pair in values],
            dtype=np.float32,
        )
        for name, values in workload.items()
    }
    projected = public_projected_features(raw_features)
    norms = np.sqrt(
        np.asarray(
            raw_features.multiply(raw_features).sum(axis=1)
        ).ravel()
    ).astype(np.float32)
    public = {
        name: pair_features_for(
            raw_features,
            projected,
            norms,
            values,
        )
        for name, values in workload.items()
    }
    topology = {
        name: topology_features(
            workload[name],
            sorted(edges),
            graph.shape[0],
            args.incidence_cap,
        )[:, : args.topology_dimension]
        for name in workload
    }
    joint = {
        name: np.concatenate(
            [public[name], topology[name]],
            axis=1,
        )
        for name in workload
    }
    train_weights = np.where(
        labels["train"] > 0,
        10.0,
        1.0,
    ).astype(np.float32)
    validation_weights = np.where(
        labels["valid"] > 0,
        10.0,
        1.0,
    ).astype(np.float32)

    runs = []
    for seed in args.seeds:
        public_scores = train_model(
            public["train"],
            public["test"],
            labels["train"],
            train_weights,
            seed,
            "f3",
            selection="released-validation",
            validation_features=public["valid"],
            validation_target=labels["valid"],
            validation_weights=validation_weights,
        )
        joint_scores = train_model(
            joint["train"],
            joint["test"],
            labels["train"],
            train_weights,
            seed,
            "f3",
            selection="released-validation",
            validation_features=joint["valid"],
            validation_target=labels["valid"],
            validation_weights=validation_weights,
        )
        run = {
            "seed": seed,
            "public": metrics(public_scores, labels["test"]),
            "joint": metrics(joint_scores, labels["test"]),
        }
        run["joint_minus_public"] = (
            run["joint"]["auc"] - run["public"]["auc"]
        )
        runs.append(run)
        print(json.dumps(run, ensure_ascii=False), flush=True)

    result = {
        "dataset": args.dataset,
        "split_salt": args.split_salt,
        "privacy": "nonprivate upper bound",
        "topology_dimension": args.topology_dimension,
        "label_counts": {
            name: {
                "total": int(len(values)),
                "positive": int(values.sum()),
            }
            for name, values in labels.items()
        },
        "runs": runs,
        "summary": {
            "public_auc": summarize(
                [run["public"]["auc"] for run in runs]
            ),
            "joint_auc": summarize(
                [run["joint"]["auc"] for run in runs]
            ),
            "public_ap": summarize(
                [
                    run["public"]["average_precision"]
                    for run in runs
                ]
            ),
            "joint_ap": summarize(
                [
                    run["joint"]["average_precision"]
                    for run in runs
                ]
            ),
            "joint_minus_public": summarize(
                [run["joint_minus_public"] for run in runs]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
