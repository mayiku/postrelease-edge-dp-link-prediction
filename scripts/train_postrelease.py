#!/usr/bin/env python
"""Phase 2: train only from a released artifact and public features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import array_edges, read_artifact  # noqa: E402
from src.postrelease_core import (  # noqa: E402
    posterior_and_weights,
    rank01,
    topology_features,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--topology-dimension", type=int, default=8)
    parser.add_argument("--incidence-cap", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata, arrays = read_artifact(args.artifact_dir)
    workload = {
        name: [
            (int(left), int(right))
            for left, right in arrays[f"workload_{name}"]
        ]
        for name in ("train", "valid", "test")
    }
    public = {
        name: arrays[f"public_{name}"]
        for name in ("train", "valid", "test")
    }
    released = array_edges(arrays["released_edges"])
    denoised = array_edges(arrays["denoised_edges"])
    topology = {
        name: topology_features(
            workload[name],
            sorted(denoised),
            int(metadata["node_count"]),
            args.incidence_cap,
        )[:, : args.topology_dimension]
        for name in ("train", "valid", "test")
    }
    train_bits = np.asarray(
        [pair in released for pair in workload["train"]],
        dtype=np.float32,
    )
    valid_bits = np.asarray(
        [pair in released for pair in workload["valid"]],
        dtype=np.float32,
    )
    train_posterior, train_weights = posterior_and_weights(
        train_bits,
        float(metadata["epsilon"]),
    )
    valid_posterior, valid_weights = posterior_and_weights(
        valid_bits,
        float(metadata["epsilon"]),
    )
    joint = {
        name: np.concatenate(
            [public[name], topology[name]],
            axis=1,
        )
        for name in ("train", "valid", "test")
    }
    seed = int(metadata["seed"]) + 201
    public_scores = train_model(
        public["train"],
        public["test"],
        train_posterior,
        train_weights,
        seed,
        "f3",
        validation_features=public["valid"],
        validation_target=valid_posterior,
        validation_weights=valid_weights,
    )
    joint_scores = train_model(
        joint["train"],
        joint["test"],
        train_posterior,
        train_weights,
        seed,
        "f3",
        validation_features=joint["valid"],
        validation_target=valid_posterior,
        validation_weights=valid_weights,
    )
    draw = int(metadata["draw"])
    train_permutation = np.random.default_rng(
        21000 + 101 * draw
    ).permutation(len(workload["train"]))
    valid_permutation = np.random.default_rng(
        20000 + 101 * draw
    ).permutation(len(workload["valid"]))
    test_permutation = np.random.default_rng(
        22000 + 101 * draw
    ).permutation(len(workload["test"]))
    placebo = {
        "train": np.concatenate(
            [public["train"], topology["train"][train_permutation]],
            axis=1,
        ),
        "valid": np.concatenate(
            [public["valid"], topology["valid"][valid_permutation]],
            axis=1,
        ),
        "test": np.concatenate(
            [public["test"], topology["test"][test_permutation]],
            axis=1,
        ),
    }
    placebo_scores = train_model(
        placebo["train"],
        placebo["test"],
        train_posterior,
        train_weights,
        seed,
        "f3",
        validation_features=placebo["valid"],
        validation_target=valid_posterior,
        validation_weights=valid_weights,
    )
    signed_train = 2.0 * (
        (train_bits - float(metadata["q"]))
        / (float(metadata["p"]) - float(metadata["q"]))
    ) - 1.0
    centroid = (
        signed_train[:, None] * topology["train"]
    ).mean(axis=0)
    fixed_fusion_scores = (
        0.75 * rank01(topology["test"] @ centroid)
        + 0.25 * rank01(public["test"][:, -3])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        public=public_scores,
        joint=joint_scores,
        placebo=placebo_scores,
        fixed_fusion=fixed_fusion_scores,
    )
    print(
        json.dumps(
            {
                "artifact_dir": str(args.artifact_dir),
                "output": str(args.output),
                "read_raw_graph": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
