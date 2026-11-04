#!/usr/bin/env python
"""Phase 1: create the only private-data-derived output for P1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import edge_array, write_artifact  # noqa: E402
from src.postrelease_lp import (  # noqa: E402
    audit_workload,
    feature_candidates,
    graph_edges,
    load_graph_data,
    pair_features_for,
    public_projected_features,
    randomized_response_graph,
    jaccard_denoise,
    split_bucket,
    split_name,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-salt", type=int, required=True)
    parser.add_argument("--draw", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=13000)
    parser.add_argument("--epsilon", type=float, default=3.5)
    parser.add_argument("--target-candidates", type=int, default=20)
    parser.add_argument("--release-candidates", type=int, default=400)
    parser.add_argument("--target-degree", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    graph, raw_features = load_graph_data(args.data)
    private_edges = graph_edges(graph)
    release_candidates = feature_candidates(
        raw_features,
        args.release_candidates,
    )
    target_pairs = feature_candidates(
        raw_features,
        args.target_candidates,
    )
    workload = {
        name: [
            pair
            for pair in target_pairs
            if split_name(split_bucket(pair, args.split_salt)) == name
        ]
        for name in ("train", "valid", "test")
    }
    workload_audit = audit_workload(workload, release_candidates)
    if any(
        workload_audit[key] != 0
        for key in (
            "train_test_intersection",
            "train_valid_intersection",
            "valid_test_intersection",
        )
    ):
        raise AssertionError("target workload splits overlap")

    seed = args.base_seed + 101 * args.draw
    released = randomized_response_graph(
        release_candidates,
        private_edges,
        args.epsilon,
        seed,
    )
    denoised = jaccard_denoise(
        release_candidates,
        released,
        graph.shape[0],
        args.target_degree,
        args.epsilon,
        seed + 17,
    )
    projected = public_projected_features(raw_features)
    norms = np.sqrt(
        np.asarray(
            raw_features.multiply(raw_features).sum(axis=1)
        ).ravel()
    ).astype(np.float32)

    arrays: dict[str, np.ndarray] = {
        "release_candidates": edge_array(release_candidates),
        "released_edges": edge_array(released),
        "denoised_edges": edge_array(denoised),
    }
    for name, pairs in workload.items():
        arrays[f"workload_{name}"] = edge_array(pairs)
        arrays[f"public_{name}"] = pair_features_for(
            raw_features,
            projected,
            norms,
            pairs,
        )

    p = math.exp(args.epsilon) / (1.0 + math.exp(args.epsilon))
    q = 1.0 / (1.0 + math.exp(args.epsilon))
    protocol_hash = hashlib.sha256(
        json.dumps(workload_audit, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata = {
        "artifact_version": "p1-release-v1",
        "dataset": args.dataset,
        "split_salt": args.split_salt,
        "draw": args.draw,
        "seed": seed,
        "epsilon": args.epsilon,
        "delta": 0.0,
        "p": p,
        "q": q,
        "target_candidates": args.target_candidates,
        "release_candidates": args.release_candidates,
        "target_degree": args.target_degree,
        "node_count": int(graph.shape[0]),
        "public_feature_dimension": int(arrays["public_train"].shape[1]),
        "release_candidate_count": len(release_candidates),
        "released_edge_count": len(released),
        "denoised_edge_count": len(denoised),
        "workload_audit": workload_audit,
        "workload_audit_hash": protocol_hash,
        "postprocessing_epsilon": 0.0,
    }
    write_artifact(args.output_dir, metadata, arrays)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
