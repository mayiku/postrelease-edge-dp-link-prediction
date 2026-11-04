#!/usr/bin/env python
"""Phase 3: trusted evaluation against private test labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import read_artifact  # noqa: E402
from src.postrelease_lp import graph_edges, load_graph_data, metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata, arrays = read_artifact(args.artifact_dir)
    test_pairs = [
        (int(left), int(right))
        for left, right in arrays["workload_test"]
    ]
    graph, _ = load_graph_data(args.data)
    private_edges = graph_edges(graph)
    test_labels = np.asarray(
        [pair in private_edges for pair in test_pairs],
        dtype=np.float32,
    )
    with np.load(args.scores) as scores:
        result = {
            model: metrics(scores[model], test_labels)
            for model in scores.files
        }
    result["dataset"] = metadata["dataset"]
    result["split_salt"] = metadata["split_salt"]
    result["draw"] = metadata["draw"]
    result["evaluation_labels_used_for_training"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
