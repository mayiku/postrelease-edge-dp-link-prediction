"""Serialization helpers for the separated release/train/evaluate pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def edge_array(edges: set[tuple[int, int]] | list[tuple[int, int]]) -> np.ndarray:
    ordered = sorted(edges)
    if not ordered:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(ordered, dtype=np.int32)


def array_edges(values: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(left), int(right))
        for left, right in np.asarray(values, dtype=np.int32)
    }


def write_artifact(
    directory: Path,
    metadata: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(directory / "arrays.npz", **arrays)


def read_artifact(
    directory: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    with np.load(directory / "arrays.npz") as stored:
        arrays = {
            name: stored[name]
            for name in stored.files
        }
    return metadata, arrays
