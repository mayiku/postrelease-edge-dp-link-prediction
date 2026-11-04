#!/usr/bin/env python
"""Run release, public post-processing, and trusted evaluation separately."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-salt", type=int, required=True)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()

    release_dir = args.workdir / "release"
    train_dir = args.workdir / "train"
    evaluation_dir = args.workdir / "evaluation"
    for directory in (release_dir, train_dir, evaluation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(SCRIPTS / "release_graph.py"),
            "--data",
            str(args.data.resolve()),
            "--dataset",
            args.dataset,
            "--split-salt",
            str(args.split_salt),
            "--draw",
            str(args.draw),
            "--output-dir",
            str(release_dir.resolve()),
        ]
    )

    # The training phase has no raw-data argument and runs from a directory
    # containing only the released artifact.
    run(
        [
            sys.executable,
            str(SCRIPTS / "train_postrelease.py"),
            "--artifact-dir",
            ".",
            "--output",
            str((train_dir / "scores.npz").resolve()),
        ],
        cwd=release_dir,
    )

    run(
        [
            sys.executable,
            str(SCRIPTS / "evaluate.py"),
            "--data",
            str(args.data.resolve()),
            "--artifact-dir",
            str(release_dir.resolve()),
            "--scores",
            str((train_dir / "scores.npz").resolve()),
            "--output",
            str((evaluation_dir / "evaluation.json").resolve()),
        ]
    )

    print(
        "separated pipeline complete: "
        f"{evaluation_dir / 'evaluation.json'}"
    )


if __name__ == "__main__":
    main()
