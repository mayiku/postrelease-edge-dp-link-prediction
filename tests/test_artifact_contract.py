from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.artifacts import read_artifact, write_artifact


ROOT = Path(__file__).resolve().parents[1]


class ArtifactContractTests(unittest.TestCase):
    def test_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = {"dataset": "toy", "epsilon": 3.5}
            arrays = {"x": np.asarray([1, 2, 3], dtype=np.int32)}
            write_artifact(path, metadata, arrays)
            loaded_metadata, loaded_arrays = read_artifact(path)
            self.assertEqual(loaded_metadata, metadata)
            np.testing.assert_array_equal(
                loaded_arrays["x"],
                arrays["x"],
            )

    def test_training_cli_has_no_raw_data_argument(self) -> None:
        source = (ROOT / "scripts" / "train_postrelease.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("--data", source)
        self.assertNotIn("load_graph_data", source)


if __name__ == "__main__":
    unittest.main()
