from __future__ import annotations

import math
import unittest

import numpy as np

from src.postrelease_lp import (
    audit_workload,
    canonical,
    posterior_and_weights,
    privacy_metadata,
    split_bucket,
    topology_features,
)


class CoreTests(unittest.TestCase):
    def test_canonical_edge(self) -> None:
        self.assertEqual(canonical((4, 2)), (2, 4))

    def test_split_is_deterministic(self) -> None:
        self.assertEqual(
            split_bucket((2, 4), 0),
            split_bucket((4, 2), 0),
        )

    def test_posterior_moves_in_expected_direction(self) -> None:
        released = np.asarray([0, 0, 1, 1], dtype=np.float32)
        posterior, weights = posterior_and_weights(released, 3.5)
        self.assertLess(posterior[0], posterior[2])
        self.assertTrue(np.all(weights > 0))

    def test_topology_target_edge_is_excluded(self) -> None:
        features = topology_features(
            [(0, 1)],
            [(0, 1)],
            node_count=3,
            incidence_cap=8,
        )
        self.assertTrue(
            math.isclose(float(features[0, 0]), 0.0, abs_tol=1e-8)
        )
        self.assertTrue(
            math.isclose(float(features[0, 8]), 0.0, abs_tol=1e-8)
        )

    def test_main_pipeline_has_no_separate_label_release(self) -> None:
        metadata = privacy_metadata(3.5, "released")
        self.assertEqual(
            metadata["training_label_source"],
            "released_graph_train_bits",
        )
        self.assertFalse(metadata["separate_label_release"])
        self.assertEqual(metadata["postprocessing_epsilon"], 0.0)

    def test_true_label_diagnostic_has_no_privacy_claim(self) -> None:
        metadata = privacy_metadata(3.5, "true")
        self.assertEqual(metadata["privacy_claim"], "none")

    def test_workload_audit_reports_disjoint_splits(self) -> None:
        workload = {
            "train": [(0, 1)],
            "valid": [(0, 2)],
            "test": [(1, 2)],
        }
        metadata = audit_workload(
            workload,
            [(0, 1), (0, 2), (1, 2), (2, 3)],
        )
        self.assertEqual(metadata["train_test_intersection"], 0)
        self.assertEqual(metadata["train_valid_intersection"], 0)
        self.assertEqual(metadata["valid_test_intersection"], 0)
        self.assertFalse(metadata["test_target_bits_used_for_training"])


if __name__ == "__main__":
    unittest.main()
