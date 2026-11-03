from __future__ import annotations

import math
import unittest

import numpy as np

from src.postrelease_lp import (
    canonical,
    posterior_and_weights,
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


if __name__ == "__main__":
    unittest.main()
