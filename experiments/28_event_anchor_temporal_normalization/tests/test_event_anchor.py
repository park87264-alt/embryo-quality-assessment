from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    SCRIPT = Path(__file__).parents[1] / "src" / "event_anchor_temporal_normalization.py"
    SPEC = importlib.util.spec_from_file_location("event_anchor", SCRIPT)
    MODULE = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(torch is not None, "PyTorch is required for the event-anchor experiment")
class EventAnchorTests(unittest.TestCase):
    def test_anchor_order_is_enforced(self):
        values = MODULE.enforce_anchor_order(np.asarray([[0.8, 0.3, 0.4]], dtype=np.float32))
        self.assertTrue(np.all(np.diff(values[0]) > 0))

    def test_piecewise_warp_maps_common_anchors(self):
        canonical = np.asarray([0.0, 0.4, 0.7, 0.9, 1.0], dtype=np.float32)
        mapped = MODULE.inverse_piecewise_warp(canonical, canonical[1:4], 10.0, np.asarray([40.0, 70.0, 90.0]), 100.0)
        np.testing.assert_allclose(mapped, [10.0, 40.0, 70.0, 90.0, 100.0])

    def test_nearest_labels(self):
        labels = MODULE.nearest_labels(np.asarray([0.0, 10.0, 20.0]), np.asarray([1, 2, 3]), np.asarray([1.0, 9.0, 18.0]))
        np.testing.assert_array_equal(labels, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
