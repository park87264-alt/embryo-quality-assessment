from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "src" / "render_roi_contribution.py"
SPEC = importlib.util.spec_from_file_location("roi_contribution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ROIContributionTests(unittest.TestCase):
    def test_region_mass_follows_weights_without_painting_center(self):
        global_mask = np.ones((4, 4), dtype=bool)
        icm = np.zeros((4, 4), dtype=bool)
        icm[:2, :2] = True
        te = np.zeros((4, 4), dtype=bool)
        te[2:, 2:] = True
        zp = np.zeros((4, 4), dtype=bool)
        zp[:, 2] = True
        density = MODULE.compute_density(
            {"global": global_mask, "icm": icm, "te": te, "zp": zp},
            {"global": 0.1, "icm": 0.8, "te": 0.1, "zp": 0.0},
        )
        self.assertAlmostEqual(density.sum(), 1.0)
        self.assertGreater(density[0, 0], density[3, 3])

    def test_invalid_weights_rejected(self):
        masks = {region: np.ones((2, 2), dtype=bool) for region in MODULE.REGIONS}
        with self.assertRaisesRegex(ValueError, "sum to one"):
            MODULE.compute_density(masks, dict.fromkeys(MODULE.REGIONS, 0.1))


if __name__ == "__main__":
    unittest.main()
