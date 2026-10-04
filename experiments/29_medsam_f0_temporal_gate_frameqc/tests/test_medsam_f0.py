from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    SCRIPT = Path(__file__).parents[1] / "src" / "medsam_f0_temporal_gate.py"
    SPEC = importlib.util.spec_from_file_location("medsam_f0", SCRIPT)
    MODULE = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(torch is not None, "PyTorch is required for the MedSAM experiment")
class MedSAMF0Tests(unittest.TestCase):
    def test_all_variants_have_stage_logits(self):
        for variant in MODULE.VARIANTS:
            with self.subTest(variant=variant):
                model = MODULE.MedSAMStageModel(variant, 256, 72, 16, 0.0).eval()
                with torch.no_grad():
                    logits = model(torch.randn(2, 4, 256), torch.randn(2, 4, 72), torch.ones(2, 4))
                self.assertEqual(tuple(logits.shape), (2, 4, 16))

    def test_quality_heads_are_icm_and_te_only(self):
        backbone = MODULE.MedSAMStageModel("medsam_temporal_gate", 256, 72, 16, 0.0)
        model = MODULE.WeakGardnerModel(backbone, 16, 0.0).eval()
        with torch.no_grad():
            output = model(
                torch.randn(2, 4, 256), torch.randn(2, 4, 72),
                torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 4),
            )
        self.assertEqual(set(output), {"ICM", "TE"})
        self.assertTrue(all(tuple(logits.shape) == (2, 3) for logits in output.values()))

    def test_qc_requires_complete_valid_probabilities(self):
        ids = np.asarray(["a"])
        frames = np.asarray([[1, 2]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qc.csv"
            pd.DataFrame({
                "embryo_id": ["a", "a"], "frame": [1, 2],
                "qc_probability_invalid": [0.1, 1.1],
            }).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "finite values"):
                MODULE.load_qc_probabilities(path, ids, frames)
            pd.DataFrame({
                "embryo_id": ["a"], "frame": [1], "qc_probability_invalid": [0.1],
            }).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "do not cover"):
                MODULE.load_qc_probabilities(path, ids, frames)

    def test_quality_split_is_fixed_for_seed(self):
        indices = np.arange(100)
        labels = np.zeros((100, 2), dtype=np.int64)
        mask = np.ones((100, 2), dtype=np.float32)
        first = MODULE.quality_split(indices, labels, mask, 42)
        second = MODULE.quality_split(indices, labels, mask, 42)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(sorted(np.concatenate(first).tolist()), indices.tolist())


if __name__ == "__main__":
    unittest.main()
