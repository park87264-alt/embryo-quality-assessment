import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).parents[1] / "src" / "fair_event_alignment_ablation.py"
SPEC = importlib.util.spec_from_file_location("fair_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FairAblationTests(unittest.TestCase):
    def test_all_variants_keep_71_curve_dimensions(self):
        base = np.ones((2, 4, 35), dtype=np.float32)
        labels = np.asarray([[0, 11, 12, 15], [12, 12, 12, 12]], dtype=np.int64)
        masks = np.ones_like(labels, dtype=np.float32)
        soft = np.full_like(masks, 0.25)
        for variant in MODULE.VARIANTS:
            curve, scalar = MODULE.variant_inputs(base, labels, masks, soft, variant)
            self.assertEqual(curve.shape, (2, 4, 71))
            self.assertEqual(scalar.shape, (2, 4))

    def test_mask_only_zeroes_event_dependent_early_features(self):
        base = np.ones((1, 2, 35), dtype=np.float32)
        labels = np.asarray([[0, 12]], dtype=np.int64)
        masks = np.ones_like(labels, dtype=np.float32)
        curve, scalar = MODULE.variant_inputs(base, labels, masks, np.zeros_like(masks), "mask_only")
        self.assertTrue(np.all(curve[0, 0, MODULE.EVENT_DEPENDENT_FEATURES] == 0))
        self.assertTrue(np.all(curve[0, 1, MODULE.EVENT_DEPENDENT_FEATURES] == 1))
        self.assertTrue(np.all(scalar == 0))

    def test_soft_gate_state_round_trip(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(40, 35)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int64)
        scaler, classifier = MODULE.fit_logistic_gate(x, y, 42)
        expected = MODULE.predict_logistic_gate(x, scaler, classifier)
        actual = MODULE.gate_from_state(x, MODULE.gate_state(scaler, classifier))
        np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)

    def test_verified_curve_loader_loads_every_key(self):
        MODULE.seed_all(42)
        source_model = MODULE.GardnerModel(curve_hidden=16, dropout=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "curve.pt"
            torch.save({"encoder": source_model.curve.state_dict(), "variant": "raw_curve"}, checkpoint)
            MODULE.seed_all(43)
            target_model = MODULE.GardnerModel(curve_hidden=16, dropout=0.1)
            audit = MODULE.verified_load_curve(target_model, checkpoint)
            self.assertEqual(audit["loaded_key_count"], audit["target_key_count"])
            self.assertGreater(audit["max_abs_parameter_change"], 0)

    def test_verified_curve_loader_rejects_empty_state(self):
        model = MODULE.GardnerModel(curve_hidden=16, dropout=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "empty.pt"
            torch.save({"model": {"head.weight": torch.zeros(1)}}, checkpoint)
            with self.assertRaises(RuntimeError):
                MODULE.verified_load_curve(model, checkpoint)


if __name__ == "__main__":
    unittest.main()
