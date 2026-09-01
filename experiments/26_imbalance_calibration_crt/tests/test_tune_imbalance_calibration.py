from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "src" / "tune_imbalance_calibration.py"
SPEC = importlib.util.spec_from_file_location("tune_imbalance_calibration", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ImbalanceCalibrationTests(unittest.TestCase):
    def test_tau_zero_is_ordinary_cross_entropy(self):
        logits = torch.tensor([[1.0, 0.2, -0.5], [0.0, 0.5, 1.2]])
        targets = torch.tensor([0, 2])
        counts = torch.tensor([100.0, 20.0, 5.0])
        actual = MODULE.balanced_softmax_loss(logits, targets, counts, 0.0)
        expected = torch.nn.functional.cross_entropy(logits, targets)
        torch.testing.assert_close(actual, expected)

    def test_sampling_strength_upweights_tail(self):
        targets = torch.tensor([0, 0, 0, 0, 1])
        weights = MODULE.sampling_weights(targets, rho=1.0, n_classes=2)
        self.assertGreater(weights[-1].item(), weights[0].item())

    def test_selection_respects_accuracy_constraint(self):
        candidates = [
            {"tau": 0.0, "metrics": {"accuracy": 0.50, "macro_f1_all_16": 0.30, "balanced_accuracy": 0.32}},
            {"tau": 0.5, "metrics": {"accuracy": 0.49, "macro_f1_all_16": 0.34, "balanced_accuracy": 0.36}},
            {"tau": 1.0, "metrics": {"accuracy": 0.45, "macro_f1_all_16": 0.40, "balanced_accuracy": 0.42}},
        ]
        selected = MODULE.choose_candidate(candidates, 0.50, 0.02, "tau")
        self.assertEqual(selected["tau"], 0.5)


if __name__ == "__main__":
    unittest.main()
