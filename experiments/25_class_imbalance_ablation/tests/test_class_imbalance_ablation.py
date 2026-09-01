from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).parents[1] / "src" / "class_imbalance_ablation.py"
SPEC = importlib.util.spec_from_file_location("class_imbalance_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ClassImbalanceAblationTests(unittest.TestCase):
    def test_inverse_frequency_upweights_tail(self):
        weights = MODULE.inverse_frequency_weights(np.asarray([100, 20, 5, 0]))
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[1], weights[0])
        self.assertEqual(weights[3], 0)

    def test_effective_number_is_less_extreme_than_inverse_frequency(self):
        counts = np.asarray([1000, 100, 5], dtype=float)
        inverse = MODULE.inverse_frequency_weights(counts)
        effective = MODULE.effective_number_weights(counts, beta=0.99)
        self.assertLess(effective.max() / effective.min(), inverse.max() / inverse.min())

    def test_all_losses_are_finite_and_differentiable(self):
        counts = torch.tensor([100.0, 20.0, 5.0])
        inverse = torch.from_numpy(MODULE.inverse_frequency_weights(counts.numpy()))
        effective = torch.from_numpy(MODULE.effective_number_weights(counts.numpy()))
        targets = torch.tensor([0, 1, 2, 0])
        for variant in MODULE.LOSS_VARIANTS:
            logits = torch.randn(4, 3, requires_grad=True)
            loss = MODULE.imbalance_loss(
                logits, targets, variant, counts, inverse, effective, focal_gamma=2.0
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_support_groups_cover_each_present_class_once(self):
        groups = MODULE.support_groups(np.asarray([100, 50, 10, 5, 0]))
        flattened = groups["tail"] + groups["medium"] + groups["head"]
        self.assertEqual(sorted(flattened), [0, 1, 2, 3])
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == "__main__":
    unittest.main()
