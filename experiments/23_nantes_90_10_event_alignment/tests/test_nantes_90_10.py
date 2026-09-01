from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "src" / "nantes_90_10_event_alignment.py"
SPEC = importlib.util.spec_from_file_location("nantes_90_10", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Nantes9010Tests(unittest.TestCase):
    def test_aggregate_uses_all_seeds(self):
        runs = []
        for variant in MODULE.VARIANTS:
            for seed, acc in [(42, 0.4), (43, 0.5), (44, 0.6)]:
                runs.append(
                    {
                        "variant": variant,
                        "accuracy": acc,
                        "balanced_accuracy": acc - 0.1,
                        "macro_f1": acc - 0.2,
                        "confusion_matrix": [[1, 0], [0, 1]],
                    }
                )
        result = MODULE.aggregate(runs)
        self.assertAlmostEqual(result["raw_curve"]["accuracy"]["mean"], 0.5)
        self.assertEqual(result["raw_curve"]["total"], 6)

    def test_ids_hash_is_order_sensitive(self):
        self.assertNotEqual(MODULE.ids_hash(["a", "b"]), MODULE.ids_hash(["b", "a"]))

    def test_write_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "value.json"
            MODULE.write_json(path, {"value": np.int64(3).item()})
            self.assertIn('"value": 3', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
