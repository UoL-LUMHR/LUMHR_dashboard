"""Smoke checks for public dashboard routes and request validation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app
from core.access_index import WEIGHT_KEYS as ACCESS_WEIGHT_KEYS, normalize_weights as normalize_access_weights
from core.need_index import WEIGHT_KEYS as NEED_WEIGHT_KEYS, normalize_weights as normalize_need_weights
from core.rural_risk_index import RURAL_RISK_WEIGHT_KEYS, normalize_rural_risk_weights


class DashboardRoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_main_pages_render(self):
        for path in (
            "/",
            "/need_index",
            "/samhi",
            "/access_index",
            "/access_gap_index",
            "/rural_risk_index",
            "/samhi_forecast",
            "/public_samhi3",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_nonfinite_weights_are_rejected(self):
        for path in (
            "/api/need_scores?dep=nan",
            "/api/access_scores?mh002=inf",
            "/api/access_gap_scores?dep=-inf",
            "/api/rural_risk_scores?w_rural=nan",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 400)
                self.assertIn("Invalid float", response.get_json()["error"])

    def test_need_scores_have_finite_normalized_values(self):
        response = self.client.get("/api/need_scores")
        self.assertEqual(response.status_code, 200)
        scores = response.get_json()["layers"]["Need_Index"]
        self.assertGreater(len(scores), 400)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores.values()))

    def test_large_finite_weights_keep_scores(self):
        response = self.client.get("/api/need_scores?dep=1e308&smi=1e308")
        self.assertEqual(response.status_code, 200)
        scores = response.get_json()["layers"]["Need_Index"]
        self.assertGreater(len(scores), 400)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores.values()))

    def test_all_weight_groups_handle_large_values(self):
        for normalizer, keys in (
            (normalize_need_weights, NEED_WEIGHT_KEYS),
            (normalize_access_weights, ACCESS_WEIGHT_KEYS),
            (normalize_rural_risk_weights, RURAL_RISK_WEIGHT_KEYS),
        ):
            with self.subTest(normalizer=normalizer.__module__):
                weights = {key: 1e308 if index < 2 else 0.0 for index, key in enumerate(keys)}
                result = normalizer(weights)
                self.assertAlmostEqual(sum(result.values()), 1.0)
                self.assertAlmostEqual(result[keys[0]], 0.5)
                self.assertAlmostEqual(result[keys[1]], 0.5)


if __name__ == "__main__":
    unittest.main()
