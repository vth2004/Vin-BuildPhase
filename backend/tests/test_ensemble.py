from __future__ import annotations

import math
import unittest
from app.services.ensemble import (
    compute_reference_point,
    compute_disagreement,
    compute_oks_suspicion,
    compute_reliability,
    compute_ranking_score,
    compute_rho,
    compute_bbox_iou,
    filter_consensus_models,
)


class TestEnsemble(unittest.TestCase):
    def test_hand_calculated_p_star(self):
        # Model 1: p1 = (100, 200), conf = 0.9, w1 = 2000 -> w1*c1 = 1800
        # Model 2: p2 = (110, 210), conf = 0.8, w2 = 3000 -> w2*c2 = 2400
        obs = [
            (100.0, 200.0, 0.9, 2000.0),
            (110.0, 210.0, 0.8, 3000.0),
        ]
        px, py = compute_reference_point(obs)
        # Expected: (1800*100 + 2400*110) / 4200 = 444000 / 4200 = 105.7142857
        # Expected: (1800*200 + 2400*210) / 4200 = 864000 / 4200 = 205.7142857
        self.assertAlmostEqual(px, 105.7142857, places=4)
        self.assertAlmostEqual(py, 205.7142857, places=4)

    def test_hand_calculated_delta(self):
        obs = [
            (100.0, 200.0, 0.9, 2000.0),
            (110.0, 210.0, 0.8, 3000.0),
        ]
        scale = 400.0
        p_star = compute_reference_point(obs)
        delta = compute_disagreement(obs, p_star, scale)

        # Disp = (1800 * (5.7143^2 + 5.7143^2) + 2400 * (4.2857^2 + 4.2857^2)) / 4200 = 48.9796
        # sqrt(Disp) = 6.9985
        # delta = 6.9985 / 400 = 0.017496
        self.assertAlmostEqual(delta, 0.017496, places=5)

    def test_hand_calculated_oks_and_reliability(self):
        scale = 400.0
        delta = 0.017496
        sigma_i = 0.025  # r_eye
        p_star = (105.7142857, 205.7142857)
        human_pt = (115.0, 215.0)

        dist = math.hypot(human_pt[0] - p_star[0], human_pt[1] - p_star[1])
        e = compute_oks_suspicion(dist, scale, delta, sigma_i, lam=0.5)
        self.assertTrue(0.18 <= e <= 0.19)

        c_bar = 0.84
        r = compute_reliability(c_bar, delta, tau=0.08, rho=1.0)
        self.assertTrue(0.81 <= r <= 0.83)

        score = compute_ranking_score(e, r, alpha=0.5)
        self.assertTrue(0.16 <= score <= 0.17)

    def test_k1_regression_equivalence(self):
        # Khi K=1: chỉ có 1 model
        obs = [(100.0, 200.0, 0.95, 2000.0)]
        scale = 350.0
        sigma_i = 0.079  # shoulder
        p_star = compute_reference_point(obs)
        self.assertEqual(p_star, (100.0, 200.0))

        delta = compute_disagreement(obs, p_star, scale)
        self.assertEqual(delta, 0.0)

        # OKS cũ trong scoring.py:
        dist = 25.0
        dist_norm = dist / scale
        k_i = 2.0 * sigma_i
        old_oks_diff = 1.0 - math.exp(- (dist_norm ** 2) / (2.0 * (k_i ** 2)))

        # OKS mới với delta=0:
        new_e = compute_oks_suspicion(dist, scale, delta, sigma_i, lam=0.5)
        self.assertAlmostEqual(old_oks_diff, new_e, places=7)

        # Reliability với delta=0, rho=1:
        r = compute_reliability(0.95, delta=0.0, tau=0.08, rho=1.0)
        self.assertAlmostEqual(r, 0.95, places=7)

    def test_consensus_iou_filter(self):
        # Box 1 và Box 2 trùng khớp: IoU ~ 1.0
        b1 = (10.0, 10.0, 100.0, 200.0)
        b2 = (12.0, 12.0, 102.0, 202.0)
        iou = compute_bbox_iou(b1, b2)
        self.assertTrue(iou > 0.80)

        matches = {
            "yolo26s-pose": {"bbox": b1, "score": 0.9},
            "rtmpose-m": {"bbox": b2, "score": 0.85},
        }
        valid, k_eff = filter_consensus_models(matches, primary_model="yolo26s-pose", iou_threshold=0.30)
        self.assertEqual(k_eff, 2)
        self.assertEqual(compute_rho(k_eff, K_target=2), 1.0)

        # Box 3 nằm ở góc khác hoàn toàn: IoU = 0.0 (bắt nhầm người khác)
        b3 = (500.0, 500.0, 600.0, 700.0)
        matches_discordant = {
            "yolo26s-pose": {"bbox": b1, "score": 0.9},
            "rtmpose-m": {"bbox": b3, "score": 0.85},
        }
        valid_disc, k_eff_disc = filter_consensus_models(matches_discordant, primary_model="yolo26s-pose", iou_threshold=0.30)
        self.assertEqual(k_eff_disc, 1)
        self.assertIn("yolo26s-pose", valid_disc)
        self.assertNotIn("rtmpose-m", valid_disc)
        self.assertEqual(compute_rho(k_eff_disc, K_target=2), 0.5)


if __name__ == "__main__":
    unittest.main()
