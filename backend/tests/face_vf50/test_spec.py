import unittest
from app.features.face_landmark_vf50.spec import (
    load_vf50_spec,
    get_anchor_points,
    get_scoring_tolerances,
    get_case_classification_spec,
    get_rules_spec,
    get_geometry_spec,
    get_matching_spec,
)


class TestVF50SpecLoader(unittest.TestCase):
    def test_spec_loads_successfully(self):
        spec = load_vf50_spec()
        self.assertEqual(spec["schema_id"], "vf_face_landmark50_v1")
        self.assertEqual(len(spec["skeletons"]), 7)

    def test_anchor_points(self):
        anchors = get_anchor_points()
        expected = {0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36}
        self.assertEqual(anchors, expected)

    def test_scoring_tolerances(self):
        tol = get_scoring_tolerances()
        self.assertEqual(tol["anchor_point_iod_ratio"], 0.03)
        self.assertEqual(tol["contour_point_iod_ratio"], 0.05)
        self.assertEqual(tol["nme_max_threshold"], 0.035)
        self.assertEqual(tol["severity_critical_multiplier"], 1.8)
        self.assertEqual(tol["suspicion_normalizer_multiplier"], 2.5)
        self.assertEqual(tol["occluded_suspicion_discount"], 0.6)

    def test_rules_spec_values(self):
        rules = get_rules_spec()
        # R08 is 1.55 in code
        self.assertEqual(rules["r08_nose_bridge"]["max_ratio"], 1.55)
        # R13 uses adaptive scale multiplier = 15.0
        self.assertEqual(rules["r13_temporal_jump"]["scale_factor_multiplier"], 15.0)
        self.assertEqual(rules["r13_temporal_jump"]["min_jump_px"], 5.0)

    def test_geometry_and_matching_spec(self):
        geom = get_geometry_spec()
        self.assertEqual(geom["iod"]["default_iod_px"], 96.0)
        matching = get_matching_spec()
        self.assertEqual(matching["cutoff_iod_multiplier"], 2.0)


if __name__ == "__main__":
    unittest.main()
