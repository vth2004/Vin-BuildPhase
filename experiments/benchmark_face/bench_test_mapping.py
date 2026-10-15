"""Small regression suite for benchmark-only VF50 mappings."""

import unittest
from pathlib import Path

import numpy as np

from experiments.benchmark_face.bench_mapping_core import load_mapping, map_points


CONFIG_DIR = Path(__file__).with_name("configs")


class MappingTests(unittest.TestCase):
    def test_wflw_direct_mapping_and_anchors(self):
        config = load_mapping(CONFIG_DIR / "wflw98_to_vf50.yaml")
        source = np.column_stack((np.arange(98), np.arange(98) * 10.0))
        mapped = map_points(source, config)
        for vf_id, source_id in config["anchors"].items():
            np.testing.assert_allclose(mapped[int(vf_id)], source[int(source_id), :2])
        np.testing.assert_allclose(mapped[14:22], source[60:68, :2])

    def test_ibug68_anchors_are_not_interpolated(self):
        config = load_mapping(CONFIG_DIR / "ibug68_to_vf50.yaml")
        theta = np.linspace(0.0, 2.0 * np.pi, 68, endpoint=False)
        source = np.column_stack((100 + 40 * np.cos(theta), 80 + 20 * np.sin(theta)))
        mapped = map_points(source, config)
        for vf_id, source_id in config["anchors"].items():
            np.testing.assert_allclose(mapped[int(vf_id)], source[int(source_id), :2], atol=1e-5)

    def test_all_vf_ids_are_finite(self):
        for name, count in (("wflw98_to_vf50.yaml", 98), ("ibug68_to_vf50.yaml", 68)):
            config = load_mapping(CONFIG_DIR / name)
            source = np.column_stack((np.arange(count), np.arange(count)))
            self.assertTrue(np.isfinite(map_points(source, config)).all())


if __name__ == "__main__":
    unittest.main()
