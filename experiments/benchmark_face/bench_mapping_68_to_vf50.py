"""CLI wrapper shared by 300W, COFW-68, MERL-RAV and 300VW."""

from pathlib import Path

from .bench_mapping_core import load_mapping, map_points

CONFIG_PATH = Path(__file__).with_name("configs") / "ibug68_to_vf50.yaml"


def map_ibug68_to_vf50(points):
    return map_points(points, load_mapping(CONFIG_PATH))
