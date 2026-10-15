"""CLI wrapper for the approved WFLW-98 to VF50 mapping."""

from pathlib import Path

from .bench_mapping_core import load_mapping, map_points

CONFIG_PATH = Path(__file__).with_name("configs") / "wflw98_to_vf50.yaml"


def map_wflw98_to_vf50(points):
    return map_points(points, load_mapping(CONFIG_PATH))
