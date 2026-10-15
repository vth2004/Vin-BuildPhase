"""Generate auditable Markdown tables from mapping YAML files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bench_mapping_core import load_mapping, mapping_rows


def render(config_path: Path) -> str:
    config = load_mapping(config_path)
    lines = [
        f"## {config['dataset_family']} → VF50",
        "",
        "| VF ID | Skeleton | Method | Source ID/polyline | Fraction | Anchor |",
        "|---:|---|---|---|---:|:---:|",
    ]
    for row in mapping_rows(config):
        fraction = "—" if row["fraction"] is None else f"{row['fraction']:.2f}"
        lines.append(
            f"| {row['vf_id']} | {row['skeleton']} | {row['method']} | "
            f"`{row['source']}` | {fraction} | {'yes' if row['anchor'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()
    sections = ["# Public Face Dataset → VF50 Mapping", ""]
    sections.extend(render(path) for path in args.configs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
