"""Normalize EXIF orientation into pixels without overwriting source images."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def normalize(source: Path, destination: Path) -> list[Path]:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be a separate directory, not source or a child of source")

    written: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            normalized = ImageOps.exif_transpose(image)
            # Saving a new image without this field prevents browsers from rotating it again.
            exif = normalized.getexif()
            exif.pop(274, None)
            save_kwargs: dict[str, object] = {}
            if path.suffix.lower() in {".jpg", ".jpeg", ".webp"}:
                save_kwargs["exif"] = exif.tobytes()
            normalized.save(target, **save_kwargs)
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply EXIF orientation to pixels into a new directory.")
    parser.add_argument("source", type=Path, help="Directory containing source images")
    parser.add_argument("destination", type=Path, help="New output directory; source is never modified")
    args = parser.parse_args()
    written = normalize(args.source, args.destination)
    print(f"Normalized {len(written)} image(s) into {args.destination}")


if __name__ == "__main__":
    main()
