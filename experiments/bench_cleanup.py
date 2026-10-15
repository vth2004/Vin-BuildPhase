"""Safely remove only DELETE entries recorded in BENCH_MANIFEST.md.

The default mode is dry-run. Pass --yes to perform deletion after reviewing the
printed targets. Paths outside the repository and symlinks are rejected.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
from pathlib import Path


ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|.*\|\s*DELETE\s*\|\s*$")


def parse_delete_entries(manifest: Path) -> list[str]:
    entries: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line)
        if match:
            entries.append(match.group(1).rstrip("/\\"))
    return entries


def resolve_safe(repo: Path, raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError(f"Unsafe manifest path: {raw}")
    target = (repo / candidate).resolve(strict=False)
    try:
        target.relative_to(repo)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository: {raw}") from exc
    if target == repo:
        raise ValueError("Repository root cannot be deleted")
    return target


def collapse_nested(paths: list[Path]) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(set(paths), key=lambda item: (len(item.parts), str(item))):
        if not any(path == parent or parent in path.parents for parent in selected):
            selected.append(path)
    return selected


def path_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_symlink():
        raise ValueError(f"Refusing symlink target: {path}")
    if path.is_file():
        return path.stat().st_size, 1
    total = 0
    count = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            for item in iterator:
                if item.is_symlink():
                    # Count the link itself, never follow it outside the tree.
                    total += item.stat(follow_symlinks=False).st_size
                    count += 1
                elif item.is_dir(follow_symlinks=False):
                    pending.append(Path(item.path))
                elif item.is_file(follow_symlinks=False):
                    total += item.stat(follow_symlinks=False).st_size
                    count += 1
    return total, count


def remove_readonly(function, path: str, _exc_info) -> None:
    """Retry deletion of read-only files commonly found in Windows Git clones."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="experiments/BENCH_MANIFEST.md")
    parser.add_argument("--yes", action="store_true", help="perform deletion")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    manifest = resolve_safe(repo, args.manifest)
    raw_entries = parse_delete_entries(manifest)
    targets = collapse_nested([resolve_safe(repo, item) for item in raw_entries])
    mode = "DELETE" if args.yes else "DRY-RUN"
    print(f"Mode: {mode}; manifest DELETE rows: {len(raw_entries)}; targets: {len(targets)}")

    total_bytes = 0
    total_files = 0
    for target in targets:
        size, files = path_size(target)
        total_bytes += size
        total_files += files
        state = "exists" if target.exists() else "missing"
        print(f"[{state}] {target} | {files} files | {size} bytes")

    print(f"TOTAL | {total_files} files | {total_bytes} bytes")
    if not args.yes:
        print("No files deleted. Re-run with --yes only after reviewing this list.")
        return 0

    for target in targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target, onerror=remove_readonly)
        else:
            target.unlink()
        print(f"Deleted: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
