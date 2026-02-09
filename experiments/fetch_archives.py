from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from experiments.common import configure_runtime_dirs, ensure_directory
except ModuleNotFoundError:
    from common import configure_runtime_dirs, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve/download official COCO archive datasets by keyword."
    )
    parser.add_argument("--algorithm", nargs="+", default=["CMA-ES", "BFGS"])
    parser.add_argument("--suite", default="bbob")
    parser.add_argument("--output", default="results/archive_paths.txt")
    return parser.parse_args()


def match_archive_entry(archive: Any, keyword: str) -> Any:
    matches_object = archive.get_all(keyword)
    matches = list(matches_object)
    if not matches:
        raise ValueError(f"No archive entries matched keyword: {keyword}")

    if len(matches) > 1:
        listing = getattr(matches_object, "as_string", None)
        listing_text = listing() if callable(listing) else str(listing)
        print(f"Multiple matches for '{keyword}'. Using the first one.")
        print(listing_text)

    return matches[0]


def resolve_entry_path(entry: Any) -> str:
    if hasattr(entry, "get"):
        return str(entry.get())
    return str(entry)


def main() -> None:
    args = parse_args()
    configure_runtime_dirs()

    try:
        from cocopp import archives  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("cocopp is required. Install with: pip install cocopp") from error

    archive = archives.get(args.suite)

    output_path = Path(args.output)
    ensure_directory(str(output_path.parent))

    lines: list[str] = []
    for algorithm in args.algorithm:
        entry = match_archive_entry(archive, algorithm)
        resolved = resolve_entry_path(entry)
        lines.append(f"{algorithm}\t{resolved}")
        print(f"{algorithm}: {resolved}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
