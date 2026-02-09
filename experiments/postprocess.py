from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.common import configure_runtime_dirs, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cocopp postprocessing and plotting.")
    parser.add_argument("--input", nargs="+", required=True, help="Local result folders.")
    parser.add_argument("--archive", nargs="*", default=[], help="Archive keywords, e.g. CMA-ES BFGS")
    parser.add_argument("--suite", default="bbob")
    parser.add_argument("--output", default="ppdata")
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


def resolve_inputs(local_inputs: list[str], archive_keywords: list[str], suite: str) -> list[str]:
    resolved = []
    for path in local_inputs:
        if not Path(path).exists():
            raise FileNotFoundError(f"Local result path does not exist: {path}")
        resolved.append(path)

    if not archive_keywords:
        return resolved

    try:
        from cocopp import archives  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("cocopp is required. Install with: pip install cocopp") from error

    archive = archives.get(suite)
    for keyword in archive_keywords:
        entry = match_archive_entry(archive, keyword)
        resolved.append(resolve_entry_path(entry))

    return resolved


def main() -> None:
    args = parse_args()
    configure_runtime_dirs()
    inputs = resolve_inputs(args.input, args.archive, args.suite)
    ensure_directory(args.output)

    command = [sys.executable, "-m", "cocopp", "-o", args.output, *inputs]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
