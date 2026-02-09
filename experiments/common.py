from __future__ import annotations

import hashlib
import os
from pathlib import Path


def parse_dimensions(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one dimension must be provided.")

    dimensions = [int(value) for value in values]
    for dimension in dimensions:
        if dimension <= 0:
            raise ValueError("Dimensions must be positive integers.")
    return dimensions


def build_suite_filter(dimensions: list[int], instances: str) -> str:
    dimensions_text = ",".join(str(value) for value in dimensions)
    return f"dimensions: {dimensions_text} instance_indices: {instances}"


def seed_for_problem(base_seed: int, problem_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{problem_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def ensure_directory(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_runtime_dirs(base_path: str = ".runtime") -> Path:
    runtime_dir = ensure_directory(base_path)
    cache_root = ensure_directory(str(runtime_dir / "cache"))
    mpl_dir = ensure_directory(str(runtime_dir / "mplconfig"))

    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
    return runtime_dir
