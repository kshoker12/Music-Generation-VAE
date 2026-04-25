from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_dir: Path
    artifacts_dir: Path


def get_paths(repo_root: str | Path) -> Paths:
    root = Path(repo_root)
    return Paths(
        repo_root=root,
        data_dir=root / "data",
        artifacts_dir=root / "artifacts",
    )

