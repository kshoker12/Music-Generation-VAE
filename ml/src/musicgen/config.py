from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"


def load_config(_: str | None = None) -> ProjectConfig:
    # Stub: later load YAML/JSON and validate.
    return ProjectConfig()

