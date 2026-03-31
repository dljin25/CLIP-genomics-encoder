from __future__ import annotations

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    yaml = YAML(typ="safe")
    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected top-level mapping in config file: {resolved_path}")
    return config


def resolve_project_path(path_value: str | Path, *, project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else PROJECT_ROOT
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path
