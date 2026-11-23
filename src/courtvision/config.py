"""Load project configuration from YAML."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/default.yaml, with the config at `path` (if any) merged over it."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if path:
        with open(path, encoding="utf-8") as f:
            config = merge(config, yaml.safe_load(f) or {})
    return config
