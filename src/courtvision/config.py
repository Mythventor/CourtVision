"""Load project configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file (defaults to configs/default.yaml)."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)
