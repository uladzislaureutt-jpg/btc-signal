from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


DEFAULT_CONFIG = Path.cwd() / "config" / "strategy.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv()
    selected = Path(path or os.getenv("BTC_CONFIG", DEFAULT_CONFIG))
    with selected.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid strategy config: {selected}")
    return config
