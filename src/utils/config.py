import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """
    Load YAML config and expand environment variables
    """
    load_dotenv()  # loads .env

    config_path = Path(config_path)
    raw_text = config_path.read_text(encoding="utf-8")

    raw_text = os.path.expandvars(raw_text)
    cfg = yaml.safe_load(raw_text)

    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """
    create all output folders.
    """
    for key in [
        "interim_partitioned_dir",
        "processed_dir",
        "outputs_dir",
        "reports_dir",
    ]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
