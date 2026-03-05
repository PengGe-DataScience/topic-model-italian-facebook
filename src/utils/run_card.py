import json
import platform
import subprocess
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict


def get_git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def write_run_card(
    path: str | Path, cfg: Dict[str, Any], extra: Dict[str, Any] | None = None
) -> None:
    """
    Writes a small metadata JSON to make the project reproducible.
    """
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit_hash(),
        "config": cfg,
    }
    if extra:
        payload.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
