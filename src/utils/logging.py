import logging
from pathlib import Path


def setup_logging(log_path: str | Path | None = None) -> None:
    """
    Simple and reliable logging.
    """
    handlers = [logging.StreamHandler()]

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
