"""Logging helpers for consistent console output."""
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional


def setup_logging(verbose: bool = False, log_path: Optional[Path] = None) -> None:
    """Configure root logging for the app."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
        except OSError as exc:
            fallback = Path(tempfile.gettempdir()) / "paper2ppt.run.log"
            try:
                handlers.append(logging.FileHandler(fallback, mode="w", encoding="utf-8"))
                print(
                    f"[WARN] Failed to open log file at {log_path} ({exc}). "
                    f"Logging to {fallback} instead.",
                    file=sys.stderr,
                )
            except OSError:
                print(
                    f"[WARN] Failed to open log file at {log_path} ({exc}). "
                    "Continuing without file logging.",
                    file=sys.stderr,
                )
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        handlers=handlers,
    )
