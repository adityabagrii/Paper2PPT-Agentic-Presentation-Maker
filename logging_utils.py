"""Logging helpers for consistent console output."""
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional


def setup_logging(verbose: bool = False, log_path: Optional[Path] = None) -> None:
    """Function setup logging.
    
    Args:
        verbose (bool):
        log_path (Optional[Path]):
    
    Returns:
        None:
    """
    level = logging.DEBUG if verbose else logging.INFO
    # Prefer rich console logging if available for colored output
    try:
        from rich.logging import RichHandler  # type: ignore

        console_handler = RichHandler(rich_tracebacks=False, markup=True)
    except Exception:
        console_handler = logging.StreamHandler()

    handlers = [console_handler]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
        except OSError as exc:
            fallback = Path(tempfile.gettempdir()) / "researchos.run.log"
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
    # RichHandler already formats level/name; keep a clean message format
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
    )
    # Silence noisy HTTP client logs (HF hub / httpx / requests / urllib3).
    for noisy in [
        "httpx",
        "requests",
        "urllib3",
        "huggingface_hub",
        "transformers",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
