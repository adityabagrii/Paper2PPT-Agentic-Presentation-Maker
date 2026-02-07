"""LLM wrapper utilities (init + safe invoke)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    from .logging_utils import setup_logging  # noqa: F401
except Exception:
    from logging_utils import setup_logging  # noqa: F401

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except Exception as e:  # pragma: no cover
    ChatNVIDIA = None
    _IMPORT_ERR = e


@dataclass
class LLMConfig:
    model: str
    api_key: str


def init_llm(cfg: LLMConfig):
    """Initialize the NVIDIA NIM chat model."""
    if ChatNVIDIA is None:
        raise RuntimeError(f"langchain_nvidia_ai_endpoints not available: {_IMPORT_ERR}")
    return ChatNVIDIA(model=cfg.model, api_key=cfg.api_key)


def safe_invoke(logger, llm, prompt: str, retries: int = 6, sleep_base: float = 0.8, debug: bool = False) -> str:
    """Invoke LLM with retries for flaky or empty responses."""
    last = ""
    for k in range(retries):
        out = llm.invoke(prompt).content or ""
        if debug:
            logger.debug("[safe_invoke] attempt %s/%s -> len=%s head=%r", k + 1, retries, len(out), out[:40])
        if out.strip():
            return out
        last = out
        time.sleep(sleep_base * (k + 1))
    return last
