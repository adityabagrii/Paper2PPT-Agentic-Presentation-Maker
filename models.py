"""Pydantic models for slide and deck structures."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SlideSpec(BaseModel):
    title: str
    bullets: List[str] = Field(default_factory=list)
    speaker_notes: str = ""
    figure_suggestions: List[str] = Field(default_factory=list)


class DeckOutline(BaseModel):
    deck_title: str
    arxiv_id: str
    slides: List[SlideSpec]
    citations: List[str] = Field(default_factory=list)
