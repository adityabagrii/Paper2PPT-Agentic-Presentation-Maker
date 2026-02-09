from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .models import DeckOutline
except Exception:
    from models import DeckOutline

logger = logging.getLogger("researchos")
TQDM_NCOLS = 100


@dataclass
class RunConfig:
    arxiv_ids: List[str]
    pdf_paths: List[Path]
    md_paths: List[Path]
    work_dir: Path
    out_dir: Path
    slide_count: int
    bullets_per_slide: int
    max_summary_chunks: int
    approve: bool
    verbose: bool
    skip_llm_sanity: bool
    llm_model: str
    llm_api_key: str
    use_figures: bool
    include_speaker_notes: bool
    user_query: str
    web_search: bool
    web_search_providers: List[str]
    arxiv_only_search: bool
    retry_slides: int
    retry_empty: int
    interactive: bool
    check_interval: int
    resume_path: Optional[Path]
    generate_flowcharts: bool
    min_flowcharts: int
    max_flowcharts: int
    flowchart_structure: str
    flowchart_depth: int
    titles_only: bool
    topic: str
    max_web_results: int
    max_web_pdfs: int
    max_web_queries: int
    topic_scholarly_only: bool
    max_llm_workers: int
    diagram_style: str
    topic_must_include: List[str]
    topic_exclude: List[str]
    topic_allow_domains: List[str]
    require_evidence: bool
    diagram_intent_aware: bool
    auto_comparisons: bool
    baseline_framing: bool
    quant_results: bool
    teaching_mode: bool
    read_mode: bool
    viva_mode: bool
    describe_experiments: bool
    exam_prep: bool
    implementation_notes: bool
    reproduction_checklist: bool
    index_paper: bool
    index_search_query: str
    daily_brief: bool
    cache_summary: bool
    chat_mode: bool
    figures_only: bool
    revise_with_figures_path: Optional[Path]
    revise_with_diagrams_path: Optional[Path]
    revise_with_figures_target: str
    revise_with_diagrams_target: str
    edit_run_path: Optional[Path]
    edit_instructions: str
    edit_target: str
    remove_media_path: Optional[Path]
    remove_media_target: str


@dataclass
class PaperContext:
    meta: Dict[str, Any]
    paper_text: str
    merged_summary: str
    sources_block: str
    source_label: str
    web_context: str
    citations: List[str]
    sources: List[Dict[str, Any]]


class OutlineJSONStore:
    def __init__(self, out_dir: Path) -> None:
        """Initialize.

        Args:
            out_dir (Path):

        Returns:
            None:
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._index = 0

    def save(self, outline: DeckOutline) -> Path:
        """Save.

        Args:
            outline (DeckOutline):

        Returns:
            Path:
        """
        self._index += 1
        path = self.out_dir / f"outline-{self._index}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(outline.model_dump(), f, indent=2, ensure_ascii=False)
        return path


def _progress_path(out_dir: Path) -> Path:
    """Function progress path.

    Args:
        out_dir (Path):

    Returns:
        Path:
    """
    return Path(out_dir) / "progress.json"
