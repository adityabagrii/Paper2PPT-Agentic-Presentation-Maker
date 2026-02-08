from __future__ import annotations

"""Core pipeline with class-based organization.

This module now acts as a thin facade that re-exports the main pipeline
components from smaller modules.
"""

try:
    from .pipeline_common import RunConfig, PaperContext, OutlineJSONStore, _progress_path, logger, TQDM_NCOLS
    from .pipeline_arxiv import ArxivClient
    from .pipeline_outline import OutlineBuilder
    from .pipeline_figures import FigureAsset, FigurePlanner
    from .pipeline_render import Renderer
    from .pipeline_core import Pipeline
except Exception:
    from pipeline_common import RunConfig, PaperContext, OutlineJSONStore, _progress_path, logger, TQDM_NCOLS
    from pipeline_arxiv import ArxivClient
    from pipeline_outline import OutlineBuilder
    from pipeline_figures import FigureAsset, FigurePlanner
    from pipeline_render import Renderer
    from pipeline_core import Pipeline

__all__ = [
    "RunConfig",
    "PaperContext",
    "OutlineJSONStore",
    "_progress_path",
    "logger",
    "TQDM_NCOLS",
    "ArxivClient",
    "OutlineBuilder",
    "FigureAsset",
    "FigurePlanner",
    "Renderer",
    "Pipeline",
]
