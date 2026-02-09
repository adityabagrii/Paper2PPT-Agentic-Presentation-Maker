from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

try:
    from .arxiv_utils import download_and_extract_arxiv_source
    from .models import DeckOutline
    from .tex_utils import (
        beamer_from_outline,
        beamer_from_outline_with_figs,
        find_main_tex_file,
        flatten_tex,
        write_beamer,
    )
except Exception:
    from arxiv_utils import download_and_extract_arxiv_source
    from models import DeckOutline
    from tex_utils import (
        beamer_from_outline,
        beamer_from_outline_with_figs,
        find_main_tex_file,
        flatten_tex,
        write_beamer,
    )

try:
    from .pipeline_common import logger
    from .pipeline_figures import FigurePlanner
except Exception:
    from pipeline_common import logger
    from pipeline_figures import FigurePlanner


class Renderer:
    @staticmethod
    def slugify_filename(s: str, max_len: int = 80) -> str:
        """Slugify filename.
        
        Args:
            s (str):
            max_len (int):
        
        Returns:
            str:
        """
        s = s.strip()
        s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
        s = s.strip("_")
        if not s:
            return "presentation"
        return s[:max_len]

    @staticmethod
    def compile_beamer(tex_path: Path) -> Optional[Path]:
        """Compile beamer.
        
        Args:
            tex_path (Path):
        
        Returns:
            Optional[Path]:
        """
        tex_path = Path(tex_path)

        if shutil.which("pdflatex") is None:
            logger.error("pdflatex not found. Install BasicTeX/MacTeX or MiKTeX and restart terminal.")
            return None

        for _ in range(2):
            cmd = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
            r = subprocess.run(cmd, cwd=str(tex_path.parent), capture_output=True, text=True)
            if r.returncode != 0:
                logger.error("pdflatex failed. Tail:\n%s", (r.stdout + "\n" + r.stderr)[-2000:])
                return None

        pdf_path = tex_path.with_suffix(".pdf")
        return pdf_path if pdf_path.exists() else None

    def render(self, outline: DeckOutline, out_dir: Path) -> Tuple[Path, Optional[Path]]:
        """Render.
        
        Args:
            outline (DeckOutline):
            out_dir (Path):
        
        Returns:
            Tuple[Path, Optional[Path]]:
        """
        filename_base = self.slugify_filename(outline.deck_title)
        logger.info("Rendering Beamer LaTeX...")
        tex = beamer_from_outline(outline)
        tex_path = write_beamer(tex, out_dir, filename_base=filename_base)

        logger.info("Compiling PDF (pdflatex)...")
        pdf_path = self.compile_beamer(tex_path)
        return tex_path, pdf_path

    def render_with_figs(
        self,
        llm,
        outline: DeckOutline,
        arxiv_id: str,
        work_dir: Path,
        out_dir: Path,
        fig_planner: FigurePlanner,
    ) -> Tuple[Path, Optional[Path]]:
        """Render with figs.
        
        Args:
            llm (Any):
            outline (DeckOutline):
            arxiv_id (str):
            work_dir (Path):
            out_dir (Path):
            fig_planner (FigurePlanner):
        
        Returns:
            Tuple[Path, Optional[Path]]:
        """
        filename_base = self.slugify_filename(outline.deck_title)
        logger.info("Preparing figures from arXiv source...")
        src_dir = work_dir / "arxiv_source"
        if not src_dir.exists():
            src_dir = download_and_extract_arxiv_source(arxiv_id, work_dir)

        main_tex = find_main_tex_file(src_dir)
        flat = flatten_tex(main_tex, max_files=120)

        fig_assets = fig_planner.extract_figures(flat, src_dir)
        logger.info("Figures found: %s", len(fig_assets))

        max_figs = len(fig_assets)
        fig_plan = fig_planner.plan_with_llm(llm, outline, fig_assets, max_figs=max_figs)
        resolved_fig_plan = fig_planner.materialize(fig_plan, fig_assets, out_dir)

        logger.info("Rendering Beamer LaTeX (with figures where valid)...")
        if resolved_fig_plan.get("slides"):
            tex = beamer_from_outline_with_figs(outline, resolved_fig_plan)
        else:
            tex = beamer_from_outline(outline)

        tex_path = write_beamer(tex, out_dir, filename_base=filename_base)
        logger.info("Compiling PDF (pdflatex)...")
        pdf_path = self.compile_beamer(tex_path)
        return tex_path, pdf_path
