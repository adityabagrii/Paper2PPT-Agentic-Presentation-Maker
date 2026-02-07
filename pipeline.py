from __future__ import annotations

"""Core pipeline with class-based organization."""
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from .arxiv_utils import download_and_extract_arxiv_source, get_arxiv_metadata
    from .llm import safe_invoke
    from .models import DeckOutline
    from .pdf_utils import extract_pdf_content
    from .web_utils import search_web
    from .tex_utils import (
        beamer_from_outline,
        beamer_from_outline_with_figs,
        build_paper_text,
        find_main_tex_file,
        flatten_tex,
        write_beamer,
    )
except Exception:
    from arxiv_utils import download_and_extract_arxiv_source, get_arxiv_metadata
    from llm import safe_invoke
    from models import DeckOutline
    from pdf_utils import extract_pdf_content
    from web_utils import search_web
    from tex_utils import (
        beamer_from_outline,
        beamer_from_outline_with_figs,
        build_paper_text,
        find_main_tex_file,
        flatten_tex,
        write_beamer,
    )

logger = logging.getLogger("paper2ppt")
TQDM_NCOLS = 100


@dataclass
class RunConfig:
    arxiv_id: str
    pdf_path: Optional[Path]
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


class OutlineJSONStore:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._index = 0

    def save(self, outline: DeckOutline) -> Path:
        self._index += 1
        path = self.out_dir / f"outline-{self._index}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(outline.model_dump(), f, indent=2, ensure_ascii=False)
        return path


class ArxivClient:
    def get_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        return get_arxiv_metadata(arxiv_id)

    def download_source(self, arxiv_id: str, out_dir: Path) -> Path:
        return download_and_extract_arxiv_source(arxiv_id, out_dir)


class OutlineBuilder:
    def __init__(self, llm, cfg: RunConfig, arxiv_client: ArxivClient) -> None:
        self.llm = llm
        self.cfg = cfg
        self.arxiv_client = arxiv_client

    @staticmethod
    def chunk_text(s: str, chunk_chars: int) -> List[str]:
        s = s.strip()
        return [s[i : i + chunk_chars] for i in range(0, len(s), chunk_chars)]

    @staticmethod
    def _preview_text(s: str, max_len: int = 60) -> str:
        s = re.sub(r"\s+", " ", (s or "").strip())
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

    @staticmethod
    def try_extract_json(text: str) -> Optional[str]:
        t = (text or "").strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\n", "", t)
            t = re.sub(r"\n```$", "", t).strip()

        start = t.find("{")
        if start == -1:
            return None

        depth = 0
        for j in range(start, len(t)):
            if t[j] == "{":
                depth += 1
            elif t[j] == "}":
                depth -= 1
                if depth == 0:
                    return t[start : j + 1]
        return None

    def summarize_chunk(
        self,
        i: int,
        chunk: str,
        meta: dict,
        user_query: str = "",
        web_context: str = "",
    ) -> str:
        for size in [1500, 1200, 900, 700, 500, 350]:
            snippet = chunk[:size]
            query_block = f"\nUser query: {user_query}\n" if user_query else ""
            web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""
            prompt = f"""
Paper title: {meta['title']}
Abstract: {meta['abstract']}
{query_block}{web_block}

Summarize chunk {i}. Plain text ONLY.

Include:
- Key ideas (max 5 bullets)
- Methods/approach
- Experiments/results (if present)
- Limitations/notes (if present)

Chunk:
{snippet}
""".strip()
            out = safe_invoke(logger, self.llm, prompt, retries=6)
            if out.strip():
                return out.strip()
        raise RuntimeError(f"Chunk {i} failed repeatedly (empty output).")

    def get_slide_titles(
        self,
        meta: dict,
        merged_summary: str,
        feedback: str = "",
        user_query: str = "",
        web_context: str = "",
    ) -> dict:
        summary = re.sub(r"\s+", " ", merged_summary).strip()[:1200]
        feedback_block = f"\nUser feedback:\n{feedback}\n" if feedback.strip() else ""
        query_block = f"\nUser query:\n{user_query}\n" if user_query else ""
        web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""

        query_rule = (
            f"- The deck must answer the user query; do not just summarize the paper\n"
            if user_query
            else ""
        )

        prompt = f"""
Return ONLY JSON.

Schema:
{{
  "deck_title": "string",
  "arxiv_id": "{self.cfg.arxiv_id}",
  "slide_titles": ["string", "..."]  // exactly {self.cfg.slide_count}
}}

Rules:
- Exactly {self.cfg.slide_count} titles
- Cover: motivation, problem, key idea, method, experiments, results, limitations, takeaways
- No extra keys
{query_rule}

Title: {meta['title']}
Abstract: {meta['abstract']}
Summary: {summary}
{query_block}{web_block}
{feedback_block}
""".strip()

        raw = safe_invoke(logger, self.llm, prompt, retries=6)
        js = self.try_extract_json(raw)
        if js is None:
            logger.error("RAW HEAD: %s", raw[:400])
            logger.error("RAW TAIL: %s", raw[-400:])
            raise RuntimeError("Could not extract slide_titles JSON.")
        obj = json.loads(js)
        if len(obj.get("slide_titles", [])) != self.cfg.slide_count:
            raise RuntimeError(f"slide_titles must have exactly {self.cfg.slide_count} entries")
        return obj

    def make_slide(
        self,
        meta: dict,
        slide_title: str,
        merged_summary: str,
        idx: int,
        feedback: str = "",
        include_speaker_notes: bool = True,
        user_query: str = "",
        web_context: str = "",
    ) -> dict:
        ctx = re.sub(r"\s+", " ", merged_summary).strip()[:1600]
        feedback_block = f"\nUser feedback:\n{feedback}\n" if feedback.strip() else ""
        query_block = f"\nUser query:\n{user_query}\n" if user_query else ""
        web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""
        source_rule = (
            "\n- If you use a web source, append '(source: URL)' to the bullet text\n"
            if web_context
            else ""
        )
        query_rule = (
            "\n- The slide content must answer the user query (not just summarize)\n"
            if user_query
            else ""
        )

        notes_schema = (
            '  "speaker_notes": "string",             // 1-3 sentences\n'
            if include_speaker_notes
            else ""
        )

        prompt = f"""
Return ONLY JSON.

Schema:
{{
  "title": "{slide_title}",
  "bullets": ["string", "..."],          // exactly {self.cfg.bullets_per_slide} bullets
{notes_schema}  "figure_suggestions": ["string", "..."]// 0-3 items
}}

Rules:
- bullets must be plain strings (no LaTeX)
- keep bullets concise and faithful
- no extra keys
{source_rule}
{query_rule}

Paper title: {meta['title']}
Abstract: {meta['abstract']}
Context: {ctx}
{query_block}{web_block}
{feedback_block}

Generate slide #{idx}: {slide_title}
""".strip()

        def _fallback_slide() -> dict:
            bullets = [f"TBD: {slide_title} (generation failed)"]
            while len(bullets) < self.cfg.bullets_per_slide:
                bullets.append("TBD: regenerate this slide")
            return {
                "title": slide_title,
                "bullets": bullets[: self.cfg.bullets_per_slide],
                "speaker_notes": "" if include_speaker_notes else "",
                "figure_suggestions": [],
            }

        for attempt in range(1, 4):
            raw = safe_invoke(logger, self.llm, prompt, retries=6)
            js = self.try_extract_json(raw)
            if js is None:
                fix = safe_invoke(
                    logger,
                    self.llm,
                    "Return ONLY valid JSON for the schema. Fix this:\n" + raw[:1800],
                    retries=6,
                )
                js = self.try_extract_json(fix)
                if js is None:
                    logger.error("Slide %s attempt %s JSON extraction failed.", idx, attempt)
                    logger.error("RAW HEAD: %s", raw[:400])
                    logger.error("RAW TAIL: %s", raw[-400:])
                    continue

            try:
                s = json.loads(js)
            except Exception:
                logger.error("Slide %s attempt %s JSON parse failed.", idx, attempt)
                continue

            if len(s.get("bullets", [])) != self.cfg.bullets_per_slide:
                fix_prompt = (
                    "Return ONLY valid JSON for the same schema. "
                    f"Fix bullets to have exactly {self.cfg.bullets_per_slide} items. "
                    "Keep title and figure_suggestions unchanged. "
                    "Here is the JSON to fix:\n"
                    + json.dumps(s, ensure_ascii=False)
                )
                fixed = safe_invoke(logger, self.llm, fix_prompt, retries=6)
                fixed_js = self.try_extract_json(fixed) or fixed
                try:
                    s = json.loads(fixed_js)
                except Exception:
                    logger.error("Slide %s attempt %s bullets fix parse failed.", idx, attempt)
                    continue
                if len(s.get("bullets", [])) != self.cfg.bullets_per_slide:
                    logger.error("Slide %s attempt %s bullets count still off.", idx, attempt)
                    continue

            if include_speaker_notes:
                if len(s.get("speaker_notes", "").strip()) < 5:
                    fix_prompt = (
                        "Return ONLY valid JSON for the same schema. "
                        "Fix speaker_notes to be 1-3 sentences. "
                        "Keep title, bullets, and figure_suggestions unchanged. "
                        "Here is the JSON to fix:\n"
                        + json.dumps(s, ensure_ascii=False)
                    )
                    fixed = safe_invoke(logger, self.llm, fix_prompt, retries=6)
                    fixed_js = self.try_extract_json(fixed) or fixed
                    try:
                        s = json.loads(fixed_js)
                    except Exception:
                        logger.error("Slide %s attempt %s speaker notes fix parse failed.", idx, attempt)
                        continue
                    if len(s.get("speaker_notes", "").strip()) < 5:
                        logger.error("Slide %s attempt %s speaker notes still too short.", idx, attempt)
                        continue
            else:
                s["speaker_notes"] = ""

            if "figure_suggestions" not in s:
                s["figure_suggestions"] = []
            return s

        logger.error("Slide %s failed after retries; using fallback.", idx)
        return _fallback_slide()

    def build_outline_once(
        self,
    ) -> Tuple[DeckOutline, Dict[str, Any], str, Dict[str, Any], str, List[Dict[str, str]]]:
        if self.cfg.pdf_path:
            logger.info("Reading local PDF: %s", self.cfg.pdf_path)
            pdf_data = extract_pdf_content(self.cfg.pdf_path, self.cfg.work_dir)
            meta = {
                "title": pdf_data["title"],
                "abstract": "",
                "url": str(self.cfg.pdf_path),
            }
            img_lines = []
            for img in pdf_data["images"]:
                img_lines.append(f"Image (page {img['page']}): {img['path']}")
            images_block = "\n".join(img_lines)
            paper_text = pdf_data["text"]
            if images_block:
                paper_text = f"{paper_text}\n\n[IMAGES]\n{images_block}".strip()
            logger.info("PDF text chars: %s", len(paper_text))
            if len(paper_text) <= 200 and not images_block:
                raise RuntimeError("PDF text too small and no images found; scanned PDF may require OCR.")
        else:
            logger.info("Fetching arXiv metadata...")
            meta = self.arxiv_client.get_metadata(self.cfg.arxiv_id)

            logger.info("Downloading and extracting arXiv source...")
            src_dir = self.arxiv_client.download_source(self.cfg.arxiv_id, self.cfg.work_dir)
            main_tex = find_main_tex_file(src_dir)
            flat = flatten_tex(main_tex, max_files=120)
            paper_text = build_paper_text(flat, max_chars=None)

            logger.info("Main TeX file: %s", main_tex)
            logger.info("paper_text chars: %s", len(paper_text))
            if len(paper_text) <= 500:
                raise RuntimeError("paper_text too small; main tex likely wrong.")

        web_sources = []
        web_context = ""
        if self.cfg.user_query and self.cfg.web_search:
            logger.info("Running web search for query: %s", self.cfg.user_query)
            web_sources = search_web(self.cfg.user_query, max_results=5)
            if web_sources:
                print("\nTop web results:")
                for i, s in enumerate(web_sources, 1):
                    print(f"{i}. {s['title']} - {s['url']}")
                print("")
                lines = []
                for i, s in enumerate(web_sources, 1):
                    lines.append(f"{i}. {s['title']} - {s['url']}\n   {s['snippet']}")
                web_context = "\n".join(lines)

        chunks = self.chunk_text(paper_text, 1500)
        N = min(self.cfg.max_summary_chunks, len(chunks))
        sums = []

        logger.info("Summarizing paper (%s chunks)...", N)
        prev_summary_preview = ""
        with tqdm(
            range(1, N + 1),
            desc="Summarize",
            unit="chunk",
            ncols=TQDM_NCOLS,
            dynamic_ncols=False,
        ) as bar:
            for i in bar:
                chunk_preview = self._preview_text(chunks[i - 1], max_len=50)
                bar.set_postfix_str(f"chunk: {chunk_preview} | prev: {prev_summary_preview}")
                s = self.summarize_chunk(i, chunks[i - 1], meta, self.cfg.user_query, web_context)
                sums.append(s)
                prev_summary_preview = self._preview_text(s, max_len=50)

        merged_summary = "\n\n".join(sums)

        logger.info("Generating slide titles (%s)...", self.cfg.slide_count)
        titles_obj = self.get_slide_titles(meta, merged_summary, user_query=self.cfg.user_query, web_context=web_context)

        logger.info("Generating slides (%s)...", self.cfg.slide_count)
        slides = []
        for idx, title in tqdm(
            list(enumerate(titles_obj["slide_titles"], 1)),
            desc="Slides",
            unit="slide",
            ncols=TQDM_NCOLS,
            dynamic_ncols=False,
        ):
            slides.append(
                self.make_slide(
                    meta,
                    title,
                    merged_summary,
                    idx,
                    include_speaker_notes=self.cfg.include_speaker_notes,
                    user_query=self.cfg.user_query,
                    web_context=web_context,
                )
            )

        if self.cfg.pdf_path:
            citations = [f"Local PDF: {self.cfg.pdf_path}"]
        else:
            citations = [f"arXiv:{self.cfg.arxiv_id}", meta.get("url", "")]
        if web_sources:
            citations.extend([f"{s['title']} - {s['url']}" for s in web_sources])

        outline_dict = {
            "deck_title": titles_obj["deck_title"],
            "arxiv_id": self.cfg.arxiv_id,
            "slides": slides,
            "citations": citations,
        }
        outline = DeckOutline.model_validate(outline_dict)
        return outline, meta, merged_summary, titles_obj, web_context, web_sources

    def regenerate_titles_with_feedback(
        self,
        meta: dict,
        merged_summary: str,
        prev_titles: List[str],
        feedback: str,
        user_query: str = "",
        web_context: str = "",
    ) -> dict:
        summary = re.sub(r"\s+", " ", merged_summary).strip()[:1200]
        prev = "\n".join([f"{i+1}. {t}" for i, t in enumerate(prev_titles)])
        query_block = f"\nUser query:\n{user_query}\n" if user_query else ""
        web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""

        prompt = f"""
Return ONLY JSON.

Schema:
{{
  "deck_title": "string",
  "arxiv_id": "{self.cfg.arxiv_id}",
  "slide_titles": ["string", "..."]  // exactly {self.cfg.slide_count}
}}

Previous slide titles:
{prev}

User feedback:
{feedback}

Revise the slide titles accordingly while keeping exactly {self.cfg.slide_count}.

Title: {meta['title']}
Abstract: {meta['abstract']}
Summary: {summary}
{query_block}{web_block}
""".strip()

        raw = safe_invoke(logger, self.llm, prompt, retries=6)
        js = self.try_extract_json(raw)
        if js is None:
            logger.error("RAW HEAD: %s", raw[:400])
            logger.error("RAW TAIL: %s", raw[-400:])
            raise RuntimeError("Could not extract revised titles JSON.")
        obj = json.loads(js)
        if len(obj.get("slide_titles", [])) != self.cfg.slide_count:
            raise RuntimeError(f"slide_titles must have exactly {self.cfg.slide_count} entries")
        return obj


class FigureAsset:
    def __init__(self, tex_path: str, resolved_path: str, caption: str, label: Optional[str]) -> None:
        self.tex_path = tex_path
        self.resolved_path = resolved_path
        self.caption = caption
        self.label = label


class FigurePlanner:
    FIG_ENV_RE = re.compile(r"\\begin\{figure\}[\s\S]*?\\end\{figure\}", re.MULTILINE)
    CAP_RE = re.compile(r"\\caption\*?\{([\s\S]*?)\}")
    LAB_RE = re.compile(r"\\label\{([\s\S]*?)\}")
    INC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")

    @staticmethod
    def _strip_tex(s: str) -> str:
        s = re.sub(r"(?m)(?<!\\\\)%.*$", "", s)
        s = re.sub(r"\\\\[a-zA-Z]+\\*?(?:\\[[^\\]]*\\])?(?:\\{[^}]*\\})?", " ", s)
        s = s.replace("{", " ").replace("}", " ").replace("\\\\", " ")
        s = re.sub(r"\\s+", " ", s).strip()
        return s

    @staticmethod
    def resolve_graphic_path(src_dir: Path, tex_ref: str) -> Optional[Path]:
        tex_ref = tex_ref.strip()
        candidates = [
            src_dir / tex_ref,
            src_dir / (tex_ref + ".pdf"),
            src_dir / (tex_ref + ".png"),
            src_dir / (tex_ref + ".jpg"),
            src_dir / (tex_ref + ".jpeg"),
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c

        base = Path(tex_ref).name
        for ext in [".pdf", ".png", ".jpg", ".jpeg"]:
            hits = list(src_dir.rglob(base if base.endswith(ext) else base + ext))
            if hits:
                return hits[0]
        return None

    def extract_figures(self, flat_tex: str, src_dir: Path) -> List[FigureAsset]:
        figs: List[FigureAsset] = []
        for env in self.FIG_ENV_RE.findall(flat_tex):
            cap_m = self.CAP_RE.search(env)
            caption = self._strip_tex(cap_m.group(1)) if cap_m else ""
            lab_m = self.LAB_RE.search(env)
            label = lab_m.group(1).strip() if lab_m else None

            for inc_m in self.INC_RE.finditer(env):
                tex_ref = inc_m.group(1).strip()
                p = self.resolve_graphic_path(src_dir, tex_ref)
                if p is None:
                    continue
                figs.append(FigureAsset(tex_ref, str(p), caption, label))

        uniq: Dict[str, FigureAsset] = {}
        for f in figs:
            uniq[f.resolved_path] = f
        return list(uniq.values())

    def plan_with_llm(self, llm, outline: DeckOutline, fig_assets: List[FigureAsset], max_figs: int = 12) -> dict:
        if not fig_assets:
            return {"slides": []}

        figs = fig_assets[:max_figs]
        catalog = "\n".join([f"- {Path(f.resolved_path).name}: {f.caption[:120]}" for f in figs])
        slide_titles = "\n".join([f"{i+1}. {s.title}" for i, s in enumerate(outline.slides)])

        prompt = f"""
Return ONLY JSON.

Schema:
{{
  "slides": [
    {{
      "slide_index": 1,
      "figures": [{{"file": "filename.ext", "why": "short", "caption": "short"}}]
    }}
  ]
}}

Rules:
- Only choose from the filenames listed below.
- At most 1 figure per slide.
- Skip slides without a strong matching figure.
- Keep explanations short.
- Generate a short, descriptive caption for the selected figure.

Slides:
{slide_titles}

Available figures (filename: caption):
{catalog}
""".strip()

        raw = safe_invoke(logger, llm, prompt, retries=6)
        js = OutlineBuilder.try_extract_json(raw)
        if js is None:
            logger.warning("Figure plan JSON parse failed. Skipping figures.")
            return {"slides": []}

        try:
            obj = json.loads(js)
        except Exception:
            return {"slides": []}

        allowed = {Path(f.resolved_path).name for f in figs}
        cleaned = {"slides": []}
        for s in obj.get("slides", []):
            if not isinstance(s, dict):
                continue
            idx = s.get("slide_index")
            figs_out = []
            for g in s.get("figures", []):
                name = g.get("file")
                if name in allowed:
                    figs_out.append({
                        "file": name,
                        "why": g.get("why", ""),
                        "caption": g.get("caption", ""),
                    })
            if idx and figs_out:
                cleaned["slides"].append({"slide_index": idx, "figures": figs_out})

        return cleaned

    def materialize(self, fig_plan: dict, fig_assets: List[FigureAsset], out_dir: Path) -> dict:
        out_dir = Path(out_dir)
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        by_name = {Path(f.resolved_path).name: f for f in fig_assets}

        resolved = {"slides": []}
        for s in fig_plan.get("slides", []):
            new_item = {"slide_index": s["slide_index"], "figures": []}
            for g in s.get("figures", []):
                name = g.get("file")
                if not name or name not in by_name:
                    continue
                src_path = Path(by_name[name].resolved_path)
                dst_path = fig_dir / name
                shutil.copy2(src_path, dst_path)
                if not dst_path.exists():
                    continue
                new_item["figures"].append({
                    "file": str(Path("figures") / name),
                    "why": g.get("why", ""),
                    "caption": g.get("caption", ""),
                })
            if new_item["figures"]:
                resolved["slides"].append(new_item)

        return resolved


class Renderer:
    @staticmethod
    def slugify_filename(s: str, max_len: int = 80) -> str:
        s = s.strip()
        s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
        s = s.strip("_")
        if not s:
            return "presentation"
        return s[:max_len]

    @staticmethod
    def compile_beamer(tex_path: Path) -> Optional[Path]:
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
        filename_base = self.slugify_filename(outline.deck_title)
        logger.info("Preparing figures from arXiv source...")
        src_dir = work_dir / "arxiv_source"
        if not src_dir.exists():
            src_dir = download_and_extract_arxiv_source(arxiv_id, work_dir)

        main_tex = find_main_tex_file(src_dir)
        flat = flatten_tex(main_tex, max_files=120)

        fig_assets = fig_planner.extract_figures(flat, src_dir)
        logger.info("Figures found: %s", len(fig_assets))

        fig_plan = fig_planner.plan_with_llm(llm, outline, fig_assets, max_figs=12)
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


class Pipeline:
    def __init__(self, cfg: RunConfig, llm) -> None:
        self.cfg = cfg
        self.llm = llm
        self.arxiv_client = ArxivClient()
        self.outline_builder = OutlineBuilder(llm, cfg, self.arxiv_client)
        self.outline_store = OutlineJSONStore(cfg.out_dir)
        self.figure_planner = FigurePlanner()
        self.renderer = Renderer()

    def sanity_checks(self) -> None:
        logger.info("Running sanity checks...")
        if self.cfg.slide_count < 2:
            raise ValueError("slide_count must be >= 2")
        if self.cfg.bullets_per_slide < 1:
            raise ValueError("bullets_per_slide must be >= 1")
        if not self.cfg.arxiv_id and not self.cfg.pdf_path:
            raise ValueError("arxiv_id or pdf_path must be provided")
        if self.cfg.pdf_path and not self.cfg.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.cfg.pdf_path}")

        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        if not self.cfg.skip_llm_sanity:
            x = safe_invoke(logger, self.llm, "Reply with exactly: OK", debug=self.cfg.verbose)
            logger.info("LLM sanity: %r", x[:50])
            if "OK" not in x:
                raise RuntimeError("LLM sanity check failed. Ensure your NVIDIA_API_KEY and model are valid.")

    @staticmethod
    def print_outline(outline: DeckOutline) -> None:
        print("\n" + "=" * 90)
        print("DECK:", outline.deck_title)
        print("arXiv:", outline.arxiv_id)
        print("=" * 90)
        for i, sl in enumerate(outline.slides, 1):
            print(f"\n{i:02d}. {sl.title}")
            for b in sl.bullets:
                print("   -", b)
            if sl.figure_suggestions:
                print("   [figs]", "; ".join(sl.figure_suggestions))
            print("   [notes]", (sl.speaker_notes[:140] + ("..." if len(sl.speaker_notes) > 140 else "")))
        print("\n" + "=" * 90)

    def build_outline_with_approval(self, max_rounds: int = 3) -> Tuple[DeckOutline, Dict[str, Any]]:
        outline, meta, merged_summary, titles_obj, web_context, web_sources = self.outline_builder.build_outline_once()
        saved_path = self.outline_store.save(outline)
        logger.info("Saved outline draft: %s", saved_path)

        if not self.cfg.approve:
            return outline, meta

        for round_no in range(1, max_rounds + 1):
            self.print_outline(outline)

            ans = input("\nApprove outline? Type 'y' to approve, or type feedback to regenerate titles: ").strip()
            if ans.lower() in ["y", "yes"]:
                print("Approved.")
                return outline, meta

            feedback = ans
            print("\nRegenerating slide titles based on feedback...\n")
            revised = self.outline_builder.regenerate_titles_with_feedback(
                meta,
                merged_summary,
                prev_titles=titles_obj["slide_titles"],
                feedback=feedback,
                user_query=self.cfg.user_query,
                web_context=web_context,
            )
            titles_obj = revised

            slides = []
            for idx, title in tqdm(
                list(enumerate(titles_obj["slide_titles"], 1)),
                desc="Slides",
                unit="slide",
                ncols=TQDM_NCOLS,
                dynamic_ncols=False,
            ):
                slides.append(
                    self.outline_builder.make_slide(
                        meta,
                        title,
                        merged_summary,
                        idx,
                        feedback=feedback,
                        include_speaker_notes=self.cfg.include_speaker_notes,
                        user_query=self.cfg.user_query,
                        web_context=web_context,
                    )
                )

            if self.cfg.pdf_path:
                citations = [f"Local PDF: {self.cfg.pdf_path}"]
            else:
                citations = [f"arXiv:{self.cfg.arxiv_id}", meta.get("url", "")]
            if web_sources:
                citations.extend([f"{s['title']} - {s['url']}" for s in web_sources])

            outline_dict = {
                "deck_title": titles_obj["deck_title"],
                "arxiv_id": self.cfg.arxiv_id,
                "slides": slides,
                "citations": citations,
            }
            outline = DeckOutline.model_validate(outline_dict)
            saved_path = self.outline_store.save(outline)
            logger.info("Saved outline draft: %s", saved_path)

        print("Max rounds reached; proceeding with latest outline.")
        return outline, meta

    def run(self) -> Tuple[DeckOutline, Optional[Path], Optional[Path]]:
        self.sanity_checks()
        outline, _meta = self.build_outline_with_approval(max_rounds=3)

        if self.cfg.use_figures and self.cfg.pdf_path:
            logger.warning("Figure insertion is not supported for local PDFs; continuing without figures.")
            tex_path, pdf_path = self.renderer.render(outline, self.cfg.out_dir)
        elif self.cfg.use_figures:
            tex_path, pdf_path = self.renderer.render_with_figs(
                self.llm,
                outline,
                self.cfg.arxiv_id,
                self.cfg.work_dir,
                self.cfg.out_dir,
                self.figure_planner,
            )
        else:
            tex_path, pdf_path = self.renderer.render(outline, self.cfg.out_dir)

        return outline, tex_path, pdf_path
