from __future__ import annotations

"""Core pipeline with class-based organization."""
import json
import logging
import re
import shutil
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from .arxiv_utils import download_and_extract_arxiv_source, get_arxiv_metadata, extract_arxiv_id
    from .llm import safe_invoke
    from .models import DeckOutline
    from .pdf_utils import extract_pdf_content
    from .web_utils import search_web
    from .flowchart_utils import build_graphviz, render_graphviz
    from .tex_utils import (
        beamer_from_outline,
        beamer_from_outline_with_figs,
        build_paper_text,
        find_main_tex_file,
        flatten_tex,
        write_beamer,
    )
except Exception:
    from arxiv_utils import download_and_extract_arxiv_source, get_arxiv_metadata, extract_arxiv_id
    from llm import safe_invoke
    from models import DeckOutline
    from pdf_utils import extract_pdf_content
    from web_utils import search_web
    from flowchart_utils import build_graphviz, render_graphviz
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
    arxiv_ids: List[str]
    pdf_paths: List[Path]
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
    topic_scholarly_only: bool
    max_llm_workers: int


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


def _progress_path(out_dir: Path) -> Path:
    return Path(out_dir) / "progress.json"


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

    def _checkpoint(self, label: str, idx: int | None = None, total: int | None = None) -> None:
        if not self.cfg.interactive:
            return
        if idx is not None and total is not None:
            if idx % self.cfg.check_interval != 0 and idx != total:
                return
            prompt = f"[{label}] step {idx}/{total}. Press Enter to continue or type 'q' to quit: "
        else:
            prompt = f"[{label}] Press Enter to continue or type 'q' to quit: "
        ans = input(prompt).strip().lower()
        if ans in {"q", "quit", "exit"}:
            raise RuntimeError("Aborted by user.")

    def _prompt_feedback(self, label: str) -> str:
        if not self.cfg.interactive:
            return ""
        ans = input(f"[{label}] Provide guidance (or press Enter to skip): ").strip()
        return ans

    def _save_progress(self, state: dict) -> None:
        try:
            path = _progress_path(self.cfg.out_dir)
            with path.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to write progress.json")

    @staticmethod
    def _print_section(title: str, lines: List[str]) -> None:
        width = 96
        print("\n" + "=" * width)
        print(title)
        print("-" * width)
        for line in lines:
            wrapped = textwrap.fill(
                line,
                width=width,
                initial_indent="",
                subsequent_indent="",
            )
            print(wrapped)
        print("=" * width + "\n")

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
        sources_block: str = "",
    ) -> str:
        for size in [1500, 1200, 900, 700, 500, 350]:
            snippet = chunk[:size]
            query_block = f"\nUser query: {user_query}\n" if user_query else ""
            web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""
            sources_block = f"\nSources:\n{sources_block}\n" if sources_block else ""
            prompt = f"""
Paper title: {meta['title']}
Abstract: {meta['abstract']}
{query_block}{web_block}{sources_block}

Summarize chunk {i}. Plain text ONLY.

Include:
- Key ideas (max 5 bullets)
- Methods/approach
- Experiments/results (if present)
- Limitations/notes (if present)

Chunk:
{snippet}
""".strip()
            out = ""
            for attempt in range(1, self.cfg.retry_empty + 1):
                out = safe_invoke(logger, self.llm, prompt, retries=6)
                if out.strip():
                    return out.strip()
                logger.warning(
                    "Chunk %s returned empty output (attempt %s/%s).",
                    i,
                    attempt,
                    self.cfg.retry_empty,
                )

            print(f"\nLLM returned empty output for this chunk after {self.cfg.retry_empty} attempts.")
            print("Prompt used:\n" + prompt[:1500] + ("\n... [truncated]" if len(prompt) > 1500 else ""))
            ans = input("Type 's' to skip this chunk, or 'q' to quit: ").strip().lower()
            if ans in {"s", "skip"}:
                logger.warning("User chose to skip empty chunk %s.", i)
                return "SKIPPED: user chose to skip empty chunk."
            raise RuntimeError(f"Chunk {i} failed with empty output.")
        raise RuntimeError(f"Chunk {i} failed repeatedly (empty output).")

    def get_slide_titles(
        self,
        meta: dict,
        merged_summary: str,
        feedback: str = "",
        user_query: str = "",
        web_context: str = "",
        sources_block: str = "",
        source_label: str = "",
    ) -> dict:
        summary = re.sub(r"\s+", " ", merged_summary).strip()[:1200]
        feedback_block = f"\nUser feedback:\n{feedback}\n" if feedback.strip() else ""
        query_block = f"\nUser query:\n{user_query}\n" if user_query else ""
        web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""
        sources_block = f"\nSources:\n{sources_block}\n" if sources_block else ""

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
  "arxiv_id": "{source_label}",
  "slide_titles": ["string", "..."]  // exactly {self.cfg.slide_count}
}}

Rules:
- Exactly {self.cfg.slide_count} titles
- Cover: motivation, problem, key idea, method, experiments, results, limitations, takeaways
- No extra keys
- Deck title must reflect the user query and the source titles when provided
{query_rule}

Title: {meta['title']}
Abstract: {meta['abstract']}
Summary: {summary}
{query_block}{web_block}{sources_block}
{feedback_block}
""".strip()

        js = None
        last_raw = ""
        for attempt in range(1, 4):
            raw = safe_invoke(logger, self.llm, prompt, retries=6)
            last_raw = raw
            js = self.try_extract_json(raw)
            if js is not None:
                break
            logger.warning("Slide titles JSON extraction failed (attempt %s/3).", attempt)
            fix = safe_invoke(
                logger,
                self.llm,
                "Return ONLY valid JSON for the schema. Fix this:\n" + raw[:1800],
                retries=6,
            )
            js = self.try_extract_json(fix)
            if js is not None:
                break
        if js is None:
            logger.error("RAW HEAD: %s", last_raw[:400])
            logger.error("RAW TAIL: %s", last_raw[-400:])
            # Fallback: create placeholder titles to avoid crash
            obj = {
                "deck_title": meta.get("title", "Presentation"),
                "arxiv_id": source_label,
                "slide_titles": [f"Slide {i+1}" for i in range(self.cfg.slide_count)],
            }
            return obj

        obj = json.loads(js)
        titles = obj.get("slide_titles", [])
        if len(titles) != self.cfg.slide_count:
            fix_prompt = (
                "Return ONLY valid JSON for the same schema. "
                f"Ensure slide_titles has exactly {self.cfg.slide_count} items. "
                "Keep deck_title and arxiv_id unchanged. "
                "Here is the JSON to fix:\n"
                + json.dumps(obj, ensure_ascii=False)
            )
            fixed = safe_invoke(logger, self.llm, fix_prompt, retries=6)
            fixed_js = self.try_extract_json(fixed) or fixed
            try:
                obj = json.loads(fixed_js)
                titles = obj.get("slide_titles", [])
            except Exception:
                titles = []
            if len(titles) != self.cfg.slide_count:
                logger.error("slide_titles count mismatch; applying fallback padding/truncation.")
                if self.cfg.interactive:
                    print("\nCurrent slide titles:")
                    for i, t in enumerate(titles, 1):
                        print(f"{i}. {t}")
                    ans = input(
                        "Type feedback to refine titles, or press Enter to auto-fix: "
                    ).strip()
                    if ans:
                        refine_prompt = (
                            "Return ONLY valid JSON for the same schema. "
                            f"Ensure slide_titles has exactly {self.cfg.slide_count} items. "
                            "Apply this user feedback: "
                            + ans
                            + "\nHere is the JSON to fix:\n"
                            + json.dumps(obj, ensure_ascii=False)
                        )
                        refined = safe_invoke(logger, self.llm, refine_prompt, retries=6)
                        refined_js = self.try_extract_json(refined) or refined
                        try:
                            obj = json.loads(refined_js)
                            titles = obj.get("slide_titles", [])
                        except Exception:
                            titles = []
                # Fallback: pad or truncate to required length
                base = titles if titles else [f"Slide {i+1}" for i in range(self.cfg.slide_count)]
                if len(base) < self.cfg.slide_count:
                    base += [f"Slide {i+1}" for i in range(len(base), self.cfg.slide_count)]
                obj["slide_titles"] = base[: self.cfg.slide_count]
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
        sources_block: str = "",
    ) -> dict:
        ctx = re.sub(r"\s+", " ", merged_summary).strip()[:1600]
        feedback_block = f"\nUser feedback:\n{feedback}\n" if feedback.strip() else ""
        query_block = f"\nUser query:\n{user_query}\n" if user_query else ""
        web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""
        sources_block = f"\nSources:\n{sources_block}\n" if sources_block else ""
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
{notes_schema}  "figure_suggestions": ["string", "..."],// 0-3 items (optional, can be empty)
  "flowchart": {{
    "steps": ["string", "..."],          // 0-8 items; use 4-7 when applicable
    "structure": "linear|branch|cycle",
    "caption": "string"
  }},
  "graphviz_diagram_ideas": ["string", "..."] // 0-3 items; non-flowchart graph ideas
}}

Rules:
- bullets must be plain strings (no LaTeX)
- keep bullets concise and faithful
- no extra keys
- For method/system/algorithm slides, include a deep flowchart in flowchart.steps.
- Flowchart steps should be specific mechanisms (not vague).
- If not suitable, set flowchart.steps to [] and caption to "".
- graphviz_diagram_ideas should mention other diagram types: dependency graph, DAG, hierarchy, decision tree, etc.
{source_rule}
{query_rule}

Paper title: {meta['title']}
Abstract: {meta['abstract']}
Context: {ctx}
{query_block}{web_block}{sources_block}
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
                "flowchart": {"steps": [], "structure": "linear", "caption": ""},
                "graphviz_diagram_ideas": [],
            }

        for attempt in range(1, self.cfg.retry_slides + 1):
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
                    "Keep title, figure_suggestions, flowchart, and graphviz_diagram_ideas unchanged. "
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
                        "Keep title, bullets, figure_suggestions, flowchart, and graphviz_diagram_ideas unchanged. "
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
            if "graphviz_diagram_ideas" not in s:
                s["graphviz_diagram_ideas"] = []
            if "flowchart" not in s or not isinstance(s.get("flowchart"), dict):
                s["flowchart"] = {"steps": [], "structure": "linear", "caption": ""}
            else:
                s["flowchart"].setdefault("steps", [])
                s["flowchart"].setdefault("structure", "linear")
                s["flowchart"].setdefault("caption", "")
            return s

        logger.error("Slide %s failed after retries; using fallback.", idx)
        return _fallback_slide()

    def build_outline_once(
        self,
    ) -> Tuple[
        DeckOutline,
        Dict[str, Any],
        str,
        Dict[str, Any],
        str,
        List[Dict[str, str]],
        str,
        str,
        List[str],
    ]:
        sources: List[Dict[str, Any]] = []

        if self.cfg.arxiv_ids:
            logger.info("Fetching arXiv metadata and sources...")

            def _load_arxiv(arxiv_id: str) -> dict:
                meta = self.arxiv_client.get_metadata(arxiv_id)
                title = meta.get("title", arxiv_id)
                abstract = meta.get("abstract", "")
                url = meta.get("url", "")

                logger.info("Downloading and extracting arXiv source: %s", arxiv_id)
                arxiv_work = self.cfg.work_dir / f"arxiv_{arxiv_id}"
                src_dir = self.arxiv_client.download_source(arxiv_id, arxiv_work)
                main_tex = find_main_tex_file(src_dir)
                flat = flatten_tex(main_tex, max_files=120)
                paper_text = build_paper_text(flat, max_chars=None)

                logger.info("Main TeX file: %s", main_tex)
                logger.info("paper_text chars: %s", len(paper_text))
                if len(paper_text) <= 500:
                    raise RuntimeError("paper_text too small; main tex likely wrong.")

                return {
                    "type": "arxiv",
                    "id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "url": url,
                    "text": paper_text,
                    "images": [],
                }

            max_workers = min(self.cfg.max_llm_workers, len(self.cfg.arxiv_ids))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_load_arxiv, a): a for a in self.cfg.arxiv_ids}
                for fut in as_completed(futures):
                    sources.append(fut.result())

        if self.cfg.pdf_paths:
            def _load_pdf(pdf_path: Path) -> dict:
                logger.info("Reading local PDF: %s", pdf_path)
                pdf_work = self.cfg.work_dir / f"pdf_{pdf_path.stem}"
                pdf_data = extract_pdf_content(pdf_path, pdf_work)
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

                return {
                    "type": "pdf",
                    "id": str(pdf_path),
                    "title": pdf_data["title"],
                    "abstract": "",
                    "url": str(pdf_path),
                    "text": paper_text,
                    "images": pdf_data["images"],
                }

            max_workers = min(self.cfg.max_llm_workers, len(self.cfg.pdf_paths))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_load_pdf, p): p for p in self.cfg.pdf_paths}
                for fut in as_completed(futures):
                    sources.append(fut.result())

        if self.cfg.pdf_paths:
            print("\nPDF sources:")
            for s in sources:
                if s["type"] == "pdf":
                    print(f"- {s['title']} ({s['id']})")
            print("")
        if self.cfg.arxiv_ids:
            print("arXiv sources:")
            for s in sources:
                if s["type"] == "arxiv":
                    print(f"- {s['title']} ({s['id']})")
            print("")

        if len(sources) == 1:
            meta = {"title": sources[0]["title"], "abstract": sources[0].get("abstract", "")}
        else:
            meta = {"title": "Multiple Sources", "abstract": "Multiple documents provided."}

        if self.cfg.arxiv_ids and not self.cfg.pdf_paths and len(self.cfg.arxiv_ids) == 1:
            source_label = f"arXiv:{self.cfg.arxiv_ids[0]}"
        elif self.cfg.arxiv_ids and not self.cfg.pdf_paths:
            source_label = f"arXiv ({len(self.cfg.arxiv_ids)})"
        elif self.cfg.pdf_paths and not self.cfg.arxiv_ids:
            source_label = f"Local PDFs ({len(self.cfg.pdf_paths)})"
        else:
            source_label = f"Mixed sources ({len(sources)})"

        sources_block_lines = []
        for i, s in enumerate(sources, 1):
            src_tag = "arXiv" if s["type"] == "arxiv" else "PDF"
            sources_block_lines.append(f"{i}. [{src_tag}] {s['title']} ({s['id']})")
        sources_block = "\n".join(sources_block_lines)

        blocks = []
        for s in sources:
            blocks.append(f"[SOURCE: {s['title']}]\n{s['text']}")
        paper_text = "\n\n".join(blocks)

        self._checkpoint("Sources collected")
        global_feedback = self._prompt_feedback("Global feedback")

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

        citations_base = []
        for s in sources:
            if s["type"] == "arxiv":
                if s.get("url"):
                    citations_base.append(f"{s['title']} - {s['url']}")
                else:
                    citations_base.append(f"arXiv:{s['id']}")
            else:
                citations_base.append(f"{s['title']} - {s['id']}")
        if web_sources:
            citations_base.extend([f"{s['title']} - {s['url']}" for s in web_sources])

        chunks = self.chunk_text(paper_text, 1500)
        N = min(self.cfg.max_summary_chunks, len(chunks))
        sums: List[str] = []

        logger.info("Summarizing paper (%s chunks)...", N)
        prev_summary_preview = ""
        if N > 1:
            self._checkpoint("Summarize (parallel)", 0, N)
            max_workers = min(self.cfg.max_llm_workers, N)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}
                for i in range(1, N + 1):
                    futures[
                        pool.submit(
                            self.summarize_chunk,
                            i,
                            chunks[i - 1],
                            meta,
                            (self.cfg.user_query + "\n" + global_feedback).strip(),
                            web_context,
                            sources_block,
                        )
                    ] = i
                results: dict[int, str] = {}
                with tqdm(
                    total=N,
                    desc="Summarize",
                    unit="chunk",
                    ncols=TQDM_NCOLS,
                    dynamic_ncols=False,
                ) as bar:
                    for fut in as_completed(futures):
                        i = futures[fut]
                        s = fut.result()
                        results[i] = s
                        prev_summary_preview = self._preview_text(s, max_len=50)
                        bar.set_postfix_str(f"chunk: {i}/{N} | prev: {prev_summary_preview}")
                        bar.update(1)
                for i in range(1, N + 1):
                    if i in results:
                        sums.append(results[i])
        else:
            s = self.summarize_chunk(
                1,
                chunks[0],
                meta,
                (self.cfg.user_query + "\n" + global_feedback).strip(),
                web_context,
                sources_block,
            )
            sums.append(s)

        merged_summary = "\n\n".join(sums)

        logger.info("Generating slide titles (%s)...", self.cfg.slide_count)
        self._checkpoint("Slide titles")
        titles_obj = self.get_slide_titles(
            meta,
            merged_summary,
            user_query=(self.cfg.user_query + "\n" + global_feedback).strip(),
            web_context=web_context,
            sources_block=sources_block,
            source_label=source_label,
        )
        self._print_section(
            "Slide titles",
            [t for t in titles_obj.get("slide_titles", [])],
        )
        titles_feedback = self._prompt_feedback("Slide titles feedback")
        if titles_feedback:
            revised = self.regenerate_titles_with_feedback(
                meta,
                merged_summary,
                prev_titles=titles_obj.get("slide_titles", []),
                feedback=titles_feedback,
                user_query=(self.cfg.user_query + "\n" + global_feedback).strip(),
                web_context=web_context,
                sources_block=sources_block,
                source_label=source_label,
            )
            titles_obj = revised
            self._print_section(
                "Revised slide titles",
                [t for t in titles_obj.get("slide_titles", [])],
            )
        self._save_progress(
            {
                "stage": "titles",
                "meta": meta,
                "merged_summary": merged_summary,
                "titles_obj": titles_obj,
                "web_context": web_context,
                "sources_block": sources_block,
                "source_label": source_label,
                "citations": citations_base,
                "slides": [],
                "work_dir": str(self.cfg.work_dir),
                "out_dir": str(self.cfg.out_dir),
            }
        )

        if self.cfg.titles_only:
            slides = [
                {
                    "title": t,
                    "bullets": [],
                    "speaker_notes": "",
                    "figure_suggestions": [],
                    "generated_images": [],
                }
                for t in titles_obj.get("slide_titles", [])
            ]
            outline_dict = {
                "deck_title": titles_obj.get("deck_title", meta.get("title", "Presentation")),
                "arxiv_id": source_label,
                "slides": slides,
                "citations": citations_base,
            }
            outline = DeckOutline.model_validate(outline_dict)
            return (
                outline,
                meta,
                merged_summary,
                titles_obj,
                web_context,
                web_sources,
                sources_block,
                source_label,
                citations_base,
            )

        logger.info("Generating slides (%s)...", self.cfg.slide_count)
        slides = []
        slide_feedback = self._prompt_feedback("Slide content feedback")
        for idx, title in tqdm(
            list(enumerate(titles_obj["slide_titles"], 1)),
            desc="Slides",
            unit="slide",
            ncols=TQDM_NCOLS,
            dynamic_ncols=False,
        ):
            self._checkpoint("Slides", idx, self.cfg.slide_count)
            slides.append(
                self.make_slide(
                    meta,
                    title,
                    merged_summary,
                    idx,
                    feedback=slide_feedback or "",
                    include_speaker_notes=self.cfg.include_speaker_notes,
                    user_query=(self.cfg.user_query + "\n" + global_feedback).strip(),
                    web_context=web_context,
                    sources_block=sources_block,
                )
            )
            self._save_progress(
                {
                    "stage": "slides",
                    "meta": meta,
                    "merged_summary": merged_summary,
                    "titles_obj": titles_obj,
                    "web_context": web_context,
                    "sources_block": sources_block,
                    "source_label": source_label,
                    "citations": citations_base,
                    "slides": slides,
                    "work_dir": str(self.cfg.work_dir),
                    "out_dir": str(self.cfg.out_dir),
                }
            )

        citations = list(citations_base)

        outline_dict = {
            "deck_title": titles_obj["deck_title"],
            "arxiv_id": source_label,
            "slides": slides,
            "citations": citations,
        }
        outline = DeckOutline.model_validate(outline_dict)
        return (
            outline,
            meta,
            merged_summary,
            titles_obj,
            web_context,
            web_sources,
            sources_block,
            source_label,
            citations,
        )

    def regenerate_titles_with_feedback(
        self,
        meta: dict,
        merged_summary: str,
        prev_titles: List[str],
        feedback: str,
        user_query: str = "",
        web_context: str = "",
        sources_block: str = "",
        source_label: str = "",
    ) -> dict:
        summary = re.sub(r"\s+", " ", merged_summary).strip()[:1200]
        prev = "\n".join([f"{i+1}. {t}" for i, t in enumerate(prev_titles)])
        query_block = f"\nUser query:\n{user_query}\n" if user_query else ""
        web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""
        sources_block = f"\nSources:\n{sources_block}\n" if sources_block else ""

        prompt = f"""
Return ONLY JSON.

Schema:
{{
  "deck_title": "string",
  "arxiv_id": "{source_label}",
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
{query_block}{web_block}{sources_block}
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
        if not self.cfg.arxiv_ids and not self.cfg.pdf_paths:
            raise ValueError("At least one arXiv ID or PDF path must be provided")
        for p in self.cfg.pdf_paths:
            if not p.exists():
                raise FileNotFoundError(f"PDF not found: {p}")

        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        if not self.cfg.skip_llm_sanity:
            x = safe_invoke(logger, self.llm, "Reply with exactly: OK", debug=self.cfg.verbose)
            logger.info("LLM sanity: %r", x[:50])
            if "OK" not in x:
                raise RuntimeError("LLM sanity check failed. Ensure your NVIDIA_API_KEY and model are valid.")

    def _load_progress(self) -> Optional[dict]:
        if not self.cfg.resume_path:
            return None
        out_dir = self.cfg.resume_path
        if out_dir.name != "outputs":
            out_dir = out_dir / "outputs"
        path = _progress_path(out_dir)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read progress.json")
            return None

    def prepare_topic_sources(self) -> None:
        """Expand a topic, search the web, and collect sources for a topic-only run."""
        if not self.cfg.topic:
            return

        prompt = f"""
You are preparing a research presentation. Expand the topic into a focused query
with key sub-questions and keywords. Return a single paragraph query.

Topic: {self.cfg.topic}
""".strip()
        expanded = safe_invoke(logger, self.llm, prompt, retries=6).strip()
        if expanded:
            self.cfg.user_query = expanded
        else:
            expanded = self.cfg.topic
            self.cfg.user_query = self.cfg.topic

        logger.info("Topic expanded query: %s", expanded)
        results = search_web(expanded, max_results=self.cfg.max_web_results)
        if self.cfg.topic_scholarly_only:
            allowed_domains = {
                "arxiv.org",
                "openaccess.thecvf.com",
                "cvpr.thecvf.com",
                "icml.cc",
                "proceedings.mlr.press",
                "neurips.cc",
                "proceedings.neurips.cc",
                "scholar.google.com",
            }
            filtered = []
            for r in results:
                url = r.get("url", "")
                try:
                    from urllib.parse import urlparse

                    host = urlparse(url).netloc.lower()
                except Exception:
                    host = ""
                if any(host == d or host.endswith("." + d) for d in allowed_domains):
                    filtered.append(r)
            results = filtered
        if not results:
            logger.warning("No web results found for topic search.")
            return

        arxiv_ids = list(self.cfg.arxiv_ids)
        pdf_urls = []
        for r in results:
            url = r.get("url", "")
            if "arxiv.org/abs/" in url or "arxiv.org/pdf/" in url:
                try:
                    arxiv_ids.append(extract_arxiv_id(url))
                except Exception:
                    pass
            elif url.lower().endswith(".pdf"):
                pdf_urls.append(url)

        # Deduplicate
        arxiv_ids = list(dict.fromkeys(arxiv_ids))
        pdf_urls = list(dict.fromkeys(pdf_urls))[: self.cfg.max_web_pdfs]

        if arxiv_ids:
            self.cfg.arxiv_ids = arxiv_ids
        if pdf_urls:
            download_dir = self.cfg.work_dir / "web_pdfs"
            download_dir.mkdir(parents=True, exist_ok=True)
            def _download_pdf(u: str) -> Optional[Path]:
                try:
                    name = Path(u.split("?")[0]).name or "paper.pdf"
                    if not name.lower().endswith(".pdf"):
                        name = name + ".pdf"
                    target = download_dir / name
                    if target.exists() and target.stat().st_size > 0:
                        return target
                    import requests

                    r = requests.get(u, stream=True, timeout=60)
                    r.raise_for_status()
                    with target.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)
                    return target
                except Exception:
                    logger.exception("Failed to download PDF from %s", u)
                    return None

            max_workers = min(self.cfg.max_llm_workers, len(pdf_urls))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_download_pdf, u): u for u in pdf_urls}
                for fut in as_completed(futures):
                    p = fut.result()
                    if p:
                        self.cfg.pdf_paths.append(p)

    @staticmethod
    def print_outline(outline: DeckOutline) -> None:
        width = 96
        print("\n" + "=" * width)
        print(f"DECK: {outline.deck_title}")
        print(f"SOURCES: {outline.arxiv_id}")
        print("=" * width)
        for i, sl in enumerate(outline.slides, 1):
            print(f"\n{i:02d}. {sl.title}")
            for b in sl.bullets:
                wrapped = textwrap.fill(
                    b,
                    width=width - 6,
                    initial_indent="   - ",
                    subsequent_indent="     ",
                )
                print(wrapped)
            if sl.figure_suggestions:
                figs = "; ".join(sl.figure_suggestions)
                print(textwrap.fill(f"[figs] {figs}", width=width, initial_indent="   ", subsequent_indent="   "))
            if sl.speaker_notes.strip():
                note = sl.speaker_notes[:220] + ("..." if len(sl.speaker_notes) > 220 else "")
                print(textwrap.fill(f"[notes] {note}", width=width, initial_indent="   ", subsequent_indent="   "))
        print("\n" + "=" * width)

    def _select_flowchart_indices(self, outline: DeckOutline) -> List[int]:
        keywords = [
            "method",
            "approach",
            "architecture",
            "pipeline",
            "framework",
            "algorithm",
            "model",
            "training",
            "inference",
            "system",
            "procedure",
            "workflow",
            "module",
        ]
        scored = []
        for i, sl in enumerate(outline.slides):
            title = (sl.title or "").lower()
            score = sum(1 for k in keywords if k in title)
            if sl.flowchart and sl.flowchart.steps:
                score += 2
            scored.append((score, i))
        scored.sort(reverse=True)
        target = min(
            len(outline.slides),
            max(self.cfg.min_flowcharts, min(self.cfg.max_flowcharts, len(outline.slides))),
        )
        chosen = [i for _score, i in scored[:target]]
        return chosen

    def _generate_flowchart_steps(self, slide: dict, topic_hint: str = "") -> dict:
        max_steps = max(6, self.cfg.flowchart_depth)
        prompt = f"""
You are an expert researcher designing a diagram that improves understanding of the entire presentation.
Create the BEST flowchart for the slide below. You must decide:
- the number of steps (4 to {max_steps})
- the structure: linear | branch | cycle
Choose what best captures the underlying mechanism and decision flow.

Requirements:
- Output ONLY JSON in this schema:
  {{ "steps": ["string", ...], "structure": "linear|branch|cycle", "caption": "string" }}
- Steps must be concrete and technical, each 3-8 words.
- Prefer mechanism-level steps (compute, update, select, aggregate, infer).
- Avoid vague verbs like \"process\", \"handle\", \"stuff\".
- Use branch when there are alternate paths/conditions, cycle when iterative refinement or feedback loops exist.
- Caption should be short and specific to the module.

Slide title: {slide.get("title","")}
Bullets: {slide.get("bullets", [])}
Speaker notes: {slide.get("speaker_notes","")}
Topic hint: {topic_hint}
""".strip()
        raw = safe_invoke(logger, self.llm, prompt, retries=6)
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("flowchart JSON not dict")
        except Exception:
            fix = safe_invoke(
                logger,
                self.llm,
                "Return ONLY valid JSON for the schema. Fix this:\n" + raw[:1800],
                retries=6,
            )
            obj = json.loads(fix)
        obj.setdefault("steps", [])
        obj.setdefault("structure", "linear")
        obj.setdefault("caption", "")
        if not isinstance(obj["steps"], list):
            obj["steps"] = []
        # Clamp overly long outputs
        if len(obj["steps"]) > max_steps:
            obj["steps"] = obj["steps"][:max_steps]
        return obj

    def _render_flowcharts(self, outline: DeckOutline) -> None:
        flow_dir = self.cfg.out_dir / "flowcharts"
        flow_dir.mkdir(parents=True, exist_ok=True)

        indices = self._select_flowchart_indices(outline)
        if not indices:
            return

        def _build_one(i: int) -> tuple[int, str | None]:
            slide = outline.slides[i]
            fc = slide.flowchart.model_dump()
            if not fc.get("steps"):
                fc = self._generate_flowchart_steps(
                    {
                        "title": slide.title,
                        "bullets": slide.bullets,
                        "speaker_notes": slide.speaker_notes,
                    },
                    topic_hint=outline.deck_title,
                )
            steps = [str(s).strip() for s in fc.get("steps", []) if str(s).strip()]
            if len(steps) < 3:
                return i, None
            structure = fc.get("structure", self.cfg.flowchart_structure) or self.cfg.flowchart_structure
            dot = build_graphviz(steps, structure=structure)
            dot_path = flow_dir / f"slide_{i+1:02d}.dot"
            png_path = flow_dir / f"slide_{i+1:02d}.png"
            dot_path.write_text(dot, encoding="utf-8")
            try:
                render_graphviz(dot_path, png_path)
            except Exception:
                logger.exception("Failed to render flowchart for slide %s", i + 1)
                return i, None
            return i, str(png_path)

        max_workers = min(self.cfg.max_llm_workers, len(indices))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_build_one, i): i for i in indices}
            for fut in as_completed(futures):
                i, path = fut.result()
                if path:
                    outline.slides[i].flowchart_images.append(path)

    def build_outline_with_approval(self, max_rounds: int = 3) -> Tuple[DeckOutline, Dict[str, Any]]:
        progress = self._load_progress()
        if progress and progress.get("stage") in {"titles", "slides"}:
            meta = progress.get("meta", {"title": "Resume", "abstract": ""})
            merged_summary = progress.get("merged_summary", "")
            titles_obj = progress.get("titles_obj", {})
            web_context = progress.get("web_context", "")
            sources_block = progress.get("sources_block", "")
            source_label = progress.get("source_label", "Resume")
            citations_base = progress.get("citations", [])
            slides = progress.get("slides", [])
            logger.info("Resuming from progress.json with %s slides.", len(slides))

            # Continue generating remaining slides
            for idx, title in enumerate(titles_obj.get("slide_titles", []), 1):
                if idx <= len(slides):
                    continue
                self.outline_builder._checkpoint("Slides", idx, self.cfg.slide_count)
                slides.append(
                    self.outline_builder.make_slide(
                        meta,
                        title,
                        merged_summary,
                        idx,
                        include_speaker_notes=self.cfg.include_speaker_notes,
                        user_query=self.cfg.user_query,
                        web_context=web_context,
                        sources_block=sources_block,
                    )
                )
                self.outline_builder._save_progress(
                    {
                        "stage": "slides",
                        "meta": meta,
                        "merged_summary": merged_summary,
                        "titles_obj": titles_obj,
                        "web_context": web_context,
                        "sources_block": sources_block,
                        "source_label": source_label,
                        "citations": citations_base,
                        "slides": slides,
                        "work_dir": str(self.cfg.work_dir),
                        "out_dir": str(self.cfg.out_dir),
                    }
                )

            outline_dict = {
                "deck_title": titles_obj.get("deck_title", "Resume"),
                "arxiv_id": source_label,
                "slides": slides,
                "citations": citations_base,
            }
            outline = DeckOutline.model_validate(outline_dict)
        else:
            (
                outline,
                meta,
                merged_summary,
                titles_obj,
                web_context,
                web_sources,
                sources_block,
                source_label,
                citations_base,
            ) = self.outline_builder.build_outline_once()
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
                sources_block=sources_block,
                source_label=source_label,
            )
            titles_obj = revised

            slides = [None] * len(titles_obj["slide_titles"])

            def _make_one(i: int, t: str) -> tuple[int, dict]:
                s = self.outline_builder.make_slide(
                    meta,
                    t,
                    merged_summary,
                    i,
                    feedback=feedback,
                    include_speaker_notes=self.cfg.include_speaker_notes,
                    user_query=self.cfg.user_query,
                    web_context=web_context,
                    sources_block=sources_block,
                )
                return i, s

            max_workers = min(self.cfg.max_llm_workers, len(titles_obj["slide_titles"]))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_make_one, idx, title): idx
                    for idx, title in enumerate(titles_obj["slide_titles"], 1)
                }
                with tqdm(
                    total=len(titles_obj["slide_titles"]),
                    desc="Slides",
                    unit="slide",
                    ncols=TQDM_NCOLS,
                    dynamic_ncols=False,
                ) as bar:
                    for fut in as_completed(futures):
                        idx, slide = fut.result()
                        slides[idx - 1] = slide
                        bar.update(1)

            citations = list(citations_base)

            outline_dict = {
                "deck_title": titles_obj["deck_title"],
                "arxiv_id": source_label,
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

        if self.cfg.interactive:
            ans = input("[Render] Press Enter to render outputs or type 'q' to quit: ").strip().lower()
            if ans in {"q", "quit", "exit"}:
                raise RuntimeError("Aborted by user.")

        if self.cfg.generate_flowcharts:
            try:
                self._render_flowcharts(outline)
            except Exception:
                logger.exception("Flowchart generation failed; continuing without flowcharts.")

        if self.cfg.use_figures and (self.cfg.pdf_paths or len(self.cfg.arxiv_ids) != 1):
            logger.warning("Figure insertion requires exactly one arXiv source; continuing without figures.")
            tex_path, pdf_path = self.renderer.render(outline, self.cfg.out_dir)
        elif self.cfg.use_figures:
            tex_path, pdf_path = self.renderer.render_with_figs(
                self.llm,
                outline,
                self.cfg.arxiv_ids[0],
                self.cfg.work_dir,
                self.cfg.out_dir,
                self.figure_planner,
            )
        else:
            tex_path, pdf_path = self.renderer.render(outline, self.cfg.out_dir)

        return outline, tex_path, pdf_path
