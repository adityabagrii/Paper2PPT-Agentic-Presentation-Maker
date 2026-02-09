from __future__ import annotations

import json
import hashlib
import logging
import re
import textwrap
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from .llm import safe_invoke
    from .models import DeckOutline
    from .pdf_utils import extract_pdf_content
    from .web_utils import search_web
    from .tex_utils import build_paper_text, find_main_tex_file, flatten_tex
    from .memory_utils import get_cached_summary, put_cached_summary
except Exception:
    from llm import safe_invoke
    from models import DeckOutline
    from pdf_utils import extract_pdf_content
    from web_utils import search_web
    from tex_utils import build_paper_text, find_main_tex_file, flatten_tex
    from memory_utils import get_cached_summary, put_cached_summary

try:
    from .pipeline_arxiv import ArxivClient
    from .pipeline_common import RunConfig, logger, TQDM_NCOLS, _progress_path
    from .arxiv_utils import get_arxiv_pdf_url
    from .arxiv_utils import extract_arxiv_id
except Exception:
    from pipeline_arxiv import ArxivClient
    from pipeline_common import RunConfig, logger, TQDM_NCOLS, _progress_path
    from arxiv_utils import get_arxiv_pdf_url
    from arxiv_utils import extract_arxiv_id

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class OutlineBuilder:
    def __init__(self, llm, cfg: RunConfig, arxiv_client: ArxivClient) -> None:
        """Initialize.
        
        Args:
            llm (Any):
            cfg (RunConfig):
            arxiv_client (ArxivClient):
        
        Returns:
            None:
        """
        self.llm = llm
        self.cfg = cfg
        self.arxiv_client = arxiv_client
        self.diagram_plan: List[dict] = []

    def _select_relevant_chunks(self, chunks: List[str], global_feedback: str = "") -> List[str]:
        """Select top chunks aligned to the topic/query for summarization."""
        if not chunks:
            return chunks
        N = min(len(chunks), self.cfg.max_summary_chunks)
        if len(chunks) <= N:
            return chunks[:N]

        query_text = " ".join(
            [
                self.cfg.topic or "",
                self.cfg.user_query or "",
                global_feedback or "",
            ]
        ).strip()
        scored: List[Tuple[float, int, str]] = []
        if SentenceTransformer is not None and query_text:
            try:
                logger.info("Loading sentence piece transformer")
                # Silence HF hub/transformers HTTP and progress noise.
                import os
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                try:
                    from huggingface_hub import logging as hf_logging
                    hf_logging.set_verbosity_error()
                except Exception:
                    pass
                try:
                    from transformers import logging as tf_logging
                    tf_logging.set_verbosity_error()
                except Exception:
                    pass
                model = SentenceTransformer("all-MiniLM-L6-v2")
                q_emb = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0]
                c_embs = model.encode(chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
                for idx, emb in enumerate(c_embs):
                    score = float((q_emb * emb).sum())
                    scored.append((score, idx, chunks[idx]))
            except Exception:
                scored = []
        if not scored:
            query_text_l = query_text.lower()
            keywords = re.findall(r"[a-z0-9]{3,}", query_text_l)
            keyset = set(keywords)
            for idx, ch in enumerate(chunks):
                words = re.findall(r"[a-z0-9]{3,}", ch.lower())
                score = sum(1 for w in words if w in keyset)
                scored.append((float(score), idx, ch))
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        selected = [c for _, _, c in scored[:N]]
        if selected:
            logger.info("Shortlisted chunk examples:")
            for i, ch in enumerate(selected[:3], 1):
                preview = re.sub(r"\s+", " ", ch.strip())[:220]
                logger.info("  %s. %s", i, preview)
        return selected

    def _checkpoint(self, label: str, idx: int | None = None, total: int | None = None) -> None:
        """Function checkpoint.
        
        Args:
            label (str):
            idx (int | None):
            total (int | None):
        
        Returns:
            None:
        """
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
        """Function prompt feedback.
        
        Args:
            label (str):
        
        Returns:
            str:
        """
        if not self.cfg.interactive:
            return ""
        ans = input(f"[{label}] Provide guidance (or press Enter to skip): ").strip()
        return ans

    @staticmethod
    def _is_claim_bullet(text: str) -> bool:
        """Check if a bullet contains a performance claim.
        
        Args:
            text (str):
        
        Returns:
            bool:
        """
        t = text.lower()
        claim_markers = [
            "improve", "outperform", "better", "worse", "increase", "decrease", "reduce",
            "higher", "lower", "faster", "slower", "accuracy", "mAP", "f1", "precision",
            "recall", "sota", "state-of-the-art", "gain", "drop", "boost", "achieve",
            "surpass", "significant", "statistically",
        ]
        return any(k in t for k in claim_markers)

    @staticmethod
    def _has_evidence_tag(text: str) -> bool:
        """Check if a bullet contains an evidence tag.
        
        Args:
            text (str):
        
        Returns:
            bool:
        """
        return bool(re.search(r"(source:|evidence:|https?://|Slide\\s+\\d+)", text, re.I))

    def _flag_ungrounded_claims(self, slide: dict, experiment_refs: List[str]) -> dict:
        """Flag or annotate claims without evidence.
        
        Args:
            slide (dict):
            experiment_refs (List[str]):
        
        Returns:
            dict:
        """
        bullets = []
        fallback_evidence = ""
        if experiment_refs:
            fallback_evidence = f"(evidence: {experiment_refs[0]})"
        for b in slide.get("bullets", []):
            if self._is_claim_bullet(b) and not self._has_evidence_tag(b):
                logger.warning("Ungrounded claim detected; flagging for evidence.")
                b = b.rstrip()
                if fallback_evidence:
                    b += f" {fallback_evidence}"
                else:
                    b += " (evidence: source TBD)"
                b += " [NEEDS EVIDENCE]"
            bullets.append(b)
        slide["bullets"] = bullets
        return slide

    def _ensure_baseline_framing(self, slide_title: str, slide: dict) -> dict:
        """Ensure baseline framing bullets on experiment/result slides.
        
        Args:
            slide_title (str):
            slide (dict):
        
        Returns:
            dict:
        """
        if not re.search(r"(experiment|result|evaluation|benchmark|ablation|comparison)", slide_title, re.I):
            return slide
        bullets = list(slide.get("bullets", []))
        need_a = "Why this baseline?"
        need_b = "What does it control for?"
        if not any(need_a.lower() in b.lower() for b in bullets):
            bullets.append(need_a)
        if not any(need_b.lower() in b.lower() for b in bullets):
            bullets.append(need_b)
        if len(bullets) > self.cfg.bullets_per_slide:
            bullets = bullets[: self.cfg.bullets_per_slide]
        slide["bullets"] = bullets
        return slide

    def _generate_quant_results_table(
        self,
        merged_summary: str,
        sources_block: str,
        web_context: str = "",
    ) -> dict:
        """Generate a quantitative results table from sources.
        
        Args:
            merged_summary (str):
            sources_block (str):
            web_context (str):
        
        Returns:
            dict:
        """
        summary = re.sub(r"\s+", " ", merged_summary).strip()[:1400]
        web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""
        prompt = f"""
Return ONLY JSON.

Schema:
{{
  "title": "Quantitative Results",
  "columns": ["Method", "Dataset", "Metric", "Score"],
  "rows": [["method", "dataset", "metric", "value"], ...]
}}

Rules:
- Extract concrete numbers from the sources when available.
- Use 6-12 rows.
- If a number is missing, write "n/a" instead of guessing.
- Use short method names and dataset names.

Sources:
{sources_block}

Summary: {summary}
{web_block}
""".strip()
        raw = safe_invoke(logger, self.llm, prompt, retries=6).strip()
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
            return {"title": "Quantitative Results", "columns": [], "rows": []}
        try:
            obj = json.loads(js)
        except Exception:
            return {"title": "Quantitative Results", "columns": [], "rows": []}
        cols = obj.get("columns", [])
        rows = obj.get("rows", [])
        if not isinstance(cols, list) or not isinstance(rows, list):
            return {"title": "Quantitative Results", "columns": [], "rows": []}
        return {
            "title": str(obj.get("title", "Quantitative Results")),
            "columns": [str(c) for c in cols],
            "rows": [[str(x) for x in r] for r in rows if isinstance(r, (list, tuple))],
        }

    def _save_progress(self, state: dict) -> None:
        """Save progress.
        
        Args:
            state (dict):
        
        Returns:
            None:
        """
        try:
            path = _progress_path(self.cfg.out_dir)
            with path.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to write progress.json")

    @staticmethod
    def _print_section(title: str, lines: List[str]) -> None:
        """Print section.
        
        Args:
            title (str):
            lines (List[str]):
        
        Returns:
            None:
        """
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
        """Function chunk text.
        
        Args:
            s (str):
            chunk_chars (int):
        
        Returns:
            List[str]:
        """
        s = s.strip()
        return [s[i : i + chunk_chars] for i in range(0, len(s), chunk_chars)]

    @staticmethod
    def _experiment_slide_refs(titles: List[str]) -> List[str]:
        """Collect experiment/result slide references.
        
        Args:
            titles (List[str]):
        
        Returns:
            List[str]:
        """
        refs = []
        for i, t in enumerate(titles, 1):
            if re.search(r"(experiment|result|evaluation|benchmark|ablation|comparison)", t, re.I):
                refs.append(f"Slide {i} - {t}")
        return refs

    def _ensure_comparison_titles(self, titles: List[str]) -> List[str]:
        """Ensure comparison titles exist when auto comparisons are enabled.
        
        Args:
            titles (List[str]):
        
        Returns:
            List[str]:
        """
        if not self.cfg.auto_comparisons:
            return titles
        want = [
            "Full Video vs Key Frames: Trade-offs",
            "Uniform Sampling vs Learned Selection",
        ]
        titles_lower = [t.lower() for t in titles]
        missing = [w for w in want if w.lower() not in " ".join(titles_lower)]
        if not missing:
            return titles
        out = list(titles)
        # Replace from the end to keep the deck length stable
        for j, w in enumerate(reversed(missing), 1):
            if len(out) >= j:
                out[-j] = w
        return out

    def _ensure_pause_question_titles(self, titles: List[str], target: int = 2) -> List[str]:
        """Ensure pause question titles exist in teaching mode.
        
        Args:
            titles (List[str]):
            target (int):
        
        Returns:
            List[str]:
        """
        if not self.cfg.teaching_mode:
            return titles
        out = list(titles)
        existing = [t for t in out if "pause question" in t.lower()]
        if len(existing) >= target:
            return out
        # Replace mid and near-end slides to keep length stable
        candidates = []
        if out:
            candidates.append(len(out) // 2)
        if len(out) > 3:
            candidates.append(len(out) - 2)
        labels = [
            "Pause Question: Check Understanding",
            "Pause Question: Apply the Idea",
        ]
        idx = 0
        for c in candidates:
            if len(existing) >= target:
                break
            if 0 <= c < len(out) and "pause question" not in out[c].lower():
                out[c] = labels[idx % len(labels)]
                idx += 1
                existing.append(out[c])
        if len(existing) < target and out:
            out[-1] = labels[min(idx, len(labels) - 1)]
        return out

    @staticmethod
    def _preview_text(s: str, max_len: int = 60) -> str:
        """Function preview text.
        
        Args:
            s (str):
            max_len (int):
        
        Returns:
            str:
        """
        s = re.sub(r"\s+", " ", (s or "").strip())
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

    @staticmethod
    def try_extract_json(text: str) -> Optional[str]:
        """Function try extract json.
        
        Args:
            text (str):
        
        Returns:
            Optional[str]:
        """
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
        """Summarize chunk.
        
        Args:
            i (int):
            chunk (str):
            meta (dict):
            user_query (str):
            web_context (str):
            sources_block (str):
        
        Returns:
            str:
        """
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

    def summarize_text(
        self,
        paper_text: str,
        meta: dict,
        global_feedback: str,
        web_context: str = "",
        sources_block: str = "",
    ) -> str:
        """Summarize long text into a merged summary.
        
        Args:
            paper_text (str):
            meta (dict):
            global_feedback (str):
            web_context (str):
            sources_block (str):
        
        Returns:
            str:
        """
        cache_key = ""
        if self.cfg.cache_summary:
            key_material = "\n".join(
                [
                    meta.get("title", ""),
                    meta.get("abstract", ""),
                    paper_text[:100000],
                    (self.cfg.user_query + "\n" + global_feedback).strip(),
                    web_context[:4000],
                    sources_block[:4000],
                ]
            )
            cache_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
            cached = get_cached_summary(cache_key, max_age_seconds=3 * 60 * 60)
            if cached:
                logger.info("Using cached summary.")
                return cached

        chunks = self.chunk_text(paper_text, 1500)
        chunks = self._select_relevant_chunks(chunks, global_feedback=global_feedback)
        N = len(chunks)
        sums = []
        prev_summary_preview = "..."
        if self.cfg.max_llm_workers > 1 and N > 1:
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
            with tqdm(
                range(1, N + 1),
                desc="Summarize",
                unit="chunk",
                ncols=TQDM_NCOLS,
                dynamic_ncols=False,
            ) as bar:
                for i in bar:
                    self._checkpoint("Summarize", i, N)
                    chunk_preview = self._preview_text(chunks[i - 1], max_len=50)
                    bar.set_postfix_str(f"chunk: {chunk_preview} | prev: {prev_summary_preview}")
                    s = self.summarize_chunk(
                        i,
                        chunks[i - 1],
                        meta,
                        (self.cfg.user_query + "\n" + global_feedback).strip(),
                        web_context,
                        sources_block,
                    )
                    sums.append(s)
                    prev_summary_preview = self._preview_text(s, max_len=50)
        merged = "\n\n".join(sums)
        if self.cfg.cache_summary and cache_key:
            put_cached_summary(cache_key, merged)
        return merged

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
        """Get slide titles.
        
        Args:
            meta (dict):
            merged_summary (str):
            feedback (str):
            user_query (str):
            web_context (str):
            sources_block (str):
            source_label (str):
        
        Returns:
            dict:
        """
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
        comparison_rule = ""
        if self.cfg.auto_comparisons:
            comparison_rule = (
                "- Include explicit comparison slides (e.g., full video vs key frames; uniform sampling vs learned selection)\n"
            )
        teaching_rule = ""
        if self.cfg.teaching_mode:
            teaching_rule = (
                "- Teaching mode: prioritize intuition and examples over equations.\n"
                "- Include 1-2 slides labeled 'Pause Question: ...' for student reflection.\n"
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
- Cover the full narrative arc from intro to conclusion. The sequence must clearly
  progress: context/motivation → problem statement → prior work/baselines →
  core idea → method details → experiments/setup → quantitative results →
  qualitative analysis → limitations/risks → conclusions/future work.
- Avoid vague, open-ended titles. Each title should be specific and scoped.
- No extra keys
- Deck title must reflect the user query and the source titles when provided
{query_rule}
{comparison_rule}
{teaching_rule}

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

        obj = None
        for attempt in range(1, 4):
            try:
                obj = json.loads(js)
                break
            except Exception:
                logger.warning("Slide titles JSON parse failed (attempt %s/3).", attempt)
                fix = safe_invoke(
                    logger,
                    self.llm,
                    "Return ONLY valid JSON for the schema. Fix this:\n" + js[:1800],
                    retries=6,
                )
                js = self.try_extract_json(fix) or fix
        if obj is None:
            logger.error("Slide titles JSON parse failed after retries; using fallback titles.")
            obj = {
                "deck_title": meta.get("title", "Presentation"),
                "arxiv_id": source_label,
                "slide_titles": [f"Slide {i+1}" for i in range(self.cfg.slide_count)],
            }
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
        if self.cfg.teaching_mode:
            obj["slide_titles"] = self._ensure_pause_question_titles(obj.get("slide_titles", []))
        return obj

    def propose_diagram_plan(
        self,
        titles: List[str],
        merged_summary: str,
        user_query: str = "",
        web_context: str = "",
        sources_block: str = "",
    ) -> List[dict]:
        """Propose diagram plan.
        
        Args:
            titles (List[str]):
            merged_summary (str):
            user_query (str):
            web_context (str):
            sources_block (str):
        
        Returns:
            List[dict]:
        """
        summary = re.sub(r"\s+", " ", merged_summary).strip()[:1200]
        query_block = f"\nUser query:\n{user_query}\n" if user_query else ""
        web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""
        sources_block = f"\nSources:\n{sources_block}\n" if sources_block else ""

        prompt = f"""
You are designing diagrams that carry core information for the deck.
Decide why each diagram is needed (intent) and specify a concrete graph spec.
Return ONLY JSON.

Schema:
{{
  "diagrams": [
    {{
      "slide_index": 1,
      "intent": "process|comparison|abstraction",
      "type": "comparison|taxonomy|pipeline|dag|sequence|block",
      "caption": "string",
      "priority": 1,
      "nodes": ["string", "..."],
      "edges": [["A","B","label"], ["A","C","label"]]
    }}
  ]
}}

Rules:
- Provide 5 to 8 diagrams total.
- Each diagram must be non-linear (not just a single chain).
- Include at least one comparison diagram and one process/pipeline diagram.
- Prefer diagrams that replace text: problem framing, method pipeline, comparisons, ablations.
- Use 6-10 nodes per diagram; edges must reference existing nodes.
- Target slide_index that best fits the diagram.

Slide titles:
{titles}

Summary: {summary}
{query_block}{web_block}{sources_block}
""".strip()

        raw = safe_invoke(logger, self.llm, prompt, retries=6).strip()
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
            logger.warning("Diagram plan JSON extraction failed; skipping.")
            return []
        try:
            obj = json.loads(js)
        except Exception:
            logger.warning("Diagram plan JSON parse failed; skipping.")
            return []
        diagrams = obj.get("diagrams", [])
        if not isinstance(diagrams, list):
            return []
        cleaned = []
        for d in diagrams:
            if not isinstance(d, dict):
                continue
            idx = d.get("slide_index")
            if not isinstance(idx, int):
                continue
            nodes = d.get("nodes", [])
            edges = d.get("edges", [])
            if not isinstance(nodes, list) or len(nodes) < 3:
                continue
            if not isinstance(edges, list):
                edges = []
            cleaned.append(
                {
                    "slide_index": idx,
                    "intent": str(d.get("intent", "process")),
                    "type": str(d.get("type", "block")),
                    "caption": str(d.get("caption", "")).strip(),
                    "priority": int(d.get("priority", 3)) if str(d.get("priority", "")).isdigit() else 3,
                    "nodes": [str(n) for n in nodes],
                    "edges": [tuple(e) for e in edges if isinstance(e, (list, tuple)) and len(e) >= 2],
                }
            )
        if len(cleaned) < 5:
            fix = safe_invoke(
                logger,
                self.llm,
                "Return ONLY valid JSON for the schema. Provide 5-8 diagrams:\n" + js[:1800],
                retries=6,
            )
            fix_js = self.try_extract_json(fix)
            if fix_js:
                try:
                    obj2 = json.loads(fix_js)
                    more = obj2.get("diagrams", [])
                    if isinstance(more, list):
                        cleaned = []
                        for d in more:
                            if not isinstance(d, dict):
                                continue
                            idx = d.get("slide_index")
                            if not isinstance(idx, int):
                                continue
                            nodes = d.get("nodes", [])
                            edges = d.get("edges", [])
                            if not isinstance(nodes, list) or len(nodes) < 3:
                                continue
                            if not isinstance(edges, list):
                                edges = []
                            cleaned.append(
                                {
                                    "slide_index": idx,
                                    "intent": str(d.get("intent", "process")),
                                    "type": str(d.get("type", "block")),
                                    "caption": str(d.get("caption", "")).strip(),
                                    "priority": int(d.get("priority", 3)) if str(d.get("priority", "")).isdigit() else 3,
                                    "nodes": [str(n) for n in nodes],
                                    "edges": [tuple(e) for e in edges if isinstance(e, (list, tuple)) and len(e) >= 2],
                                }
                            )
                except Exception:
                    pass
        return cleaned

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
        experiment_refs: Optional[List[str]] = None,
    ) -> dict:
        """Function make slide.
        
        Args:
            meta (dict):
            slide_title (str):
            merged_summary (str):
            idx (int):
            feedback (str):
            include_speaker_notes (bool):
            user_query (str):
            web_context (str):
            sources_block (str):
        
        Returns:
            dict:
        """
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
        evidence_rule = ""
        experiment_hint = ""
        if self.cfg.require_evidence:
            evidence_rule = (
                "\n- Any performance/accuracy/comparison claim must include evidence tags. "
                "Use either '(source: URL)' or '(evidence: Slide N - Results/Experiments)'.\n"
            )
            if experiment_refs:
                experiment_hint = (
                    "\nExperiment slide references (for evidence tags):\n- "
                    + "\n- ".join(experiment_refs)
                    + "\n"
                )
        baseline_rule = ""
        if self.cfg.baseline_framing:
            if re.search(r"(experiment|result|evaluation|benchmark|ablation|comparison)", slide_title, re.I):
                baseline_rule = (
                    "\n- Include two bullets that explicitly answer: "
                    "'Why this baseline?' and 'What does it control for?'\n"
                )
        query_rule = (
            "\n- The slide content must answer the user query (not just summarize)\n"
            if user_query
            else ""
        )
        teaching_rule = ""
        pause_rule = ""
        if self.cfg.teaching_mode:
            teaching_rule = (
                "\n- Teaching mode: prefer intuition and concrete examples; avoid equations and heavy jargon.\n"
            )
            if "pause question" in (slide_title or "").lower():
                pause_rule = (
                    "\n- This is a pause question slide: bullets must be student questions (not answers).\n"
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
- Prefer different diagram structures across slides (linear/branch/cycle).
- If not suitable, set flowchart.steps to [] and caption to "".
- graphviz_diagram_ideas should mention other diagram types: comparison chart, dependency graph, DAG, hierarchy, decision tree, ablation map, problem-solution map.
- Focus on visually depicting both the problem statement and the solution; diagrams should carry essential information.
{source_rule}
{evidence_rule}
{baseline_rule}
{query_rule}
{teaching_rule}
{pause_rule}

Paper title: {meta['title']}
Abstract: {meta['abstract']}
Context: {ctx}
{query_block}{web_block}{sources_block}{experiment_hint}
{feedback_block}

Generate slide #{idx}: {slide_title}
""".strip()

        def _fallback_slide() -> dict:
            """Function fallback slide.
            
            Returns:
                dict:
            """
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
                "tables": [],
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
            if "tables" not in s:
                s["tables"] = []
            if self.cfg.baseline_framing:
                s = self._ensure_baseline_framing(slide_title, s)
            if self.cfg.require_evidence:
                s = self._flag_ungrounded_claims(s, experiment_refs or [])
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
        """Build outline once.
        
        Returns:
            Tuple[DeckOutline, Dict[str, Any], str, Dict[str, Any], str, List[Dict[str, str]], str, str, List[str]]:
        """
        self._save_progress(
            {
                "stage": "start",
                "slides": [],
                "work_dir": str(self.cfg.work_dir),
                "out_dir": str(self.cfg.out_dir),
            }
        )
        sources: List[Dict[str, Any]] = []

        if self.cfg.arxiv_ids:
            logger.info("Fetching arXiv metadata and sources...")

            def _load_arxiv(arxiv_id: str) -> dict:
                try:
                    meta = self.arxiv_client.get_metadata(arxiv_id)
                    title = meta.get("title", arxiv_id)
                    abstract = meta.get("abstract", "")
                    url = meta.get("url", "")

                    logger.info("Downloading and extracting arXiv source: %s", arxiv_id)
                    arxiv_work = self.cfg.work_dir / f"arxiv_{arxiv_id}"
                    src_dir = None
                    last_err = None
                    for attempt in range(1, 3):
                        try:
                            src_dir = self.arxiv_client.download_source(arxiv_id, arxiv_work)
                            break
                        except Exception as e:
                            last_err = e
                            logger.warning("arXiv source download failed (%s/%s) for %s", attempt, 2, arxiv_id)
                    if src_dir is None:
                        logger.warning("Falling back to PDF for arXiv %s", arxiv_id)
                        try:
                            pdf_url = get_arxiv_pdf_url(arxiv_id)
                            pdf_dir = self.cfg.work_dir / "web_pdfs"
                            pdf_dir.mkdir(parents=True, exist_ok=True)
                            name = Path(pdf_url.split("?")[0]).name or f"{arxiv_id}.pdf"
                            if not name.lower().endswith(".pdf"):
                                name = name + ".pdf"
                            pdf_path = pdf_dir / name
                            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                                import requests

                                r = requests.get(pdf_url, stream=True, timeout=60)
                                r.raise_for_status()
                                with pdf_path.open("wb") as f:
                                    for chunk in r.iter_content(chunk_size=1024 * 256):
                                        if chunk:
                                            f.write(chunk)
                            paper_text = extract_pdf_content(pdf_path)
                            return {
                                "type": "pdf",
                                "id": str(pdf_path),
                                "title": title,
                                "url": pdf_url,
                                "text": paper_text,
                                "images": [],
                            }
                        except Exception as e:
                            raise RuntimeError(f"Failed to download arXiv source for {arxiv_id}: {last_err}") from e

                    main_tex = None
                    last_err = None
                    for attempt in range(1, 4):
                        try:
                            main_tex = find_main_tex_file(src_dir)
                            break
                        except Exception as e:
                            last_err = e
                            logger.warning("Main TeX discovery failed (%s/%s) for %s", attempt, 3, arxiv_id)
                    if main_tex is None:
                        raise RuntimeError(f"Failed to find main TeX for {arxiv_id}: {last_err}")

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
                except Exception:
                    logger.exception("Skipping arXiv source due to errors: %s", arxiv_id)
                return None

            max_workers = min(2, self.cfg.max_llm_workers, len(self.cfg.arxiv_ids))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_load_arxiv, a): a for a in self.cfg.arxiv_ids}
                for fut in as_completed(futures):
                    item = fut.result()
                    if item:
                        sources.append(item)

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

            max_workers = min(2, self.cfg.max_llm_workers, len(self.cfg.pdf_paths))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_load_pdf, p): p for p in self.cfg.pdf_paths}
                for fut in as_completed(futures):
                    item = fut.result()
                    if item:
                        sources.append(item)

        if self.cfg.md_paths:
            def _load_md(md_path: Path) -> dict:
                logger.info("Reading markdown file: %s", md_path)
                try:
                    text = md_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = md_path.read_text(encoding="utf-8", errors="replace")
                title = md_path.stem
                for line in text.splitlines():
                    if line.strip().startswith("#"):
                        title = line.lstrip("#").strip() or title
                        break
                paper_text = text.strip()
                logger.info("Markdown text chars: %s", len(paper_text))
                if len(paper_text) <= 200:
                    raise RuntimeError("Markdown text too small.")

                return {
                    "type": "markdown",
                    "id": str(md_path),
                    "title": title,
                    "abstract": "",
                    "url": str(md_path),
                    "text": paper_text,
                    "images": [],
                }

            max_workers = min(2, self.cfg.max_llm_workers, len(self.cfg.md_paths))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_load_md, p): p for p in self.cfg.md_paths}
                for fut in as_completed(futures):
                    item = fut.result()
                    if item:
                        sources.append(item)

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
        if self.cfg.md_paths:
            print("Markdown sources:")
            for s in sources:
                if s["type"] == "markdown":
                    print(f"- {s['title']} ({s['id']})")
            print("")

        if self.cfg.approve:
            print("\nSelected sources:")
            for i, s in enumerate(sources, 1):
                print(f"{i}. [{s['type']}] {s['title']} ({s['id']})")
            ans = input(
                "Approve sources? (y=go ahead, a=add sources, r=remove sources): "
            ).strip().lower()
            if ans in {"a", "add"}:
                add = input(
                    "Enter arXiv IDs/URLs to add (comma-separated): "
                ).strip()
                if add:
                    for part in add.replace(";", ",").split(","):
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            self.cfg.arxiv_ids.append(extract_arxiv_id(part))
                        except Exception:
                            self.cfg.arxiv_ids.append(part)
                return self.build_outline_once()
            if ans in {"r", "remove"}:
                to_remove = input(
                    "Enter source indices to remove (comma-separated): "
                ).strip()
                remove_idx = set()
                for part in to_remove.replace(";", ",").split(","):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        remove_idx.add(int(part))
                    except Exception:
                        pass
                if remove_idx:
                    for idx in sorted(remove_idx, reverse=True):
                        if 1 <= idx <= len(sources):
                            s = sources[idx - 1]
                            if s["type"] == "arxiv" and s["id"] in self.cfg.arxiv_ids:
                                self.cfg.arxiv_ids.remove(s["id"])
                            if s["type"] == "pdf" and s["id"] in [str(p) for p in self.cfg.pdf_paths]:
                                self.cfg.pdf_paths = [p for p in self.cfg.pdf_paths if str(p) != s["id"]]
                            if s["type"] == "markdown" and s["id"] in [str(p) for p in self.cfg.md_paths]:
                                self.cfg.md_paths = [p for p in self.cfg.md_paths if str(p) != s["id"]]
                return self.build_outline_once()

        if len(sources) == 1:
            meta = {"title": sources[0]["title"], "abstract": sources[0].get("abstract", "")}
        else:
            meta = {"title": "Multiple Sources", "abstract": "Multiple documents provided."}
        if not sources:
            raise RuntimeError(
                "No sources collected. Provide arXiv/PDF/Markdown sources or use --topic."
            )

        if self.cfg.arxiv_ids and not self.cfg.pdf_paths and not self.cfg.md_paths and len(self.cfg.arxiv_ids) == 1:
            source_label = f"arXiv:{self.cfg.arxiv_ids[0]}"
        elif self.cfg.arxiv_ids and not self.cfg.pdf_paths and not self.cfg.md_paths:
            source_label = f"arXiv ({len(self.cfg.arxiv_ids)})"
        elif self.cfg.pdf_paths and not self.cfg.arxiv_ids and not self.cfg.md_paths:
            source_label = f"Local PDFs ({len(self.cfg.pdf_paths)})"
        elif self.cfg.md_paths and not self.cfg.arxiv_ids and not self.cfg.pdf_paths:
            source_label = f"Markdown files ({len(self.cfg.md_paths)})"
        else:
            source_label = f"Mixed sources ({len(sources)})"

        sources_block_lines = []
        for i, s in enumerate(sources, 1):
            if s["type"] == "arxiv":
                src_tag = "arXiv"
            elif s["type"] == "markdown":
                src_tag = "Markdown"
            else:
                src_tag = "PDF"
            sources_block_lines.append(f"{i}. [{src_tag}] {s['title']} ({s['id']})")
        sources_block = "\n".join(sources_block_lines)

        blocks = []
        for s in sources:
            blocks.append(f"[SOURCE: {s['title']}]\n{s['text']}")
        paper_text = "\n\n".join(blocks)

        self._checkpoint("Sources collected")
        global_feedback = self._prompt_feedback("Global feedback")
        citations_base: List[str] = []
        web_sources = []
        web_context = ""
        if self.cfg.user_query and self.cfg.web_search and not (self.cfg.topic or "").strip():
            logger.info("Running web search for query: %s", self.cfg.user_query)
            try:
                if self.cfg.arxiv_only_search:
                    import arxiv

                    search = arxiv.Search(query=self.cfg.user_query, max_results=5)
                    web_sources = [
                        {
                            "title": r.title,
                            "url": r.entry_id,
                            "snippet": (r.summary or "")[:400],
                        }
                        for r in search.results()
                    ]
                else:
                    from web_utils import search_research_focused

                    web_sources = search_research_focused(self.cfg.user_query, max_results=5)
            except Exception:
                web_sources = []
        elif (self.cfg.topic or "").strip():
            logger.info("Skipping outline-stage web search (topic mode).")
            if web_sources:
                print("\nTop web results:")
                for i, s in enumerate(web_sources, 1):
                    print(f"{i}. {s['title']} - {s['url']}")
                print("")
                lines = []
                for i, s in enumerate(web_sources, 1):
                    lines.append(f"{i}. {s['title']} - {s['url']}\n   {s['snippet']}")
                web_context = "\n".join(lines)
        self._save_progress(
            {
                "stage": "sources",
                "meta": meta,
                "paper_text": paper_text,
                "web_context": web_context,
                "sources_block": sources_block,
                "source_label": source_label,
                "citations": citations_base,
                "slides": [],
                "work_dir": str(self.cfg.work_dir),
                "out_dir": str(self.cfg.out_dir),
                "global_feedback": global_feedback,
            }
        )

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
        chunks = self._select_relevant_chunks(chunks, global_feedback=global_feedback)
        N = len(chunks)
        sums: List[str] = []

        logger.info("Summarizing paper (%s chunks)...", N)
        if N == 0:
            raise RuntimeError("No text chunks available for summarization.")
        prev_summary_preview = ""
        max_workers = min(self.cfg.max_llm_workers, N)
        if N > 1 and max_workers > 1:
            self._checkpoint("Summarize (parallel)", 0, N)
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
            with tqdm(
                range(1, N + 1),
                desc="Summarize",
                unit="chunk",
                ncols=TQDM_NCOLS,
                dynamic_ncols=False,
            ) as bar:
                for i in bar:
                    self._checkpoint("Summarize", i, N)
                    chunk_preview = self._preview_text(chunks[i - 1], max_len=50)
                    bar.set_postfix_str(f"chunk: {chunk_preview} | prev: {prev_summary_preview}")
                    s = self.summarize_chunk(
                        i,
                        chunks[i - 1],
                        meta,
                        (self.cfg.user_query + "\n" + global_feedback).strip(),
                        web_context,
                        sources_block,
                    )
                    sums.append(s)
                    prev_summary_preview = self._preview_text(s, max_len=50)

        merged_summary = "\n\n".join(sums)
        self._save_progress(
            {
                "stage": "summary",
                "meta": meta,
                "paper_text": paper_text,
                "merged_summary": merged_summary,
                "web_context": web_context,
                "sources_block": sources_block,
                "source_label": source_label,
                "citations": citations_base,
                "slides": [],
                "work_dir": str(self.cfg.work_dir),
                "out_dir": str(self.cfg.out_dir),
                "global_feedback": global_feedback,
            }
        )

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
        if self.cfg.auto_comparisons:
            titles_obj["slide_titles"] = self._ensure_comparison_titles(
                titles_obj.get("slide_titles", [])
            )
        if self.cfg.teaching_mode:
            titles_obj["slide_titles"] = self._ensure_pause_question_titles(
                titles_obj.get("slide_titles", [])
            )
        self._print_section(
            "Slide titles",
            [t for t in titles_obj.get("slide_titles", [])],
        )
        if self.cfg.approve:
            ans = input("\nApprove slide titles? Type 'y' to approve or enter feedback: ").strip()
            titles_feedback = "" if ans.lower() in {"y", "yes"} else ans
            # Optional slide count adjustment
            adj = input("Change slide count? (enter number or press Enter to keep): ").strip()
            if adj:
                try:
                    new_count = int(adj)
                    if new_count > 0 and new_count != self.cfg.slide_count:
                        if new_count > self.cfg.slide_count:
                            extra = input(
                                "Add headings/themes to include (comma-separated, optional): "
                            ).strip()
                            if extra:
                                titles_feedback = (titles_feedback + "\n" if titles_feedback else "") + (
                                    "Add these headings/themes: " + extra
                                )
                        self.cfg.slide_count = new_count
                        titles_feedback = (titles_feedback + "\n" if titles_feedback else "") + (
                            f"Adjust to exactly {new_count} slide titles."
                        )
                except Exception:
                    pass
        else:
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
            if self.cfg.auto_comparisons:
                titles_obj["slide_titles"] = self._ensure_comparison_titles(
                    titles_obj.get("slide_titles", [])
                )
            if self.cfg.teaching_mode:
                titles_obj["slide_titles"] = self._ensure_pause_question_titles(
                    titles_obj.get("slide_titles", [])
                )
            self._print_section(
                "Revised slide titles",
                [t for t in titles_obj.get("slide_titles", [])],
            )
        diagram_plan = []
        if self.cfg.diagram_intent_aware:
            diagram_plan = self.propose_diagram_plan(
                titles_obj.get("slide_titles", []),
                merged_summary,
                user_query=(self.cfg.user_query + "\n" + global_feedback).strip(),
                web_context=web_context,
                sources_block=sources_block,
            )
        self.diagram_plan = diagram_plan
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
                "diagram_plan": diagram_plan,
                "slides": [],
                "work_dir": str(self.cfg.work_dir),
                "out_dir": str(self.cfg.out_dir),
                "global_feedback": global_feedback,
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
                    "tables": [],
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
        experiment_refs = self._experiment_slide_refs(titles_obj.get("slide_titles", []))
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
                    experiment_refs=experiment_refs,
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
                    "diagram_plan": diagram_plan,
                    "slides": slides,
                    "work_dir": str(self.cfg.work_dir),
                    "out_dir": str(self.cfg.out_dir),
                    "global_feedback": global_feedback,
                }
            )

        citations = list(citations_base)

        if self.cfg.quant_results:
            table = self._generate_quant_results_table(
                merged_summary,
                sources_block,
                web_context=web_context,
            )
            if table.get("columns") and table.get("rows"):
                slides.append(
                    {
                        "title": table.get("title", "Quantitative Results"),
                        "bullets": [],
                        "speaker_notes": "",
                        "figure_suggestions": [],
                        "generated_images": [],
                        "flowchart": {"steps": [], "structure": "linear", "caption": ""},
                        "graphviz_diagram_ideas": [],
                        "tables": [table],
                    }
                )

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
        """Function regenerate titles with feedback.
        
        Args:
            meta (dict):
            merged_summary (str):
            prev_titles (List[str]):
            feedback (str):
            user_query (str):
            web_context (str):
            sources_block (str):
            source_label (str):
        
        Returns:
            dict:
        """
        summary = re.sub(r"\s+", " ", merged_summary).strip()[:1200]
        prev = "\n".join([f"{i+1}. {t}" for i, t in enumerate(prev_titles)])
        query_block = f"\nUser query:\n{user_query}\n" if user_query else ""
        web_block = f"\nWeb sources:\n{web_context}\n" if web_context else ""
        sources_block = f"\nSources:\n{sources_block}\n" if sources_block else ""
        teaching_rule = ""
        if self.cfg.teaching_mode:
            teaching_rule = (
                "\nTeaching mode:\n"
                "- prioritize intuition and examples over equations\n"
                "- include 1-2 slide titles labeled 'Pause Question: ...'\n"
            )

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
{query_block}{web_block}{sources_block}{teaching_rule}
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
        if self.cfg.teaching_mode:
            obj["slide_titles"] = self._ensure_pause_question_titles(obj.get("slide_titles", []))
        return obj
