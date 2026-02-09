from __future__ import annotations

import json
import time
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from .arxiv_utils import extract_arxiv_id, get_arxiv_pdf_url
    from .llm import safe_invoke
    from .pdf_utils import extract_pdf_content
    from .models import DeckOutline
    from .web_utils import search_web
    from .flowchart_utils import build_graphviz, build_graphviz_from_nodes_edges, render_graphviz
    from .tex_utils import build_paper_text, find_main_tex_file, flatten_tex
    from .memory_utils import load_journal_for_date, now_iso, upsert_paper
except Exception:
    from arxiv_utils import extract_arxiv_id, get_arxiv_pdf_url
    from llm import safe_invoke
    from pdf_utils import extract_pdf_content
    from models import DeckOutline
    from web_utils import search_web
    from flowchart_utils import build_graphviz, build_graphviz_from_nodes_edges, render_graphviz
    from tex_utils import build_paper_text, find_main_tex_file, flatten_tex
    from memory_utils import load_journal_for_date, now_iso, upsert_paper

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None
    Panel = None
    Table = None

try:
    from .pipeline_arxiv import ArxivClient
    from .pipeline_common import PaperContext, RunConfig, TQDM_NCOLS, _progress_path, logger
    from .pipeline_common import OutlineJSONStore
    from .pipeline_figures import FigurePlanner
    from .pipeline_outline import OutlineBuilder
    from .pipeline_render import Renderer
except Exception:
    from pipeline_arxiv import ArxivClient
    from pipeline_common import PaperContext, RunConfig, TQDM_NCOLS, _progress_path, logger
    from pipeline_common import OutlineJSONStore
    from pipeline_figures import FigurePlanner
    from pipeline_outline import OutlineBuilder
    from pipeline_render import Renderer


def _get_console():
    """Get console.

    Returns:
        Any:
    """
    return Console() if Console else None


class Pipeline:
    def __init__(self, cfg: RunConfig, llm) -> None:
        """Initialize.
        
        Args:
            cfg (RunConfig):
            llm (Any):
        
        Returns:
            None:
        """
        self.cfg = cfg
        self.llm = llm
        self.arxiv_client = ArxivClient()
        self.outline_builder = OutlineBuilder(llm, cfg, self.arxiv_client)
        self.outline_store = OutlineJSONStore(cfg.out_dir)
        self.figure_planner = FigurePlanner()
        self.renderer = Renderer()
        self._embedder = None

    def sanity_checks(self) -> None:
        """Function sanity checks.
        
        Returns:
            None:
        """
        logger.info("Running sanity checks...")
        if self.cfg.slide_count < 2:
            raise ValueError("slide_count must be >= 2")
        if self.cfg.bullets_per_slide < 1:
            raise ValueError("bullets_per_slide must be >= 1")
        if not self.cfg.resume_path:
            if not self.cfg.arxiv_ids and not self.cfg.pdf_paths and not self.cfg.md_paths and not self.cfg.topic:
                raise ValueError("Provide arXiv/PDF/Markdown sources or use --topic")
        for p in self.cfg.pdf_paths:
            if not p.exists():
                raise FileNotFoundError(f"PDF not found: {p}")
        for p in self.cfg.md_paths:
            if not p.exists():
                raise FileNotFoundError(f"Markdown not found: {p}")

        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        if not self.cfg.skip_llm_sanity:
            x = safe_invoke(logger, self.llm, "Reply with exactly: OK", debug=self.cfg.verbose)
            logger.info("LLM sanity: %r", x[:50])
            if "OK" not in x:
                raise RuntimeError("LLM sanity check failed. Ensure your NVIDIA_API_KEY and model are valid.")

    def _load_progress(self) -> Optional[dict]:
        """Load progress.
        
        Returns:
            Optional[dict]:
        """
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
        """Prepare topic sources.
        
        Returns:
            None:
        """
        if not self.cfg.topic:
            return

        prompt = f"""
You are preparing a research presentation. Expand the topic into an elaborative,
critical-analysis query with clear sub-questions, scope, evaluation angles, and
expected evidence types. Include datasets/benchmarks, methods, metrics, and
comparative baselines where relevant. Return a single paragraph query (no bullets).

Topic: {self.cfg.topic}
""".strip()
        expanded = safe_invoke(logger, self.llm, prompt, retries=6).strip()
        if not expanded:
            expanded = self.cfg.topic
        # Query approval + feedback loop
        for _ in range(3):
            print("\nExpanded topic query:\n")
            print(expanded)
            ans = input("\nApprove query? Type 'y' to approve, or provide feedback to refine: ").strip()
            if ans.lower() in {"y", "yes"}:
                break
            if ans:
                refine_prompt = f"""
Refine the topic query based on user feedback. Make it more analytically complete:
include missing evaluation dimensions, methods, datasets, metrics, baselines, and
potential failure modes. Return a single paragraph query (no bullets).

Original topic: {self.cfg.topic}
Current query: {expanded}
User feedback: {ans}
""".strip()
                expanded = safe_invoke(logger, self.llm, refine_prompt, retries=6).strip() or expanded
            else:
                break

        self.cfg.user_query = expanded

        # Persist the expanded query to both work/ and outputs/
        try:
            self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
            self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
            (self.cfg.work_dir / "query.txt").write_text(expanded + "\n", encoding="utf-8")
            (self.cfg.out_dir / "query.txt").write_text(expanded + "\n", encoding="utf-8")
        except Exception:
            logger.exception("Failed to write query.txt")

        # Sanitize query for web search (strip markdown markers/newlines and verbose sections)
        def _strip_query_markers(text: str) -> str:
            text = re.sub(r"(?is)\\bkeywords?:.*$", " ", text)
            text = re.sub(r"(?is)\\bquery paragraph\\s*:\\s*", " ", text)
            return text

        clean_query = _strip_query_markers(expanded)
        clean_query = re.sub(r"[\\*`_#]+", " ", clean_query)
        clean_query = re.sub(r"\s+", " ", clean_query).strip()

        def _keyword_query(text: str, max_terms: int = 12) -> str:
            """Function keyword query.
            
            Args:
                text (str):
                max_terms (int):
            
            Returns:
                str:
            """
            stop = {
                "the", "and", "or", "of", "in", "to", "for", "with", "on", "by", "from",
                "a", "an", "is", "are", "was", "were", "be", "as", "that", "this", "these",
                "those", "how", "what", "why", "which", "when", "where", "who", "whom",
                "into", "about", "across", "such", "their", "they", "them", "we", "you",
                "your", "our", "using", "use", "used", "based", "more", "most", "less",
                "than", "still", "also", "while", "not", "no",
            }
            words = re.findall(r"[A-Za-z0-9]+", text.lower())
            filtered = [w for w in words if w not in stop and len(w) > 2]
            # simple de-dup while preserving order
            seen = set()
            out = []
            for w in filtered:
                if w in seen:
                    continue
                seen.add(w)
                out.append(w)
                if len(out) >= max_terms:
                    break
            return " ".join(out)

        short_query = _keyword_query(clean_query or self.cfg.topic or expanded)
        logger.info("Topic expanded query: %s", expanded)
        logger.info("Web search query (sanitized): %s", clean_query)
        logger.info("Web search query (keywords): %s", short_query)
        results: List[dict] = []
        allowed_domains: set[str] = set()
        if self.cfg.topic_allow_domains:
            allowed_domains = set(self.cfg.topic_allow_domains)
        elif self.cfg.topic_scholarly_only:
            allowed_domains = {
                "arxiv.org",
                "openaccess.thecvf.com",
                "cvpr.thecvf.com",
                "icml.cc",
                "proceedings.mlr.press",
                "neurips.cc",
                "proceedings.neurips.cc",
                "scholar.google.com",
                "openreview.net",
                "aclanthology.org",
            }

        def _matches_keywords(item: dict) -> bool:
            """Check topic keyword filters.
            
            Args:
                item (dict):
            
            Returns:
                bool:
            """
            text = " ".join([item.get("title", ""), item.get("snippet", ""), item.get("url", "")]).lower()
            for kw in self.cfg.topic_must_include:
                if kw.lower() not in text:
                    return False
            for kw in self.cfg.topic_exclude:
                if kw.lower() in text:
                    return False
            return True

        def _filter_results(items: List[dict]) -> List[dict]:
            """Apply allowlist and keyword filters.
            
            Args:
                items (List[dict]):
            
            Returns:
                List[dict]:
            """
            filtered_items = items
            if allowed_domains:
                filtered_items = []
                for r in items:
                    url = r.get("url", "")
                    try:
                        from urllib.parse import urlparse

                        host = urlparse(url).netloc.lower()
                    except Exception:
                        host = ""
                    if any(host == d or host.endswith("." + d) for d in allowed_domains):
                        filtered_items.append(r)
            if filtered_items:
                filtered_items = [r for r in filtered_items if _matches_keywords(r)]
            return filtered_items

        # Ask LLM for search queries and run them first
        # Derive core terms from the topic for anchored query generation.
        core_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9\\-]+", self.cfg.topic)
        core_terms = [t.lower() for t in core_terms if len(t) > 2]
        core_terms = list(dict.fromkeys(core_terms))[:6]  # keep it tight
        core_terms_text = ", ".join(core_terms) if core_terms else self.cfg.topic

        target_n = max(1, self.cfg.max_web_queries)
        query_prompt = f"""
Generate exactly {target_n} diverse web search queries (short keyword phrases) that stay strictly
within this domain. Each query MUST include at least 2-3 of these core topic terms:
{core_terms_text}

Queries should still vary across:
- method names / families
- datasets / benchmarks
- metrics / evaluation criteria
- comparisons (e.g., vs. baselines)
- limitations / failure cases

Avoid vague or generic queries that omit the core topic terms.
Return ONLY a JSON array of strings.

Topic: {self.cfg.topic}
""".strip()
        def _parse_query_list(raw: str) -> List[str]:
            if not raw:
                return []
            cleaned = raw.strip()
            # Strip markdown code fences if present.
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z0-9]*\\s*", "", cleaned)
                cleaned = re.sub(r"\\s*```$", "", cleaned)
                cleaned = cleaned.strip()
            try:
                import json as _json

                parsed = _json.loads(cleaned)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
            # Fallback: extract quoted strings or numbered lines.
            quoted = re.findall(r'"([^"]+)"', cleaned)
            if quoted:
                return [q.strip() for q in quoted if q.strip()]
            lines = [re.sub(r"^\\s*\\d+\\.?\\s*", "", l).strip() for l in cleaned.splitlines()]
            return [l for l in lines if l]

        def _filter_queries_to_topic(queries: List[str]) -> List[str]:
            if not queries:
                return queries
            if not core_terms:
                return queries
            filtered = []
            for q in queries:
                ql = q.lower()
                hits = sum(1 for t in core_terms if t in ql)
                if hits >= 2:
                    filtered.append(q)
            return filtered or queries

        query_list = []
        raw_q = ""
        for attempt in range(1, 4):
            raw_q = safe_invoke(logger, self.llm, query_prompt, retries=4).strip()
            query_list = _parse_query_list(raw_q)
            query_list = _filter_queries_to_topic(query_list)
            if query_list:
                break
            if raw_q:
                logger.warning(
                    "LLM search query generation failed (attempt %s/3). Raw output: %s",
                    attempt,
                    raw_q,
                )
            time.sleep(0.75 * attempt)

        if query_list:
            query_list = query_list[:target_n]
            console = _get_console()
            if console and Panel:
                body = "\n".join([f"{i}. {q}" for i, q in enumerate(query_list, 1)])
                console.print(Panel(body, title="QUERIES BY LLM", expand=False))
            else:
                print("\n----------QUERIES BY LLM----------------")
                for i, q in enumerate(query_list, 1):
                    print(f"{i}. {q}")
                print("----------------------------------------\n")
            aggregated = []
            max_attempts = 3
            for q in query_list:
                q = str(q).strip()
                if not q:
                    continue
                for attempt in range(1, max_attempts + 1):
                    logger.info("Web search run (LLM query) %s/%s: %s", attempt, max_attempts, q)
                    try:
                        before = len(aggregated)
                        fetch_limit = min(60, max(self.cfg.max_web_results * 3, self.cfg.max_web_results))
                        if self.cfg.arxiv_only_search:
                            import arxiv

                            search = arxiv.Search(query=q, max_results=fetch_limit)
                            added = []
                            for r in search.results():
                                item = {
                                    "title": r.title,
                                    "url": r.entry_id,
                                    "snippet": (r.summary or "")[:400],
                                }
                                aggregated.append(item)
                                added.append(item)
                            logger.info("arXiv results added: %s", len(aggregated) - before)
                            if added:
                                logger.info("arXiv added titles:")
                                for i, it in enumerate(added, 1):
                                    logger.info("  %s. %s", i, it.get("title", "").strip())
                        else:
                            from web_utils import search_research_focused

                            added = search_research_focused(q, max_results=self.cfg.max_web_results)
                            aggregated.extend(added)
                            logger.info("Research search results added: %s", len(aggregated) - before)
                            if added:
                                logger.info("Research search added titles:")
                                for i, it in enumerate(added, 1):
                                    logger.info("  %s. %s", i, it.get("title", "").strip())
                        if aggregated:
                            break
                    except Exception:
                        logger.exception("Web search failed for query: %s", q)
                    time.sleep(1.0 + (attempt - 1) * 0.75)
            # de-dup
            seen = set()
            deduped = []
            for r in aggregated:
                url = r.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                deduped.append(r)
            results = _filter_results(deduped[: self.cfg.max_web_results])

        # Fallback: use keyword query if LLM queries were empty or returned nothing
        if not results:
            fallback_q = short_query or clean_query
            if fallback_q:
                for attempt in range(1, 4):
                    logger.info("Web search run (fallback) %s/3: %s", attempt, fallback_q)
                    try:
                        fetch_limit = min(60, max(self.cfg.max_web_results * 3, self.cfg.max_web_results))
                        if self.cfg.arxiv_only_search:
                            import arxiv

                            search = arxiv.Search(query=fallback_q, max_results=fetch_limit)
                            results = [
                                {
                                    "title": r.title,
                                    "url": r.entry_id,
                                    "snippet": (r.summary or "")[:400],
                                }
                                for r in search.results()
                            ]
                            logger.info("arXiv results (fallback): %s", len(results))
                            if results:
                                logger.info("arXiv fallback titles:")
                                for i, it in enumerate(results, 1):
                                    logger.info("  %s. %s", i, it.get("title", "").strip())
                        else:
                            from web_utils import search_research_focused

                            results = search_research_focused(fallback_q, max_results=self.cfg.max_web_results)
                            logger.info("Research search results (fallback): %s", len(results))
                            if results:
                                logger.info("Research search fallback titles:")
                                for i, it in enumerate(results, 1):
                                    logger.info("  %s. %s", i, it.get("title", "").strip())
                        results = _filter_results(results)
                        if results:
                            break
                    except Exception:
                        logger.exception("Web search failed for fallback query.")
                    time.sleep(1.0 + (attempt - 1) * 0.75)

        # Rank results with simple heuristic for transparency
        def _rank_reason(item: dict) -> tuple[int, str]:
            title = item.get("title", "").lower()
            url = item.get("url", "").lower()
            snippet = item.get("snippet", "").lower()
            text = " ".join([title, snippet])
            score = 0
            reasons = []
            # domain bonus
            venue_map = {
                "openaccess.thecvf.com": "CVPR/ICCV/ECCV",
                "cvpr.thecvf.com": "CVPR",
                "arxiv.org": "arXiv",
                "neurips.cc": "NeurIPS",
                "proceedings.neurips.cc": "NeurIPS",
                "icml.cc": "ICML",
                "proceedings.mlr.press": "ICML",
                "openreview.net": "OpenReview",
                "aclanthology.org": "ACL",
                "scholar.google.com": "Scholar",
            }
            for d in allowed_domains:
                if d in url:
                    score += 3
                    venue = venue_map.get(d, d)
                    reasons.append(f"venue:{venue}")
                    break
            # keyword hits
            for kw in self.cfg.topic_must_include:
                if kw.lower() in text:
                    score += 2
                    reasons.append(f"kw:{kw}")
            # recency: year in title/snippet
            m = re.search(r"(20\\d{2})", text)
            if m:
                score += 1
                reasons.append(f"year:{m.group(1)}")
            if "cited by" in text or "citations" in text or "citation" in text:
                score += 1
                reasons.append("citations:mentioned")
            return score, ", ".join(reasons) or "relevance"

        if results:
            ranked = []
            for r in results:
                score, reason = _rank_reason(r)
                r["_score"] = score
                r["_reason"] = reason
                ranked.append(r)
            results = sorted(ranked, key=lambda x: x.get("_score", 0), reverse=True)

        # Lightweight topic-term filter to drop obviously irrelevant results.
        def _topic_term_filter(items: List[dict]) -> List[dict]:
            if not items or not (self.cfg.topic or "").strip():
                return items
            topic_text = " ".join([self.cfg.topic, self.cfg.user_query]).lower()
            core_terms = set(re.findall(r"[a-z0-9]{3,}", topic_text))
            # Ensure core video/keyframe terms are present in the filter set.
            core_terms.update(["video", "videos", "frame", "frames", "keyframe", "keyframes", "sampling"])
            filtered = []
            for it in items:
                text = " ".join([it.get("title", ""), it.get("snippet", ""), it.get("url", "")]).lower()
                if any(t in text for t in core_terms):
                    filtered.append(it)
            return filtered or items

        if results and not self.cfg.topic_must_include:
            before = len(results)
            results = _topic_term_filter(results)
            if len(results) != before:
                logger.info("Topic term filter: %s -> %s", before, len(results))
                dropped = [r for r in ranked if r not in results] if 'ranked' in locals() else []
                for i, r in enumerate(dropped, 1):
                    logger.info(
                        "Dropped (topic-term) %s. %s | %s",
                        i,
                        r.get("title", "").strip(),
                        r.get("url", "").strip(),
                    )

        # Enrich with arXiv metadata (title/abstract) and LLM-filter by relevance.
        def _llm_filter_results(items: List[dict]) -> List[dict]:
            if not items:
                return items
            # Extract arXiv ids when possible and fetch metadata for LLM decision.
            enriched = []
            for s in items:
                url = s.get("url", "")
                arxiv_id = ""
                if "arxiv.org" in (url or ""):
                    try:
                        arxiv_id = extract_arxiv_id(url)
                    except Exception:
                        arxiv_id = ""
                enriched.append({**s, "arxiv_id": arxiv_id})

            # Fetch metadata for arXiv items with progress.
            logger.info("Fetching arXiv metadata for LLM filtering (%s items)...", len(enriched))
            with tqdm(total=len(enriched), desc="LLM Filter Metadata", ncols=100) as bar:
                for s in enriched:
                    if s.get("arxiv_id"):
                        try:
                            meta = self.arxiv_client.get_metadata(s["arxiv_id"])
                            s["title"] = meta.get("title", s.get("title", ""))
                            s["abstract"] = meta.get("abstract", "")
                        except Exception:
                            s.setdefault("abstract", "")
                    bar.update(1)

            max_items = min(20, len(enriched))
            lines = []
            for i, s in enumerate(enriched[:max_items], 1):
                title = s.get("title", "")
                abstract = s.get("abstract", "") or s.get("snippet", "")
                url = s.get("url", "")
                arxiv_id = s.get("arxiv_id", "")
                lines.append(f"{i}. {title}\n   {abstract}\n   {url}\n   id:{arxiv_id}")
            prompt = f"""
You are filtering web search results for relevance.
Topic: {self.cfg.topic}
Return ONLY a JSON array of integers (1-based indices) for results that are clearly relevant.
If unsure, be conservative and include fewer.

Results:
{chr(10).join(lines)}
""".strip()
            raw = safe_invoke(logger, self.llm, prompt, retries=3).strip()
            try:
                import json as _json
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```[a-zA-Z0-9]*\\s*", "", cleaned)
                    cleaned = re.sub(r"\\s*```$", "", cleaned)
                    cleaned = cleaned.strip()
                idxs = _json.loads(cleaned)
                if not isinstance(idxs, list):
                    logger.warning("LLM relevance filter returned non-list JSON; using unfiltered results.")
                    return enriched
                keep = []
                for i in idxs:
                    try:
                        i = int(i)
                    except Exception:
                        continue
                    if 1 <= i <= max_items:
                        keep.append(enriched[i - 1])
                return keep or enriched
            except Exception:
                logger.warning("LLM relevance filter failed; using unfiltered results. Raw: %s", raw)
                return enriched

        if results and (self.cfg.topic or "").strip():
            before = len(results)
            pre = list(results)
            results = _llm_filter_results(results)
            logger.info("LLM filtered results: %s -> %s", before, len(results))
            kept = set((r.get("title", ""), r.get("url", "")) for r in results)
            logger.info("LLM filtered titles (kept):")
            for i, s in enumerate(results, 1):
                logger.info("  %s. %s", i, s.get("title", "").strip())
            for i, s in enumerate(results, 1):
                logger.info(
                    "LLM keep %s. %s | %s",
                    i,
                    s.get("title", "").strip(),
                    s.get("url", "").strip(),
                )
            for i, s in enumerate(pre, 1):
                key = (s.get("title", ""), s.get("url", ""))
                if key not in kept:
                    logger.info(
                        "LLM drop %s. %s | %s",
                        i,
                        s.get("title", "").strip(),
                        s.get("url", "").strip(),
                    )

        # Debug output for topic search (after any LLM-query retry)
        out_dir = self.cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "topic_web_results.txt"
        if results:
            lines = []
            for i, s in enumerate(results, 1):
                reason = s.get("_reason", "")
                lines.append(
                    f"{i}. {s.get('title','')} - {s.get('url','')}\n"
                    f"   {s.get('snippet','')}\n"
                    f"   reason: {reason}"
                )
            results_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Topic web results saved: %s", results_path)
            logger.info("Topic web results count: %s", len(results))
            console = _get_console()
            if console and Table:
                table = Table(title="WEB SEARCH RESULTS", show_lines=True)
                table.add_column("#", style="cyan", no_wrap=True)
                table.add_column("Title", style="bold")
                table.add_column("URL")
                table.add_column("Snippet")
                table.add_column("Reason")
                for i, s in enumerate(results, 1):
                    table.add_row(
                        str(i),
                        s.get("title", ""),
                        s.get("url", ""),
                        s.get("snippet", ""),
                        s.get("_reason", ""),
                    )
                console.print(table)
            else:
                print("\n-----------WEB SEARCH RESULTS------------")
                for i, s in enumerate(results, 1):
                    title = s.get("title", "")
                    url = s.get("url", "")
                    snippet = s.get("snippet", "")
                    reason = s.get("_reason", "")
                    print(f"{i}. {title}\n   {url}\n   {snippet}\n   reason: {reason}\n")
                print("----------------------------------------\n")
            # Always log final filtered results for visibility.
            logger.info("Final web results after filtering (%s):", len(results))
            for i, s in enumerate(results, 1):
                logger.info(
                    "%s. %s | %s",
                    i,
                    s.get("title", "").strip(),
                    s.get("url", "").strip(),
                )
        else:
            logger.warning("No web results found for topic search.")
            # Fallback: query arXiv directly for scholarly-only topic mode
            if self.cfg.topic_scholarly_only:
                try:
                    import arxiv

                    query = short_query or clean_query or self.cfg.topic
                    search = arxiv.Search(query=query, max_results=self.cfg.max_web_results)
                    arxiv_ids = [r.get_short_id() for r in search.results()]
                    if arxiv_ids:
                        logger.info("arXiv fallback results: %s", len(arxiv_ids))
                        results_path.write_text(
                            "arXiv fallback results:\n" + "\n".join(arxiv_ids) + "\n",
                            encoding="utf-8",
                        )
                        self.cfg.arxiv_ids = list(dict.fromkeys(self.cfg.arxiv_ids + arxiv_ids))
                        return
                except Exception:
                    logger.exception("arXiv fallback search failed.")
            results_path.write_text("No results.\n", encoding="utf-8")
            hint = "Try rephrasing the topic."
            if self.cfg.topic_scholarly_only:
                hint += " Or disable --topic-scholarly-only."
            raise RuntimeError(f"No web results found for topic search. {hint}")

        arxiv_ids = list(self.cfg.arxiv_ids)
        pdf_urls: List[str] = []
        for r in results:
            url = r.get("url", "")
            if "arxiv.org/abs/" in url or "arxiv.org/pdf/" in url:
                try:
                    arxiv_id = extract_arxiv_id(url)
                    arxiv_ids.append(arxiv_id)
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

            max_workers = min(2, self.cfg.max_llm_workers, len(pdf_urls))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_download_pdf, u): u for u in pdf_urls}
                for fut in as_completed(futures):
                    p = fut.result()
                    if p:
                        self.cfg.pdf_paths.append(p)

    @staticmethod
    def print_outline(outline: DeckOutline) -> None:
        """Print outline.
        
        Args:
            outline (DeckOutline):
        
        Returns:
            None:
        """
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
        """Select flowchart indices.
        
        Args:
            outline (DeckOutline):
        
        Returns:
            List[int]:
        """
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
        forced = set()
        for i, sl in enumerate(outline.slides):
            title = (sl.title or "").lower()
            score = sum(1 for k in keywords if k in title)
            if sl.flowchart and sl.flowchart.steps:
                score += 2
            if any(k in title for k in ["pipeline", "architecture", "framework", "training", "inference"]):
                forced.add(i)
            scored.append((score, i))
        scored.sort(reverse=True)
        target = min(
            len(outline.slides),
            max(self.cfg.min_flowcharts, min(self.cfg.max_flowcharts, len(outline.slides))),
        )
        chosen = []
        for i in forced:
            if i not in chosen:
                chosen.append(i)
        for _score, i in scored:
            if i in chosen:
                continue
            if len(chosen) >= target and forced:
                break
            chosen.append(i)
            if len(chosen) >= target:
                break
        return chosen

    def _generate_flowchart_steps(self, slide: dict, topic_hint: str = "") -> dict:
        """Generate flowchart steps.
        
        Args:
            slide (dict):
            topic_hint (str):
        
        Returns:
            dict:
        """
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
        """Render flowcharts.
        
        Args:
            outline (DeckOutline):
        
        Returns:
            None:
        """
        flow_dir = self.cfg.out_dir / "flowcharts"
        flow_dir.mkdir(parents=True, exist_ok=True)

        indices = self._select_flowchart_indices(outline)
        if not indices:
            return

        for i in indices:
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
                continue
            structure = fc.get("structure", self.cfg.flowchart_structure) or self.cfg.flowchart_structure
            style = (self.cfg.diagram_style or "flowchart").lower()
            if style == "flowchart":
                dot = build_graphviz(steps, structure=structure)
            else:
                nodes = steps
                edges = [(steps[j], steps[j + 1], "") for j in range(len(steps) - 1)]
                rankdir = "TB" if style in {"sequence", "dag"} else "LR"
                dot = build_graphviz_from_nodes_edges(nodes, edges, title=fc.get("caption", ""), rankdir=rankdir)
            dot_path = flow_dir / f"slide_{i+1:02d}.dot"
            png_path = flow_dir / f"slide_{i+1:02d}.png"
            dot_path.write_text(dot, encoding="utf-8")
            try:
                render_graphviz(dot_path, png_path)
            except Exception:
                logger.exception("Failed to render flowchart for slide %s", i + 1)
                continue
            slide.flowchart_images.append(str(png_path))

    def _render_planned_diagrams(self, outline: DeckOutline, diagram_plan: List[dict]) -> None:
        """Render planned diagrams from a diagram plan.
        
        Args:
            outline (DeckOutline):
            diagram_plan (List[dict]):
        
        Returns:
            None:
        """
        if not diagram_plan:
            return
        flow_dir = self.cfg.out_dir / "flowcharts"
        flow_dir.mkdir(parents=True, exist_ok=True)
        # Prioritize by priority then slide order
        diagram_plan = sorted(
            diagram_plan,
            key=lambda d: (int(d.get("priority", 3)), int(d.get("slide_index", 9999))),
        )
        target = min(len(diagram_plan), max(5, min(self.cfg.slide_count, 8)))
        rendered = 0
        for d in diagram_plan:
            if rendered >= target:
                break
            idx = d.get("slide_index")
            if not isinstance(idx, int) or idx < 1 or idx > len(outline.slides):
                continue
            nodes = [str(n).strip() for n in d.get("nodes", []) if str(n).strip()]
            edges_in = d.get("edges", [])
            edges = []
            for e in edges_in:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    a = str(e[0])
                    b = str(e[1])
                    lbl = str(e[2]) if len(e) >= 3 else ""
                    edges.append((a, b, lbl))
            if len(nodes) < 3:
                continue
            # Avoid purely linear chains
            if edges and len(edges) == len(nodes) - 1:
                edges.append((nodes[0], nodes[-1], "context"))
            rankdir = "LR" if d.get("type") in {"pipeline", "sequence", "block"} else "TB"
            dot = build_graphviz_from_nodes_edges(nodes, edges, title=d.get("caption", ""), rankdir=rankdir)
            dot_path = flow_dir / f"planned_slide_{idx:02d}_{rendered+1:02d}.dot"
            png_path = flow_dir / f"planned_slide_{idx:02d}_{rendered+1:02d}.png"
            dot_path.write_text(dot, encoding="utf-8")
            try:
                render_graphviz(dot_path, png_path)
            except Exception:
                logger.exception("Failed to render planned diagram for slide %s", idx)
                continue
            slide = outline.slides[idx - 1]
            slide.flowchart_images.append(str(png_path))
            cap = d.get("caption", "")
            if cap:
                slide.image_captions.append(cap)
            rendered += 1

    def _attach_figures_from_arxiv_sources(self, outline: DeckOutline) -> None:
        """Attach figures from arxiv sources.
        
        Args:
            outline (DeckOutline):
        
        Returns:
            None:
        """
        if not self.cfg.arxiv_ids:
            return
        fig_assets = []
        for arxiv_id in self.cfg.arxiv_ids:
            try:
                arxiv_work = self.cfg.work_dir / f"arxiv_{arxiv_id}"
                src_dir = arxiv_work / "arxiv_source"
                if not src_dir.exists():
                    src_dir = self.arxiv_client.download_source(arxiv_id, arxiv_work)
                main_tex = find_main_tex_file(src_dir)
                flat = flatten_tex(main_tex, max_files=120)
                fig_assets.extend(self.figure_planner.extract_figures(flat, src_dir))
            except Exception:
                logger.exception("Skipping figure extraction for arXiv: %s", arxiv_id)
                continue
        if not fig_assets:
            return

        fig_plan = self.figure_planner.plan_with_llm(self.llm, outline, fig_assets, max_figs=12)
        resolved = self.figure_planner.materialize(fig_plan, fig_assets, self.cfg.out_dir)

        for s in resolved.get("slides", []):
            idx = s.get("slide_index")
            if not idx or idx < 1 or idx > len(outline.slides):
                continue
            for g in s.get("figures", []):
                fpath = g.get("file")
                caption = g.get("caption", "")
                if not fpath:
                    continue
                outline.slides[idx - 1].generated_images.append(str(fpath))
                if caption:
                    outline.slides[idx - 1].image_captions.append(str(caption))

    def _generate_deck_diagrams(self, outline: DeckOutline) -> None:
        """Generate deck diagrams.
        
        Args:
            outline (DeckOutline):
        
        Returns:
            None:
        """
        prompt = f"""
You are designing diagrams that carry core information for the entire deck.
Generate 2-3 diagram specs that visually explain the problem, solution, and comparisons.

Return ONLY JSON in this schema:
{{
  "diagrams": [
    {{
      "type": "comparison|taxonomy|pipeline|problem_solution|flowchart",
      "title": "string",
      "nodes": ["string", "..."],
      "edges": [["from","to","label"], "..."], // label can be empty string
      "caption": "string"
    }}
  ]
}}

Rules:
- Prefer diagrams that replace text: show problem framing, method pipeline, and comparisons.
- Keep nodes short (2-6 words).
- Use 6-10 nodes per diagram.
- Use at least one comparison diagram if applicable.
- Use edges to encode relationships (e.g., improves, reduces, enables).

Deck title: {outline.deck_title}
Slide titles: {[s.title for s in outline.slides]}
""".strip()
        raw = safe_invoke(logger, self.llm, prompt, retries=6)
        if not raw.strip():
            logger.warning("Deck diagram LLM returned empty output; skipping deck diagrams.")
            return
        js = OutlineBuilder.try_extract_json(raw)
        try:
            obj = json.loads(js or raw)
        except Exception:
            fix = safe_invoke(
                logger,
                self.llm,
                "Return ONLY valid JSON for the schema. Fix this:\n" + raw[:1800],
                retries=6,
            )
            js = OutlineBuilder.try_extract_json(fix)
            try:
                obj = json.loads(js or fix)
            except Exception:
                logger.warning("Deck diagram JSON parse failed; skipping deck diagrams.")
                return

        diagrams = obj.get("diagrams", [])
        if not isinstance(diagrams, list) or not diagrams:
            return

        deck_dir = self.cfg.out_dir / "flowcharts"
        deck_dir.mkdir(parents=True, exist_ok=True)

        for i, d in enumerate(diagrams[:3], 1):
            nodes = [str(n).strip() for n in d.get("nodes", []) if str(n).strip()]
            if len(nodes) < 3:
                continue
            edges_raw = d.get("edges", [])
            edges = []
            if isinstance(edges_raw, list):
                for e in edges_raw:
                    if isinstance(e, list) and len(e) >= 2:
                        a = str(e[0]).strip()
                        b = str(e[1]).strip()
                        lbl = str(e[2]).strip() if len(e) > 2 else ""
                        if a and b:
                            edges.append((a, b, lbl))
            title = str(d.get("title", "")).strip()
            dtype = str(d.get("type", "pipeline")).strip().lower()
            rankdir = "LR" if dtype in {"pipeline", "flowchart"} else "TB"
            dot = build_graphviz_from_nodes_edges(nodes, edges, title=title, rankdir=rankdir)
            dot_path = deck_dir / f"deck_diagram_{i:02d}.dot"
            png_path = deck_dir / f"deck_diagram_{i:02d}.png"
            dot_path.write_text(dot, encoding="utf-8")
            try:
                render_graphviz(dot_path, png_path)
            except Exception:
                logger.exception("Failed to render deck diagram %s", i)
                continue

            outline.slides.append(
                {
                    "title": title or f"Diagram {i}",
                    "bullets": [],
                    "speaker_notes": "",
                    "figure_suggestions": [],
                    "generated_images": [],
                    "flowchart": {"steps": [], "structure": "linear", "caption": ""},
                    "flowchart_images": [str(png_path)],
                    "graphviz_diagram_ideas": [],
                }
            )

    def build_outline_with_approval(self, max_rounds: int = 3) -> Tuple[DeckOutline, Dict[str, Any]]:
        """Build outline with approval.
        
        Args:
            max_rounds (int):
        
        Returns:
            Tuple[DeckOutline, Dict[str, Any]]:
        """
        progress = self._load_progress()
        if progress and progress.get("stage") in {"titles", "slides", "summary", "sources"}:
            meta = progress.get("meta", {"title": "Resume", "abstract": ""})
            merged_summary = progress.get("merged_summary", "")
            titles_obj = progress.get("titles_obj", {})
            web_context = progress.get("web_context", "")
            sources_block = progress.get("sources_block", "")
            source_label = progress.get("source_label", "Resume")
            citations_base = progress.get("citations", [])
            slides = progress.get("slides", [])
            global_feedback = progress.get("global_feedback", "")
            diagram_plan = progress.get("diagram_plan", [])
            self.outline_builder.diagram_plan = diagram_plan
            logger.info("Resuming from progress.json with %s slides.", len(slides))

            if progress.get("stage") in {"summary", "sources"}:
                paper_text = progress.get("paper_text", "")
                if not merged_summary and paper_text:
                    merged_summary = self.outline_builder.summarize_text(
                        paper_text,
                        meta,
                        global_feedback,
                        web_context=web_context,
                        sources_block=sources_block,
                    )
                titles_obj = self.outline_builder.get_slide_titles(
                    meta,
                    merged_summary,
                    user_query=(self.cfg.user_query + "\n" + global_feedback).strip(),
                    web_context=web_context,
                    sources_block=sources_block,
                    source_label=source_label,
                )
                if self.cfg.auto_comparisons:
                    titles_obj["slide_titles"] = self.outline_builder._ensure_comparison_titles(
                        titles_obj.get("slide_titles", [])
                    )
                if self.cfg.teaching_mode:
                    titles_obj["slide_titles"] = self.outline_builder._ensure_pause_question_titles(
                        titles_obj.get("slide_titles", [])
                    )
                if self.cfg.diagram_intent_aware:
                    diagram_plan = self.outline_builder.propose_diagram_plan(
                        titles_obj.get("slide_titles", []),
                        merged_summary,
                        user_query=(self.cfg.user_query + "\n" + global_feedback).strip(),
                        web_context=web_context,
                        sources_block=sources_block,
                    )
                self.outline_builder.diagram_plan = diagram_plan
                self.outline_builder._save_progress(
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
                        "slides": slides,
                        "work_dir": str(self.cfg.work_dir),
                        "out_dir": str(self.cfg.out_dir),
                        "global_feedback": global_feedback,
                    }
                )

            if self.cfg.diagram_intent_aware and not diagram_plan:
                diagram_plan = self.outline_builder.propose_diagram_plan(
                    titles_obj.get("slide_titles", []),
                    merged_summary,
                    user_query=(self.cfg.user_query + "\n" + global_feedback).strip(),
                    web_context=web_context,
                    sources_block=sources_block,
                )
                self.outline_builder.diagram_plan = diagram_plan
            experiment_refs = self.outline_builder._experiment_slide_refs(
                titles_obj.get("slide_titles", [])
            )
            # Continue generating remaining slides (resume with progress bar)
            slide_items = list(enumerate(titles_obj.get("slide_titles", []), 1))
            with tqdm(slide_items, desc="Slides (resume)", unit="slide", ncols=TQDM_NCOLS) as bar:
                for idx, title in bar:
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
                            experiment_refs=experiment_refs,
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
                            "diagram_plan": diagram_plan,
                            "slides": slides,
                            "work_dir": str(self.cfg.work_dir),
                            "out_dir": str(self.cfg.out_dir),
                            "global_feedback": global_feedback,
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
                        sources_block=sources_block,
                    )
                )

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

    def build_paper_context(self) -> PaperContext:
        """Build paper context without slide generation."""
        self.outline_builder._save_progress(
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
                    for attempt in range(1, 4):
                        try:
                            src_dir = self.arxiv_client.download_source(arxiv_id, arxiv_work)
                            break
                        except Exception as e:
                            last_err = e
                            logger.warning("arXiv source download failed (%s/%s) for %s", attempt, 3, arxiv_id)
                    if src_dir is None:
                        raise RuntimeError(f"Failed to download arXiv source for {arxiv_id}: {last_err}")

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

        if not sources:
            raise RuntimeError("No sources collected. Provide arXiv/PDF/Markdown sources or use --topic.")

        if len(sources) == 1:
            meta = {"title": sources[0]["title"], "abstract": sources[0].get("abstract", "")}
        else:
            meta = {"title": "Multiple Sources", "abstract": "Multiple documents provided."}

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

        web_context = ""
        web_sources = []
        if self.cfg.user_query and self.cfg.web_search:
            logger.info("Running web search for query: %s", self.cfg.user_query)
            web_sources = search_web(self.cfg.user_query, max_results=5)
            if web_sources:
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

        merged_summary = self.outline_builder.summarize_text(
            paper_text,
            meta,
            global_feedback="",
            web_context=web_context,
            sources_block=sources_block,
        )

        self.outline_builder._save_progress(
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
                "global_feedback": "",
            }
        )

        # Persist context for chat/RAG reuse
        try:
            ctx_path = self.cfg.out_dir / "paper_context.json"
            chunks = self.outline_builder.chunk_text(paper_text, 1200)
            embeddings = []
            embed_model = ""
            if self.cfg.chat_mode:
                try:
                    embeddings, embed_model = self._embed_texts(chunks)
                except Exception:
                    embeddings = []
                    embed_model = ""
            ctx_path.write_text(
                json.dumps(
                    {
                        "meta": meta,
                        "sources_block": sources_block,
                        "source_label": source_label,
                        "summary": merged_summary,
                        "chunks": chunks,
                        "embeddings": embeddings,
                        "embed_model": embed_model,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to write paper_context.json")

        return PaperContext(
            meta=meta,
            paper_text=paper_text,
            merged_summary=merged_summary,
            sources_block=sources_block,
            source_label=source_label,
            web_context=web_context,
            citations=citations_base,
            sources=sources,
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if not text:
            return text
        t = text.strip()
        if t.startswith("```"):
            parts = t.split("```", 2)
            if len(parts) >= 3:
                return parts[1].strip()
        return t

    @staticmethod
    def _reading_notes_too_short(text: str) -> bool:
        if not text:
            return True
        # Heuristic: require at least 500 words and all headings present
        required = ["## Problem", "## Key Idea", "## Method", "## Results", "## Limitations", "## What I Learned"]
        if any(h not in text for h in required):
            return True
        word_count = len(re.findall(r"[A-Za-z0-9]+", text))
        return word_count < 500

    @staticmethod
    def _reading_missing_sections(text: str) -> List[str]:
        required = ["Problem", "Key Idea", "Method", "Results", "Limitations", "What I Learned"]
        missing = []
        for h in required:
            if f"## {h}" not in text:
                missing.append(h)
        return missing

    def _write_markdown(self, name: str, content: str) -> Path:
        path = self.cfg.out_dir / name
        path.write_text(content.strip() + "\n", encoding="utf-8")
        return path

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available")
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], str]:
        if not texts:
            return [], ""
        model = self._get_embedder()
        vecs = model.encode(texts, normalize_embeddings=True).tolist()
        return vecs, getattr(model, "model_card", "") or "all-MiniLM-L6-v2"

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        for i in range(len(a)):
            dot += a[i] * b[i]
        return dot

    def generate_reading_notes(self, ctx: PaperContext) -> Path:
        sections = ["Problem", "Key Idea", "Method", "Results", "Limitations", "What I Learned"]
        min_words = 120

        def _gen_section(sec: str, min_words_required: int) -> str:
            base_prompt = f"""
Write the **{sec}** section for the reading notes in markdown.

Rules:
- Use structured markdown for readability:
  - Start with a 1-2 sentence overview line.
  - Include a "Key Points" bullet list (4-6 bullets, bold lead phrases).
  - Include a short "Evidence/Details" subsection (2-4 bullets or 1 short paragraph).
  - End with "Implications" bullets (2-3 bullets).
- Be detailed and specific; explain core concepts clearly.
- Do NOT use code fences.
- Only write content for this section (no other top-level headings).
- Ensure at least {min_words_required} words.

Title: {ctx.meta.get('title', '')}
Abstract: {ctx.meta.get('abstract', '')}
Summary: {ctx.merged_summary[:6000]}
Sources:
{ctx.sources_block}
""".strip()
            last = base_prompt
            for attempt in range(3):
                raw = safe_invoke(logger, self.llm, last, retries=6)
                cleaned = self._strip_code_fences(raw).strip()
                word_count = len(re.findall(r"[A-Za-z0-9]+", cleaned))
                if cleaned and word_count >= min_words_required:
                    return cleaned
                last = (
                    f"Expand the **{sec}** section to be more detailed (>= {min_words_required} words). "
                    "Do NOT use code fences. Only write the section content.\n\n"
                    f"Title: {ctx.meta.get('title', '')}\n"
                    f"Summary: {ctx.merged_summary[:6000]}\n"
                )
            return cleaned or "TBD: Section generation failed."

        pieces = []
        header = [
            f"# {ctx.meta.get('title', 'Reading Notes')}",
            "",
            f"**Source:** {ctx.source_label or 'Mixed sources'}",
            "",
            "## Contents",
            *[f"- {s}" for s in sections],
            "",
        ]
        pieces.append("\n".join(header))
        with tqdm(
            sections,
            desc="Reading notes",
            unit="section",
            ncols=TQDM_NCOLS,
            dynamic_ncols=False,
        ) as bar:
            for sec in bar:
                bar.set_postfix_str(f"section: {sec}")
                logger.info("Generating reading section: %s", sec)
                body = _gen_section(sec, min_words)
                pieces.append(f"## {sec}\n{body.strip()}\n")

        combined = "\n".join(pieces).strip()
        if self._reading_notes_too_short(combined):
            # Expand each section once more if still short
            expanded = []
            with tqdm(
                sections,
                desc="Reading expand",
                unit="section",
                ncols=TQDM_NCOLS,
                dynamic_ncols=False,
            ) as bar:
                for sec in bar:
                    bar.set_postfix_str(f"section: {sec}")
                    logger.info("Expanding reading section: %s", sec)
                    body = _gen_section(sec, min_words)
                    expanded.append(f"## {sec}\n{body.strip()}\n")
            combined = "\n".join(expanded).strip()

        notes_path = self._write_markdown("reading_notes.md", combined)
        if self.cfg.generate_flowcharts:
            try:
                logger.info("Generating reading diagrams...")
                diagrams = self.generate_reading_diagrams(ctx)
                if diagrams:
                    md = notes_path.read_text(encoding="utf-8").rstrip()
                    md = self._insert_section_diagrams(md, diagrams)
                    notes_path.write_text(md.strip() + "\n", encoding="utf-8")
                else:
                    logger.warning("No diagrams generated for reading notes.")
            except Exception:
                logger.exception("Reading diagram generation failed; continuing without diagrams.")
        return notes_path

    def generate_viva_notes(self, ctx: PaperContext) -> Path:
        prompt = f"""
You are helping a student prepare for a viva. Return structured markdown with these sections:

## Common Questions
## Why The Method Works
## Failure Cases
## Comparison Traps Reviewers Ask About

Rules:
- Use bullets.
- Keep answers crisp and defensible.
- Tie to the paper context only.

Title: {ctx.meta.get('title', '')}
Abstract: {ctx.meta.get('abstract', '')}
Summary: {ctx.merged_summary[:4000]}
Sources:
{ctx.sources_block}
""".strip()
        raw = safe_invoke(logger, self.llm, prompt, retries=6)
        return self._write_markdown("viva_notes.md", raw)

    def generate_reading_diagrams(self, ctx: PaperContext) -> List[dict]:
        """Generate and render diagrams for reading notes."""
        prompt = f"""
Return ONLY JSON.

Schema:
{{
  "diagrams": [
    {{
      "section": "Problem|Key Idea|Method|Results|Limitations|What I Learned",
      "type": "comparison|taxonomy|pipeline|problem_solution|flowchart",
      "title": "string",
      "nodes": ["string", "..."],
      "edges": [["from","to","label"], "..."],
      "caption": "string"
    }}
  ]
}}

Rules:
- Provide 5-7 diagrams total.
- Keep nodes short (2-6 words).
- Use 6-10 nodes per diagram.
- Use edges to encode relationships (label can be empty).
- Prefer diagrams that explain the problem and method.
- Ensure each required section has at least one diagram assigned via the 'section' field.

Title: {ctx.meta.get('title', '')}
Abstract: {ctx.meta.get('abstract', '')}
Summary: {ctx.merged_summary[:3000]}
Sources:
{ctx.sources_block}
""".strip()
        raw = safe_invoke(logger, self.llm, prompt, retries=6)
        js = OutlineBuilder.try_extract_json(raw)
        if js is None:
            fix = safe_invoke(
                logger,
                self.llm,
                "Return ONLY valid JSON for the schema. Fix this:\n" + raw[:1800],
                retries=6,
            )
            js = OutlineBuilder.try_extract_json(fix)
        if js is None:
            return []
        try:
            obj = json.loads(js)
        except Exception:
            return []

        diagrams = obj.get("diagrams", [])
        if not isinstance(diagrams, list) or not diagrams:
            # Retry once with a more constrained prompt
            retry = safe_invoke(
                logger,
                self.llm,
                "Return ONLY JSON with 5-7 diagrams and include 'section' for each. Use the original schema.",
                retries=6,
            )
            js = OutlineBuilder.try_extract_json(retry)
            if not js:
                return []
            try:
                obj = json.loads(js)
            except Exception:
                return []
            diagrams = obj.get("diagrams", [])
            if not isinstance(diagrams, list) or not diagrams:
                return []

        flow_dir = self.cfg.out_dir / "flowcharts"
        flow_dir.mkdir(parents=True, exist_ok=True)
        rendered: List[dict] = []
        for i, d in enumerate(diagrams[:7], 1):
            nodes = [str(n).strip() for n in d.get("nodes", []) if str(n).strip()]
            if len(nodes) < 3:
                continue
            edges_raw = d.get("edges", [])
            edges = []
            if isinstance(edges_raw, list):
                for e in edges_raw:
                    if isinstance(e, list) and len(e) >= 2:
                        a = str(e[0]).strip()
                        b = str(e[1]).strip()
                        lbl = str(e[2]).strip() if len(e) > 2 else ""
                        if a and b:
                            edges.append((a, b, lbl))
            title = str(d.get("title", "")).strip()
            dtype = str(d.get("type", "pipeline")).strip().lower()
            rankdir = "LR" if dtype in {"pipeline", "flowchart"} else "TB"
            dot = build_graphviz_from_nodes_edges(nodes, edges, title=title, rankdir=rankdir)
            dot_path = flow_dir / f"reading_diagram_{i:02d}.dot"
            png_path = flow_dir / f"reading_diagram_{i:02d}.png"
            dot_path.write_text(dot, encoding="utf-8")
            try:
                render_graphviz(dot_path, png_path)
            except Exception:
                logger.exception("Failed to render reading diagram %s", i)
                continue
            rendered.append(
                {
                    "png": str(png_path.relative_to(self.cfg.out_dir)),
                    "caption": title or d.get("caption", ""),
                    "section": str(d.get("section", "")).strip(),
                }
            )
        return rendered

    def generate_section_diagrams(self, ctx: PaperContext, sections: List[str], prefix: str) -> List[dict]:
        """Generate and render diagrams for arbitrary note sections."""
        if not sections:
            return []
        sec_list = " | ".join(sections)
        prompt = f"""
Return ONLY JSON.

Schema:
{{
  "diagrams": [
    {{
      "section": "{sec_list}",
      "type": "comparison|taxonomy|pipeline|problem_solution|flowchart",
      "title": "string",
      "nodes": ["string", "..."],
      "edges": [["from","to","label"], "..."],
      "caption": "string"
    }}
  ]
}}

Rules:
- Provide 4-6 diagrams total.
- Keep nodes short (2-6 words).
- Use 6-10 nodes per diagram.
- Use edges to encode relationships (label can be empty).
- Ensure each listed section has at least one diagram assigned via 'section'.

Title: {ctx.meta.get('title', '')}
Abstract: {ctx.meta.get('abstract', '')}
Summary: {ctx.merged_summary[:3000]}
Sources:
{ctx.sources_block}
""".strip()
        raw = safe_invoke(logger, self.llm, prompt, retries=6)
        js = OutlineBuilder.try_extract_json(raw)
        if js is None:
            fix = safe_invoke(
                logger,
                self.llm,
                "Return ONLY valid JSON for the schema. Fix this:\n" + raw[:1800],
                retries=6,
            )
            js = OutlineBuilder.try_extract_json(fix)
        if js is None:
            return []
        try:
            obj = json.loads(js)
        except Exception:
            return []

        diagrams = obj.get("diagrams", [])
        if not isinstance(diagrams, list) or not diagrams:
            return []

        flow_dir = self.cfg.out_dir / "flowcharts"
        flow_dir.mkdir(parents=True, exist_ok=True)
        rendered: List[dict] = []
        for i, d in enumerate(diagrams[:6], 1):
            nodes = [str(n).strip() for n in d.get("nodes", []) if str(n).strip()]
            if len(nodes) < 3:
                continue
            edges_raw = d.get("edges", [])
            edges = []
            if isinstance(edges_raw, list):
                for e in edges_raw:
                    if isinstance(e, list) and len(e) >= 2:
                        a = str(e[0]).strip()
                        b = str(e[1]).strip()
                        lbl = str(e[2]).strip() if len(e) > 2 else ""
                        if a and b:
                            edges.append((a, b, lbl))
            title = str(d.get("title", "")).strip()
            dtype = str(d.get("type", "pipeline")).strip().lower()
            rankdir = "LR" if dtype in {"pipeline", "flowchart"} else "TB"
            dot = build_graphviz_from_nodes_edges(nodes, edges, title=title, rankdir=rankdir)
            dot_path = flow_dir / f"{prefix}_diagram_{i:02d}.dot"
            png_path = flow_dir / f"{prefix}_diagram_{i:02d}.png"
            dot_path.write_text(dot, encoding="utf-8")
            try:
                render_graphviz(dot_path, png_path)
            except Exception:
                logger.exception("Failed to render %s diagram %s", prefix, i)
                continue
            rendered.append(
                {
                    "png": str(png_path.relative_to(self.cfg.out_dir)),
                    "caption": title or d.get("caption", ""),
                    "section": str(d.get("section", "")).strip(),
                }
            )
        return rendered

    @staticmethod
    def _insert_section_diagrams(md: str, diagrams: List[dict]) -> str:
        if not md or not diagrams:
            return md
        section_map = {}
        for d in diagrams:
            sec = d.get("section", "").strip()
            if not sec:
                continue
            section_map.setdefault(sec, []).append(d)

        def _block(ds: List[dict]) -> str:
            out = []
            for d in ds:
                rel = d.get("png", "")
                cap = d.get("caption", "")
                if not rel:
                    continue
                out.append(f"![{cap}]({rel})")
                if cap:
                    out.append(f"_Caption: {cap}_")
                out.append("")
            return "\n".join(out).rstrip()

        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip()).lower()

        norm_map = { _norm(k): v for k, v in section_map.items() }

        lines = md.splitlines()
        out_lines = []
        inserted = set()
        i = 0
        while i < len(lines):
            line = lines[i]
            out_lines.append(line)
            m = re.match(r"^#{2,3}\s+(.+?)\s*$", line)
            if m:
                sec = m.group(1).strip()
                key = _norm(sec)
                if key in norm_map:
                    out_lines.append("")
                    out_lines.append(_block(norm_map[key]))
                    inserted.add(key)
            i += 1

        # Fallback: append any diagrams not inserted
        missing = [v for k, v in norm_map.items() if k not in inserted]
        if missing:
            out_lines.append("")
            out_lines.append("## Diagrams")
            out_lines.append("")
            for ds in missing:
                out_lines.append(_block(ds))
        return "\n".join(out_lines).strip()

    def generate_experiment_description(self, ctx: PaperContext) -> Path:
        sections = [
            "Dataset Description",
            "Baselines",
            "Metrics",
            "Protocol",
            "Reported Results (Table)",
        ]

        def _has_markdown_table(text: str) -> bool:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for i in range(len(lines) - 1):
                if lines[i].startswith("|") and "|" in lines[i]:
                    sep = lines[i + 1]
                    if sep.startswith("|") and set(sep.replace("|", "").replace(" ", "").replace(":", "").replace("-", "")) == set():
                        return True
            return False

        def _gen_section(sec: str) -> str:
            prompt = f"""
Write the **{sec}** section in markdown.

Rules:
- Be precise and structured; go beyond an overview.
- Include dataset names, splits, and task definitions when available.
- If details are missing, explicitly label them as missing.
- Only write content for this section (no other headings).
{(" - For Reported Results, include a markdown table with columns: Dataset | Split | Metric | Value | Notes. Use \"Not reported\" when missing." if "Reported Results" in sec else "")}

Title: {ctx.meta.get('title', '')}
Abstract: {ctx.meta.get('abstract', '')}
Summary: {ctx.merged_summary[:5000]}
Sources:
{ctx.sources_block}
            """.strip()
            body = self._strip_code_fences(safe_invoke(logger, self.llm, prompt, retries=6)).strip()
            if "Reported Results" in sec:
                # Ensure a markdown table exists (robust check)
                if not _has_markdown_table(body):
                    logger.warning("Reported Results section missing table; inserting template.")
                    body += (
                        "\n\n| Dataset | Split | Metric | Value | Notes |\n"
                        "|---|---|---|---|---|\n"
                        "| Not reported | Not reported | Not reported | Not reported | Not reported |\n"
                    )
            return body

        pieces = []
        with tqdm(sections, desc="Experiment notes", unit="section", ncols=TQDM_NCOLS, dynamic_ncols=False) as bar:
            for sec in bar:
                bar.set_postfix_str(f"section: {sec}")
                logger.info("Generating experiment section: %s", sec)
                body = _gen_section(sec)
                pieces.append(f"## {sec}\n{body}\n")

        path = self._write_markdown("experiment_description.md", "\n".join(pieces).strip())
        if self.cfg.generate_flowcharts:
            try:
                diagrams = self.generate_section_diagrams(
                    ctx,
                    ["Dataset Description", "Baselines", "Metrics", "Protocol", "Reported Results (Table)"],
                    "experiment",
                )
                if diagrams:
                    md = path.read_text(encoding="utf-8").rstrip()
                    md = self._insert_section_diagrams(md, diagrams)
                    path.write_text(md.strip() + "\n", encoding="utf-8")
            except Exception:
                logger.exception("Experiment diagram generation failed; continuing without diagrams.")
        return path

    def generate_exam_prep(self, ctx: PaperContext) -> Path:
        prompt = f"""
Create exam prep materials in markdown with sections:

## MCQs
Provide 5 questions. Each must include options A-D and the correct answer.

## Short Answers
Provide 5 short-answer questions with brief model answers.

## Derivation Questions
Provide 3 derivation-style questions; include expected steps or outline.

## Trick Questions
Provide 3 trick questions and explain the trap.

Rules:
- Keep questions grounded in the paper context.
- Avoid requiring external knowledge.

Title: {ctx.meta.get('title', '')}
Abstract: {ctx.meta.get('abstract', '')}
Summary: {ctx.merged_summary[:4000]}
Sources:
{ctx.sources_block}
""".strip()
        raw = safe_invoke(logger, self.llm, prompt, retries=6)
        return self._write_markdown("exam_prep.md", raw)

    def generate_implementation_notes(self, ctx: PaperContext) -> Path:
        sections = [
            "Model Components",
            "Architecture Details",
            "Training Loop Sketch",
            "Loss Functions",
            "Gotchas",
        ]

        def _gen_section(sec: str) -> str:
            prompt = f"""
Write the **{sec}** section in markdown.

Rules:
- Be practical, specific, and in-depth (not a high-level overview).
- Describe modules, data flow, and key operations with enough detail to reimplement.
- Include caveats when details are missing and label them explicitly.
- Only write content for this section (no other headings).

Title: {ctx.meta.get('title', '')}
Abstract: {ctx.meta.get('abstract', '')}
Summary: {ctx.merged_summary[:5000]}
Sources:
{ctx.sources_block}
""".strip()
            return self._strip_code_fences(safe_invoke(logger, self.llm, prompt, retries=6)).strip()

        pieces = []
        with tqdm(sections, desc="Implementation notes", unit="section", ncols=TQDM_NCOLS, dynamic_ncols=False) as bar:
            for sec in bar:
                bar.set_postfix_str(f"section: {sec}")
                logger.info("Generating implementation section: %s", sec)
                body = _gen_section(sec)
                pieces.append(f"## {sec}\n{body}\n")

        path = self._write_markdown("implementation_notes.md", "\n".join(pieces).strip())
        if self.cfg.generate_flowcharts:
            try:
                diagrams = self.generate_section_diagrams(
                    ctx,
                    ["Model Components", "Architecture Details", "Training Loop Sketch", "Loss Functions", "Gotchas"],
                    "implementation",
                )
                if diagrams:
                    md = path.read_text(encoding="utf-8").rstrip()
                    md = self._insert_section_diagrams(md, diagrams)
                    path.write_text(md.strip() + "\n", encoding="utf-8")
            except Exception:
                logger.exception("Implementation diagram generation failed; continuing without diagrams.")
        return path

    def generate_reproduction_checklist(self, ctx: PaperContext) -> Path:
        sections = [
            "Key Hyperparameters",
            "Training Setup",
            "Missing Details / Ambiguities",
            "Required Compute",
            "Likely Traps / Failure Modes",
            "Step-by-Step Reproduction Checklist",
        ]

        def _gen_section(sec: str) -> str:
            prompt = f"""
Write the **{sec}** section in markdown.

Rules:
- Use bullets in this section.
- Include **training hyperparameters** (optimizer, lr, batch size, epochs, schedulers, augmentations) if available.
- Be concrete, specific, and cautious about assumptions.
- If a detail is missing from the sources, explicitly label it as missing.
- Avoid code fences.
- Only write content for this section (no other headings).

Title: {ctx.meta.get('title', '')}
Abstract: {ctx.meta.get('abstract', '')}
Summary: {ctx.merged_summary[:5000]}
Sources:
{ctx.sources_block}
""".strip()
            return self._strip_code_fences(safe_invoke(logger, self.llm, prompt, retries=6)).strip()

        pieces = []
        with tqdm(sections, desc="Repro checklist", unit="section", ncols=TQDM_NCOLS, dynamic_ncols=False) as bar:
            for sec in bar:
                bar.set_postfix_str(f"section: {sec}")
                logger.info("Generating checklist section: %s", sec)
                body = _gen_section(sec)
                pieces.append(f"## {sec}\n{body}\n")

        path = self._write_markdown("reproduction_checklist.md", "\n".join(pieces).strip())
        if self.cfg.generate_flowcharts:
            try:
                diagrams = self.generate_section_diagrams(
                    ctx,
                    ["Key Hyperparameters", "Training Setup", "Missing Details / Ambiguities", "Required Compute", "Likely Traps / Failure Modes", "Step-by-Step Reproduction Checklist"],
                    "repro",
                )
                if diagrams:
                    md = path.read_text(encoding="utf-8").rstrip()
                    md = self._insert_section_diagrams(md, diagrams)
                    path.write_text(md.strip() + "\n", encoding="utf-8")
            except Exception:
                logger.exception("Repro diagram generation failed; continuing without diagrams.")
        return path

    def index_paper(self, ctx: PaperContext) -> dict:
        prompt = f"""
Return ONLY JSON.

Schema:
{{
  "summary": "string",
  "key_claims": ["string", "..."],
  "methods": ["string", "..."],
  "datasets": ["string", "..."],
  "keywords": ["string", "..."]
}}

Rules:
- Keep lists short (3-7 items).
- Use plain text only.
- Stay faithful to the provided context.

Title: {ctx.meta.get('title', '')}
Abstract: {ctx.meta.get('abstract', '')}
Summary: {ctx.merged_summary[:4000]}
Sources:
{ctx.sources_block}
""".strip()
        raw = safe_invoke(logger, self.llm, prompt, retries=6)
        js = OutlineBuilder.try_extract_json(raw)
        if js is None:
            fix = safe_invoke(
                logger,
                self.llm,
                "Return ONLY valid JSON for the schema. Fix this:\n" + raw[:1800],
                retries=6,
            )
            js = OutlineBuilder.try_extract_json(fix) or fix
        try:
            obj = json.loads(js)
        except Exception:
            obj = {"summary": "", "key_claims": [], "methods": [], "datasets": [], "keywords": []}

        entry = {
            "paper_id": ctx.source_label,
            "title": ctx.meta.get("title", ctx.source_label),
            "summary": obj.get("summary", ""),
            "key_claims": obj.get("key_claims", []) or [],
            "methods": obj.get("methods", []) or [],
            "datasets": obj.get("datasets", []) or [],
            "keywords": obj.get("keywords", []) or [],
            "sources": ctx.sources_block,
            "updated_at": now_iso(),
        }
        upsert_paper(entry)
        (self.cfg.out_dir / "paper_index_entry.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return entry

    def generate_daily_brief(self, date_str: str) -> Path:
        entries = load_journal_for_date(date_str)
        if not entries:
            content = f"# Daily Research Brief ({date_str})\n\nNo runs recorded today.\n"
            return self._write_markdown(f"daily_brief_{date_str}.md", content)

        blocks = []
        for e in entries:
            blocks.append(
                "\n".join(
                    [
                        f"- time: {e.get('time','')}",
                        f"- mode: {', '.join(e.get('modes', []) or [])}",
                        f"- source: {e.get('source_label','')}",
                        f"- outputs: {', '.join(e.get('outputs', []) or [])}",
                        f"- notes: {e.get('summary_excerpt','')[:400]}",
                    ]
                )
            )
        prompt = f"""
You are a research assistant writing a daily brief in markdown.
Summarize:

## Papers Read Today
## Ideas Explored
## TODOs Inferred From Runs

Be concise and practical. Only use the provided entries.

Date: {date_str}
Entries:
{chr(10).join(blocks)}
""".strip()
        raw = safe_invoke(logger, self.llm, prompt, retries=6)
        return self._write_markdown(f"daily_brief_{date_str}.md", raw)

    def run_non_slide(self) -> List[Path]:
        self.sanity_checks()
        self.prepare_topic_sources()
        logger.info("Starting non-slide mode...")
        ctx = None
        if self.cfg.resume_path and (
            self.cfg.read_mode
            or self.cfg.viva_mode
            or self.cfg.describe_experiments
            or self.cfg.exam_prep
            or self.cfg.implementation_notes
            or self.cfg.reproduction_checklist
        ):
            # Prefer paper_context.json if present
            resume_out = self.cfg.resume_path
            if resume_out.name != "outputs":
                resume_out = resume_out / "outputs"
            ctx_path = resume_out / "paper_context.json"
            if ctx_path.exists():
                try:
                    logger.info("Loading paper context from %s", ctx_path)
                    obj = json.loads(ctx_path.read_text(encoding="utf-8"))
                    meta = obj.get("meta", {})
                    paper_text = "\n\n".join(obj.get("chunks", []))
                    merged_summary = obj.get("summary", "")
                    sources_block = obj.get("sources_block", "")
                    source_label = obj.get("source_label", "")
                    ctx = PaperContext(
                        meta=meta,
                        paper_text=paper_text,
                        merged_summary=merged_summary,
                        sources_block=sources_block,
                        source_label=source_label,
                        web_context="",
                        citations=[],
                        sources=[],
                    )
                except Exception:
                    logger.exception("Failed to read paper_context.json; falling back to progress.json")
            if ctx is None:
                progress = self._load_progress()
                if progress:
                    logger.info("Loading progress.json from resume path")
                    meta = progress.get("meta", {})
                    paper_text = progress.get("paper_text", "")
                    merged_summary = progress.get("merged_summary", "")
                    sources_block = progress.get("sources_block", "")
                    source_label = progress.get("source_label", "")
                    web_context = progress.get("web_context", "")
                    citations = progress.get("citations", [])
                    if paper_text:
                        ctx = PaperContext(
                            meta=meta,
                            paper_text=paper_text,
                            merged_summary=merged_summary,
                            sources_block=sources_block,
                            source_label=source_label,
                            web_context=web_context,
                            citations=citations,
                            sources=[],
                        )
            if ctx is not None and not ctx.merged_summary:
                raise RuntimeError(
                    "Resume requested for non-slide mode but no merged_summary found. "
                    "Re-run without --resume to regenerate summary, or ensure paper_context.json exists."
                )
            if ctx is None:
                logger.warning(
                    "Resume requested but no paper_context.json or usable progress.json found. "
                    "Falling back to fresh extraction."
                )
            if ctx is None:
                # Try building context from the latest outline JSON in outputs.
                try:
                    outline_files = sorted(
                        resume_out.glob("outline-*.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if outline_files:
                        logger.info("Building read context from outline: %s", outline_files[0])
                        obj = json.loads(outline_files[0].read_text(encoding="utf-8"))
                        slides = obj.get("slides", []) or []
                        bullets = []
                        notes = []
                        for s in slides:
                            bullets.extend(s.get("bullets", []) or [])
                            if s.get("speaker_notes"):
                                notes.append(s.get("speaker_notes", ""))
                        merged_summary = "\n".join([*bullets, *notes]).strip()
                        citations = obj.get("citations", []) or []
                        sources_block = "\n".join(citations)
                        ctx = PaperContext(
                            meta={"title": obj.get("deck_title", "Presentation")},
                            paper_text=merged_summary,
                            merged_summary=merged_summary,
                            sources_block=sources_block,
                            source_label=obj.get("arxiv_id", ""),
                            web_context="",
                            citations=citations,
                            sources=[],
                        )
                except Exception:
                    logger.exception("Failed to build context from outline JSON.")
        if ctx is None:
            ctx = self.build_paper_context()

        outputs: List[Path] = []
        if self.cfg.read_mode:
            outputs.append(self.generate_reading_notes(ctx))
        if self.cfg.viva_mode:
            outputs.append(self.generate_viva_notes(ctx))
        if self.cfg.describe_experiments:
            outputs.append(self.generate_experiment_description(ctx))
        if self.cfg.exam_prep:
            outputs.append(self.generate_exam_prep(ctx))
        if self.cfg.implementation_notes:
            outputs.append(self.generate_implementation_notes(ctx))
        if self.cfg.reproduction_checklist:
            outputs.append(self.generate_reproduction_checklist(ctx))
        if self.cfg.index_paper:
            self.index_paper(ctx)
            outputs.append(self.cfg.out_dir / "paper_index_entry.json")

        return outputs

    def chat_with_paper(self) -> Path:
        """Interactive chat about the paper using simple retrieval."""
        self.sanity_checks()
        self.prepare_topic_sources()

        ctx_path = self.cfg.out_dir / "paper_context.json"
        if ctx_path.exists():
            try:
                obj = json.loads(ctx_path.read_text(encoding="utf-8"))
                meta = obj.get("meta", {})
                summary = obj.get("summary", "")
                sources_block = obj.get("sources_block", "")
                chunks = obj.get("chunks", [])
                embeddings = obj.get("embeddings", [])
            except Exception:
                meta = {}
                summary = ""
                sources_block = ""
                chunks = []
                embeddings = []
        else:
            ctx = self.build_paper_context()
            meta = ctx.meta
            summary = ctx.merged_summary
            sources_block = ctx.sources_block
            chunks = self.outline_builder.chunk_text(ctx.paper_text, 1200)
            embeddings = []

        if not chunks:
            chunks = [summary] if summary else []
        if self.cfg.chat_mode and chunks and not embeddings:
            try:
                embeddings, _ = self._embed_texts(chunks)
                ctx_path.write_text(
                    json.dumps(
                        {
                            "meta": meta,
                            "sources_block": sources_block,
                            "source_label": meta.get("title", ""),
                            "summary": summary,
                            "chunks": chunks,
                            "embeddings": embeddings,
                            "embed_model": "all-MiniLM-L6-v2",
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except Exception:
                embeddings = []

        history_path = self.cfg.out_dir / "chat_history.md"
        history_path.write_text(
            f"# Paper Chat\n\nTitle: {meta.get('title','')}\n\nSources:\n{sources_block}\n\n",
            encoding="utf-8",
        )

        console = _get_console()
        if console and Panel:
            console.print(Panel("Chat mode started. Type a question, or 'exit' to quit.", title="ResearchOS Chat"))
        else:
            print("\nChat mode started. Type a question, or 'exit' to quit.\n")
        while True:
            q = input("You> ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                break

            if embeddings:
                retrieved = self._retrieve_chunks_semantic(q, chunks, embeddings, k=4)
            else:
                retrieved = self._retrieve_chunks(q, chunks, k=4)
            context_block = "\n\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(retrieved)])
            prompt = f"""
You are a helpful research assistant. Answer the question strictly using the provided context.
If the answer is not in the context, say so and ask a clarifying question.

Title: {meta.get('title','')}
Summary: {summary[:2000]}
Sources:
{sources_block}

Context:
{context_block}

Question: {q}
""".strip()

            a = safe_invoke(logger, self.llm, prompt, retries=6).strip()
            if not a:
                a = "I couldn't generate an answer. Please rephrase."
            if console and Panel:
                console.print(Panel(a, title="Assistant", border_style="green"))
            else:
                print(f"\nAssistant> {a}\n")

            with history_path.open("a", encoding="utf-8") as f:
                f.write(f"## Q\n{q}\n\n## A\n{a}\n\n")

        if console and Panel:
            console.print(Panel(f"Chat history saved to: {history_path}", title="Saved", border_style="cyan"))
        else:
            print(f"Chat history saved to: {history_path}")
        return history_path

    @staticmethod
    def _retrieve_chunks(query: str, chunks: List[str], k: int = 4) -> List[str]:
        terms = set(re.findall(r"[A-Za-z0-9]+", query.lower()))
        if not terms:
            return chunks[:k]
        scored = []
        for c in chunks:
            text = c.lower()
            score = sum(1 for t in terms if t in text)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _s, c in scored[:k]]

    def _retrieve_chunks_semantic(
        self,
        query: str,
        chunks: List[str],
        embeddings: List[List[float]],
        k: int = 4,
    ) -> List[str]:
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            return self._retrieve_chunks(query, chunks, k=k)
        try:
            qvecs, _ = self._embed_texts([query])
            qvec = qvecs[0] if qvecs else []
        except Exception:
            return self._retrieve_chunks(query, chunks, k=k)
        scored = []
        for i, vec in enumerate(embeddings):
            scored.append((self._cosine_sim(qvec, vec), chunks[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _s, c in scored[:k]]

    def run(self) -> Tuple[DeckOutline, Optional[Path], Optional[Path]]:
        """Run.
        
        Returns:
            Tuple[DeckOutline, Optional[Path], Optional[Path]]:
        """
        self.sanity_checks()
        self.prepare_topic_sources()
        outline, _meta = self.build_outline_with_approval(max_rounds=3)

        if self.cfg.interactive:
            ans = input("[Render] Press Enter to render outputs or type 'q' to quit: ").strip().lower()
            if ans in {"q", "quit", "exit"}:
                raise RuntimeError("Aborted by user.")

        if self.cfg.diagram_intent_aware:
            try:
                self._render_planned_diagrams(outline, self.outline_builder.diagram_plan)
            except Exception:
                logger.exception("Planned diagram rendering failed; continuing without planned diagrams.")

        if self.cfg.generate_flowcharts:
            try:
                self._render_flowcharts(outline)
                self._generate_deck_diagrams(outline)
            except Exception:
                logger.exception("Flowchart generation failed; continuing without flowcharts.")

        if self.cfg.use_figures:
            # For topic/multi-source runs, attach figures via LLM+captions instead of strict single-arXiv flow.
            if self.cfg.pdf_paths or self.cfg.md_paths or len(self.cfg.arxiv_ids) != 1:
                try:
                    self._attach_figures_from_arxiv_sources(outline)
                except Exception:
                    logger.exception("Figure attachment failed; continuing without figures.")
                tex_path, pdf_path = self.renderer.render(outline, self.cfg.out_dir)
            else:
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

    def generate_figures_only(self) -> List[Path]:
        """Generate diagrams only (no slides)."""
        self.sanity_checks()
        self.prepare_topic_sources()
        ctx = self.build_paper_context()

        if not self.cfg.generate_flowcharts:
            logger.info("Figures-only requested; enabling diagram generation.")

        diagrams: List[dict] = []
        try:
            diagrams = self.generate_reading_diagrams(ctx)
        except Exception:
            logger.exception("Figure generation failed.")
            return []

        outputs: List[Path] = []
        for d in diagrams:
            rel = d.get("png", "")
            if not rel:
                continue
            outputs.append(self.cfg.out_dir / rel)

        if diagrams:
            lines = ["# Figures", ""]
            for d in diagrams:
                rel = d.get("png", "")
                cap = d.get("caption", "")
                if not rel:
                    continue
                lines.append(f"![{cap}]({rel})")
                if cap:
                    lines.append(f"_Caption: {cap}_")
                lines.append("")
            fig_md = self._write_markdown("figures_only.md", "\n".join(lines).strip())
            outputs.append(fig_md)

        return outputs

    def revise_with_figures(self, run_dir: Path) -> tuple[Path, Optional[Path]]:
        """Revise an existing run by attaching figures from LaTeX sources.

        Args:
            run_dir (Path): Run directory containing work/ and outputs/

        Returns:
            (tex_path, pdf_path)
        """
        base_dir = Path(run_dir).expanduser().resolve()
        if base_dir.name in {"outputs", "work"}:
            base_dir = base_dir.parent
        work_dir = base_dir / "work"
        out_dir = base_dir / "outputs"
        if not out_dir.exists():
            raise RuntimeError(f"outputs/ not found under: {base_dir}")
        if not work_dir.exists():
            raise RuntimeError(f"work/ not found under: {base_dir}")

        outline_path = None
        outlines = sorted(out_dir.glob("outline-*.json"), key=lambda p: p.stat().st_mtime)
        if outlines:
            outline_path = outlines[-1]
        else:
            raise RuntimeError("No outline-*.json found in outputs/ to revise.")

        outline_dict = json.loads(outline_path.read_text(encoding="utf-8"))
        outline = DeckOutline.model_validate(outline_dict)

        # Infer arXiv IDs from work dir
        arxiv_ids: List[str] = []
        for p in work_dir.glob("arxiv_*"):
            if p.is_dir():
                arxiv_id = p.name.replace("arxiv_", "", 1)
                if arxiv_id and arxiv_id != "source":
                    arxiv_ids.append(arxiv_id)
        if not arxiv_ids:
            logger.warning("No arXiv sources found under work/. Figures may not be attachable.")
        else:
            logger.info("Revising with figures from arXiv IDs: %s", ", ".join(arxiv_ids))

        # Temporarily override cfg paths and arxiv_ids for figure attachment
        orig_work = self.cfg.work_dir
        orig_out = self.cfg.out_dir
        orig_arxiv = list(self.cfg.arxiv_ids)
        try:
            self.cfg.work_dir = work_dir
            self.cfg.out_dir = out_dir
            self.cfg.arxiv_ids = arxiv_ids
            self._attach_figures_from_arxiv_sources(outline)
        finally:
            self.cfg.work_dir = orig_work
            self.cfg.out_dir = orig_out
            self.cfg.arxiv_ids = orig_arxiv

        # Save revised outline
        revised_path = out_dir / "outline-figures.json"
        revised_path.write_text(json.dumps(outline.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

        tex_path, pdf_path = self.renderer.render(outline, out_dir)
        return tex_path, pdf_path

    def revise_with_diagrams(
        self, run_dir: Path
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Revise an existing run by generating and attaching diagrams.

        Returns:
            (tex_path, pdf_path, notes_path)
        """
        base_dir = Path(run_dir).expanduser().resolve()
        if base_dir.name in {"outputs", "work"}:
            base_dir = base_dir.parent
        work_dir = base_dir / "work"
        out_dir = base_dir / "outputs"
        if not out_dir.exists():
            raise RuntimeError(f"outputs/ not found under: {base_dir}")

        outline_path = None
        outlines = sorted(out_dir.glob("outline-*.json"), key=lambda p: p.stat().st_mtime)
        if outlines:
            outline_path = outlines[-1]

        ctx = None
        ctx_path = out_dir / "paper_context.json"
        if ctx_path.exists():
            obj = json.loads(ctx_path.read_text(encoding="utf-8"))
            ctx = PaperContext(
                meta=obj.get("meta", {}),
                paper_text=obj.get("paper_text", ""),
                merged_summary=obj.get("merged_summary", ""),
                sources_block=obj.get("sources_block", ""),
                source_label=obj.get("source_label", ""),
                web_context=obj.get("web_context", ""),
                citations=obj.get("citations", []),
                sources=obj.get("sources", []),
            )
        else:
            progress_path = out_dir / "progress.json"
            if progress_path.exists():
                obj = json.loads(progress_path.read_text(encoding="utf-8"))
                ctx = PaperContext(
                    meta=obj.get("meta", {}),
                    paper_text=obj.get("paper_text", ""),
                    merged_summary=obj.get("merged_summary", ""),
                    sources_block=obj.get("sources_block", ""),
                    source_label=obj.get("source_label", ""),
                    web_context=obj.get("web_context", ""),
                    citations=obj.get("citations", []),
                    sources=obj.get("sources", []),
                )

        tex_path = None
        pdf_path = None
        notes_path = None

        if outline_path:
            outline_dict = json.loads(outline_path.read_text(encoding="utf-8"))
            outline = DeckOutline.model_validate(outline_dict)

            merged_summary = ctx.merged_summary if ctx else ""
            if not merged_summary:
                merged_summary = (ctx.paper_text if ctx else "")[:8000]

            titles = [s.title if hasattr(s, "title") else s.get("title", "Slide") for s in outline.slides]
            diagram_plan = self.outline_builder.propose_diagram_plan(
                titles=titles,
                merged_summary=merged_summary or "",
                user_query=(self.cfg.user_query or ""),
                web_context=(ctx.web_context if ctx else ""),
                sources_block=(ctx.sources_block if ctx else ""),
            )
            self.outline_builder.diagram_plan = diagram_plan

            if self.cfg.diagram_intent_aware:
                self._render_planned_diagrams(outline, diagram_plan)
            if self.cfg.generate_flowcharts:
                self._render_flowcharts(outline)
                self._generate_deck_diagrams(outline)

            revised_path = out_dir / "outline-diagrams.json"
            revised_path.write_text(json.dumps(outline.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
            tex_path, pdf_path = self.renderer.render(outline, out_dir)
        else:
            notes_path = out_dir / "reading_notes.md"
            if notes_path.exists() and ctx:
                logger.info("Adding diagrams to existing reading notes...")
                diagrams = self.generate_reading_diagrams(ctx)
                if diagrams:
                    md = notes_path.read_text(encoding="utf-8").rstrip()
                    md = self._insert_section_diagrams(md, diagrams)
                    notes_path.write_text(md.strip() + "\n", encoding="utf-8")
            else:
                logger.warning("No outline or reading notes found to revise.")

        return tex_path, pdf_path, notes_path

    def edit_run(self, run_dir: Path, instructions: str, target: str = "auto") -> List[Path]:
        """Edit an existing run using instructions without recomputing sources.

        Args:
            run_dir (Path): Run directory containing work/ and outputs/
            instructions (str): Edit instructions
            target (str): auto|slides|reading|implementation|experiment|repro|viva|exam
        """
        base_dir = Path(run_dir).expanduser().resolve()
        if base_dir.name in {"outputs", "work"}:
            base_dir = base_dir.parent
        out_dir = base_dir / "outputs"
        if not out_dir.exists():
            raise RuntimeError(f"outputs/ not found under: {base_dir}")

        outputs: List[Path] = []

        def _edit_markdown(src_path: Path, label: str) -> Path:
            original = src_path.read_text(encoding="utf-8")
            prompt = f"""
You are editing an existing {label} document. Apply the user instructions carefully.
Keep structure and headings unless the instructions require changes. Return FULL revised markdown.

Instructions:
{instructions}

Document:
{original}
""".strip()
            revised = self._strip_code_fences(safe_invoke(logger, self.llm, prompt, retries=6)).strip()
            out_path = src_path.with_name(src_path.stem + "_edited.md")
            out_path.write_text(revised + "\n", encoding="utf-8")
            return out_path

        outline_path = None
        outlines = sorted(out_dir.glob("outline-*.json"), key=lambda p: p.stat().st_mtime)
        if outlines:
            outline_path = outlines[-1]

        # If target is slides or auto with outline, edit outline JSON
        if target in {"slides", "auto"} and outline_path:
            outline_dict = json.loads(outline_path.read_text(encoding="utf-8"))
            prompt = f"""
Return ONLY JSON for the slide deck with the SAME schema as the input.
Apply the user instructions. Preserve slide count unless instructions say otherwise.

Instructions:
{instructions}

Deck JSON:
{json.dumps(outline_dict, ensure_ascii=False)}
""".strip()
            raw = safe_invoke(logger, self.llm, prompt, retries=6)
            js = self.outline_builder.try_extract_json(raw) or raw
            revised_dict = json.loads(js)
            outline = DeckOutline.model_validate(revised_dict)
            revised_path = out_dir / "outline-edited.json"
            revised_path.write_text(json.dumps(outline.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
            outputs.append(revised_path)
            tex_path, pdf_path = self.renderer.render(outline, out_dir)
            outputs.append(tex_path)
            if pdf_path:
                outputs.append(pdf_path)
            return outputs

        # Fallback to document edits
        doc_map = {
            "reading": "reading_notes.md",
            "implementation": "implementation_notes.md",
            "experiment": "experiment_description.md",
            "repro": "reproduction_checklist.md",
            "viva": "viva_notes.md",
            "exam": "exam_prep.md",
        }
        if target == "auto":
            for key, fname in doc_map.items():
                p = out_dir / fname
                if p.exists():
                    outputs.append(_edit_markdown(p, key))
            if outputs:
                return outputs
            raise RuntimeError("No outline or known markdown outputs found to edit.")
        else:
            fname = doc_map.get(target)
            if not fname:
                raise RuntimeError(f"Unknown edit target: {target}")
            p = out_dir / fname
            if not p.exists():
                raise RuntimeError(f"{fname} not found in outputs/")
            outputs.append(_edit_markdown(p, target))
            return outputs
