"""Streamlit GUI for Paper2ppt."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import streamlit as st

from arxiv_utils import extract_arxiv_id, get_arxiv_metadata
from llm import LLMConfig, init_llm, safe_invoke
from logging_utils import setup_logging
from memory_utils import search_index, today_str
from pipeline import Pipeline, RunConfig


APP_TITLE = "Paper2ppt GUI"
CONFIG_PATH = Path.home() / ".paper2ppt_gui.json"


class _LogBufferHandler(logging.Handler):
    def __init__(self) -> None:
        """Initialize.
        
        Returns:
            None:
        """
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Function emit.
        
        Args:
            record (logging.LogRecord):
        
        Returns:
            None:
        """
        msg = self.format(record)
        self.lines.append(msg)
        if len(self.lines) > 1000:
            self.lines = self.lines[-1000:]


def _split_list_args(values: list[str]) -> list[str]:
    """Split list args.
    
    Args:
        values (list[str]):
    
    Returns:
        list[str]:
    """
    out: list[str] = []
    for v in values:
        if not v:
            continue
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip() for p in s.replace(";", ",").split(",")]
        out.extend([p for p in parts if p])
    return out


def _slugify(s: str, max_len: int = 80) -> str:
    """Slugify.
    
    Args:
        s (str):
        max_len (int):
    
    Returns:
        str:
    """
    import re

    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = s.strip("_")
    return (s or "paper").strip()[:max_len]


def _query_summary(query: str) -> str:
    """Function query summary.
    
    Args:
        query (str):
    
    Returns:
        str:
    """
    import re

    words = re.findall(r"[A-Za-z0-9]+", query or "")
    if not words:
        return "Query"
    return "_".join(words[:2])


def _collect_pdfs(paths: list[str], dirs: list[str]) -> list[Path]:
    """Collect pdfs.
    
    Args:
        paths (list[str]):
        dirs (list[str]):
    
    Returns:
        list[Path]:
    """
    pdfs: list[Path] = []
    for p in _split_list_args(paths):
        pdfs.append(Path(p).expanduser().resolve())
    for d in _split_list_args(dirs):
        dpath = Path(d).expanduser().resolve()
        if dpath.exists() and dpath.is_dir():
            pdfs.extend(sorted(dpath.glob("*.pdf")))
    return pdfs


def _download_pdfs(urls: list[str], out_dir: Path) -> list[Path]:
    """Download pdfs.
    
    Args:
        urls (list[str]):
        out_dir (Path):
    
    Returns:
        list[Path]:
    """
    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for u in _split_list_args(urls):
        try:
            name = Path(u.split("?")[0]).name or "paper.pdf"
            if not name.lower().endswith(".pdf"):
                name = name + ".pdf"
            target = out_dir / name
            if target.exists() and target.stat().st_size > 0:
                downloaded.append(target)
                continue
            r = requests.get(u, stream=True, timeout=30)
            r.raise_for_status()
            with target.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            downloaded.append(target)
        except Exception as exc:
            st.warning(f"Failed to download {u}: {exc}")
    return downloaded


def _save_uploads(files, out_dir: Path) -> list[Path]:
    """Save uploads.
    
    Args:
        files (Any):
        out_dir (Path):
    
    Returns:
        list[Path]:
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        target = out_dir / f.name
        if target.exists() and target.stat().st_size > 0:
            saved.append(target)
            continue
        with target.open("wb") as fh:
            fh.write(f.getbuffer())
        saved.append(target)
    return saved


def _resolve_sources(arxiv_text: str, pdf_text: str, pdf_dir_text: str, pdf_url_text: str, uploads: list[Path]) -> tuple[list[str], list[Path]]:
    """Resolve sources.
    
    Args:
        arxiv_text (str):
        pdf_text (str):
        pdf_dir_text (str):
        pdf_url_text (str):
        uploads (list[Path]):
    
    Returns:
        tuple[list[str], list[Path]]:
    """
    arxiv_inputs = _split_list_args([arxiv_text]) if arxiv_text else []
    arxiv_ids = sorted({extract_arxiv_id(a) for a in arxiv_inputs if a})

    pdf_paths = _collect_pdfs([pdf_text] if pdf_text else [], [pdf_dir_text] if pdf_dir_text else [])
    if uploads:
        pdf_paths.extend(uploads)

    if pdf_url_text:
        downloads = _download_pdfs([pdf_url_text], Path(st.session_state["work_dir"]) / "downloads")
        pdf_paths.extend(downloads)

    deduped: dict[str, Path] = {}
    for p in pdf_paths:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        deduped[key] = p
    return arxiv_ids, list(deduped.values())


def _load_gui_config() -> dict:
    """Load gui config.
    
    Returns:
        dict:
    """
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_gui_config(data: dict) -> None:
    """Save gui config.
    
    Args:
        data (dict):
    
    Returns:
        None:
    """
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _derive_run_name(arxiv_ids: list[str], pdf_paths: list[Path], query: str) -> str:
    """Function derive run name.
    
    Args:
        arxiv_ids (list[str]):
        pdf_paths (list[Path]):
        query (str):
    
    Returns:
        str:
    """
    title = ""
    if arxiv_ids:
        try:
            meta = get_arxiv_metadata(arxiv_ids[0])
            title = meta.get("title", arxiv_ids[0])
        except Exception:
            title = arxiv_ids[0]
    elif pdf_paths:
        title = pdf_paths[0].stem
    else:
        title = "paper"

    run_name = _slugify(title)
    if query:
        if len(arxiv_ids) + len(pdf_paths) > 1:
            run_name = f"Q-{_query_summary(query)}-MultiSource"
        else:
            run_name = f"Q-{_query_summary(query)}-{run_name}"
    return run_name


def main() -> None:
    """Function main.
    
    Returns:
        None:
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Create Beamer presentations and reading artifacts from arXiv papers and local PDFs.")

    if "uploads" not in st.session_state:
        st.session_state["uploads"] = []
    if "work_dir" not in st.session_state:
        st.session_state["work_dir"] = tempfile.mkdtemp(prefix="paper2ppt_gui_")
    if "gui_config" not in st.session_state:
        st.session_state["gui_config"] = _load_gui_config()

    with st.sidebar:
        st.header("Inputs")
        arxiv_text = st.text_input("arXiv IDs/URLs (comma-separated)")
        pdf_text = st.text_input("PDF paths (comma-separated)")
        pdf_dir_text = st.text_input("PDF directory")
        pdf_url_text = st.text_input("PDF URL(s) (comma-separated)")
        uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded:
            saved = _save_uploads(uploaded, Path(st.session_state["work_dir"]) / "uploads")
            st.session_state["uploads"].extend(saved)
        if st.session_state["uploads"]:
            st.write("Uploaded PDFs:")
            for idx, p in enumerate(st.session_state["uploads"], 1):
                st.write(f"{idx}. {p.name}")
            if st.button("Clear uploaded PDFs"):
                st.session_state["uploads"] = []

        st.header("Run Settings")
        mode = st.selectbox(
            "Mode",
            [
                "Slides",
                "Read Notes",
                "Viva Notes",
                "Experiment Description",
                "Exam Prep",
                "Implementation Notes",
                "Reproduction Checklist",
                "Index Paper",
                "Search Index",
                "Daily Brief",
                "Chat",
            ],
        )
        slides = st.number_input("Slides", min_value=2, max_value=60, value=10, step=1)
        bullets = st.number_input("Bullets per slide", min_value=1, max_value=10, value=4, step=1)
        query = st.text_input("User query")
        use_figures = st.checkbox("Use figures (single arXiv only)")
        generate_flowcharts = st.checkbox("Generate flowcharts (Graphviz)")
        diagram_intent_aware = st.checkbox("Diagram intent-aware")
        diagram_style = st.selectbox("Diagram style", ["flowchart", "block", "sequence", "dag"], index=0)
        min_flowcharts = st.number_input("Min flowcharts", min_value=0, max_value=10, value=3, step=1)
        max_flowcharts = st.number_input("Max flowcharts", min_value=0, max_value=10, value=4, step=1)
        with_notes = st.checkbox("Include speaker notes")
        teaching_mode = st.checkbox("Teaching mode (pause questions)")
        require_evidence = st.checkbox("Require evidence tags")
        auto_comparisons = st.checkbox("Auto comparisons")
        baseline_framing = st.checkbox("Baseline framing")
        quant_results = st.checkbox("Quant results table")
        skip_sanity = st.checkbox("Skip LLM sanity check")
        approve = st.checkbox("Require outline approval", value=True)
        retry_slides = st.number_input("Retry slides", min_value=1, max_value=6, value=3, step=1)
        web_search = st.checkbox("Enable web search", value=True)
        model = st.text_input("NVIDIA model", value="nvidia/llama-3.1-nemotron-ultra-253b-v1")
        max_llm_workers = st.number_input("Max LLM workers", min_value=1, max_value=16, value=4, step=1)
        cache_summary = st.checkbox("Cache summaries (3h TTL)")
        resume_path = st.text_input("Resume path (optional)")
        search_query = st.text_input("Index search query", value="") if mode == "Search Index" else ""
        chat_question = st.text_input("Chat question", value="") if mode == "Chat" else ""

        default_root = st.session_state["gui_config"].get(
            "root_dir", str(Path.home() / "paper2ppt_runs")
        )
        root_dir = st.text_input("Root runs directory", value=default_root)
        if st.button("Save as default root"):
            st.session_state["gui_config"]["root_dir"] = root_dir
            _save_gui_config(st.session_state["gui_config"])
            st.success("Default root saved")
        work_override = st.text_input("Work directory (optional)")
        out_override = st.text_input("Output directory (optional)")

    arxiv_ids, pdf_paths = _resolve_sources(arxiv_text, pdf_text, pdf_dir_text, pdf_url_text, st.session_state["uploads"])
    st.subheader("Sources")
    if not arxiv_ids and not pdf_paths:
        st.info("Add at least one arXiv ID/URL or PDF.")
    if arxiv_ids:
        st.write("arXiv IDs:")
        st.write(arxiv_ids)
    if pdf_paths:
        st.write("PDFs:")
        st.write([str(p) for p in pdf_paths])

    run_name = _derive_run_name(arxiv_ids, pdf_paths, query)
    run_root = Path(root_dir).expanduser().resolve()
    run_dir = run_root / run_name
    work_dir = Path(work_override).expanduser().resolve() if work_override else (run_dir / "work")
    out_dir = Path(out_override).expanduser().resolve() if out_override else (run_dir / "outputs")

    st.subheader("Output")
    st.write(f"Run directory: {run_dir}")

    if st.button("Run Paper2ppt"):
        if mode not in {"Daily Brief", "Search Index"} and not arxiv_ids and not pdf_paths:
            st.error("Provide at least one source to continue.")
            return

        cfg = RunConfig(
            arxiv_ids=arxiv_ids,
            pdf_paths=pdf_paths,
            work_dir=work_dir,
            out_dir=out_dir,
            slide_count=int(slides),
            bullets_per_slide=int(bullets),
            max_summary_chunks=30,
            approve=approve,
            verbose=False,
            skip_llm_sanity=skip_sanity,
            llm_model=model,
            llm_api_key=os.environ.get("NVIDIA_API_KEY", ""),
            use_figures=use_figures,
            include_speaker_notes=with_notes,
            user_query=query.strip(),
            web_search=web_search,
            retry_slides=int(retry_slides),
            retry_empty=3,
            interactive=False,
            check_interval=5,
            resume_path=Path(resume_path).expanduser().resolve() if resume_path else None,
            generate_flowcharts=generate_flowcharts,
            min_flowcharts=int(min_flowcharts),
            max_flowcharts=int(max_flowcharts),
            flowchart_structure="linear",
            flowchart_depth=8,
            max_llm_workers=int(max_llm_workers),
            topic="",
            max_web_results=6,
            max_web_pdfs=4,
            topic_scholarly_only=False,
            titles_only=False,
            diagram_style=diagram_style,
            topic_must_include=[],
            topic_exclude=[],
            topic_allow_domains=[],
            require_evidence=require_evidence,
            diagram_intent_aware=diagram_intent_aware,
            auto_comparisons=auto_comparisons,
            baseline_framing=baseline_framing,
            quant_results=quant_results,
            teaching_mode=teaching_mode,
            read_mode=(mode == "Read Notes"),
            viva_mode=(mode == "Viva Notes"),
            describe_experiments=(mode == "Experiment Description"),
            exam_prep=(mode == "Exam Prep"),
            implementation_notes=(mode == "Implementation Notes"),
            reproduction_checklist=(mode == "Reproduction Checklist"),
            index_paper=(mode == "Index Paper"),
            index_search_query=search_query,
            daily_brief=(mode == "Daily Brief"),
            cache_summary=cache_summary,
            chat_mode=(mode == "Chat"),
        )

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.user_query:
            (cfg.out_dir / "query.txt").write_text(cfg.user_query + "\n", encoding="utf-8")
        setup_logging(False, log_path=cfg.out_dir / "run.log")
        log_handler = _LogBufferHandler()
        log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(log_handler)

        result: dict = {}

        def _run_pipeline() -> None:
            """Run pipeline.
            
            Returns:
                None:
            """
            try:
                cached = st.session_state.get("_llm_cache")
                if cached and cached.get("model") == cfg.llm_model and cached.get("key") == cfg.llm_api_key:
                    llm = cached["llm"]
                else:
                    llm = init_llm(LLMConfig(model=cfg.llm_model, api_key=cfg.llm_api_key))
                    st.session_state["_llm_cache"] = {
                        "model": cfg.llm_model,
                        "key": cfg.llm_api_key,
                        "llm": llm,
                    }
                pipeline = Pipeline(cfg, llm)
                if mode == "Daily Brief":
                    out_path = pipeline.generate_daily_brief(today_str())
                    result["daily_brief"] = out_path
                elif mode == "Search Index":
                    result["search_results"] = search_index(search_query)
                elif mode == "Chat":
                    ctx_path = cfg.out_dir / "paper_context.json"
                    if not ctx_path.exists():
                        pipeline.build_paper_context()
                    obj = json.loads(ctx_path.read_text(encoding="utf-8"))
                    meta = obj.get("meta", {})
                    summary = obj.get("summary", "")
                    sources_block = obj.get("sources_block", "")
                    chunks = obj.get("chunks", [])
                    embeddings = obj.get("embeddings", [])
                    if not embeddings:
                        try:
                            embeddings, _ = pipeline._embed_texts(chunks)
                        except Exception:
                            embeddings = []
                    if chat_question.strip():
                        if embeddings:
                            retrieved = pipeline._retrieve_chunks_semantic(chat_question, chunks, embeddings, k=4)
                        else:
                            retrieved = pipeline._retrieve_chunks(chat_question, chunks, k=4)
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

Question: {chat_question}
""".strip()
                        answer = safe_invoke(logging.getLogger("paper2ppt"), llm, prompt, retries=6).strip()
                        result["chat_answer"] = answer
                else:
                    if mode == "Slides":
                        outline, tex_path, pdf_path = pipeline.run()
                        result["outline"] = outline
                        result["tex_path"] = tex_path
                        result["pdf_path"] = pdf_path
                    else:
                        outputs = pipeline.run_non_slide()
                        result["outputs"] = outputs
            except Exception as exc:
                result["error"] = str(exc)

        log_box = st.empty()
        with st.spinner("Running pipeline..."):
            t = threading.Thread(target=_run_pipeline, daemon=True)
            t.start()
            while t.is_alive():
                log_box.text_area("Live logs", value="\n".join(log_handler.lines), height=300)
                time.sleep(0.5)
            log_box.text_area("Live logs", value="\n".join(log_handler.lines), height=300)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Done")
            if "tex_path" in result:
                st.write(f"TeX: {result['tex_path']}")
            if "pdf_path" in result:
                st.write(f"PDF: {result['pdf_path']}")
            if "outputs" in result:
                st.write("Generated outputs:")
                st.write([str(p) for p in result["outputs"]])
            if "daily_brief" in result:
                st.write(f"Daily brief: {result['daily_brief']}")
            if "search_results" in result:
                st.write("Search results:")
                st.write(result["search_results"])
            if "chat_answer" in result:
                st.markdown("**Answer:**")
                st.write(result["chat_answer"])
            st.write("Output directory:")
            st.write(str(cfg.out_dir.resolve()))

    st.divider()
    st.caption("Tip: set NVIDIA_API_KEY in your environment before launching Streamlit.")


if __name__ == "__main__":
    main()
