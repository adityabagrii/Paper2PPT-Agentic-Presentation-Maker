"""CLI entrypoint for running the Paper2ppt pipeline from the terminal lists all the arguments that can be passed to the framework for the slide deck generation."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import hashlib
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from .arxiv_utils import extract_arxiv_id, get_arxiv_metadata
    from .llm import LLMConfig, init_llm
    from .logging_utils import setup_logging
    from .pipeline import Pipeline, RunConfig
    from .memory_utils import append_journal, search_index, today_str
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent))
    from arxiv_utils import extract_arxiv_id, get_arxiv_metadata
    from llm import LLMConfig, init_llm
    from logging_utils import setup_logging
    from pipeline import Pipeline, RunConfig
    from memory_utils import append_journal, search_index, today_str

logger = logging.getLogger("paper2ppt")
VERSION = "0.6.4"


def _requirements_path() -> Path | None:
    """Function requirements path.
    
    Returns:
        Path | None:
    """
    req = Path(__file__).parent / "requirements.txt"
    return req if req.exists() else None


def _requirements_hash(req_path: Path) -> str:
    """Function requirements hash.
    
    Args:
        req_path (Path):
    
    Returns:
        str:
    """
    h = hashlib.sha256()
    h.update(req_path.read_bytes())
    return h.hexdigest()


def ensure_requirements_installed() -> None:
    """Ensure requirements installed.
    
    Returns:
        None:
    """
    req_path = _requirements_path()
    if not req_path:
        logger.debug("requirements.txt not found; skipping auto-install.")
        return

    state_path = Path.home() / ".paper2ppt_requirements.sha256"
    current = _requirements_hash(req_path)
    previous = state_path.read_text(encoding="utf-8").strip() if state_path.exists() else ""
    if current == previous:
        return

    logger.info("requirements.txt changed; installing/updating dependencies...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("Dependency install failed.\n%s", (r.stdout + "\n" + r.stderr)[-4000:])
        raise RuntimeError("Failed to install updated requirements.")

    state_path.write_text(current + "\n", encoding="utf-8")


def print_helper() -> None:
    """Print helper.
    
    Returns:
        None:
    """
    print("Paper2ppt help")
    print("")
    print("Quick start:")
    print('  paper2ppt --arxiv "https://arxiv.org/abs/2602.05883" --slides 10 --bullets 4')
    print('  paper2ppt --pdf "/path/to/paper.pdf" --slides 10 --bullets 4')
    print('  paper2ppt --pdf-url "https://example.com/paper.pdf" --slides 10 --bullets 4')
    print('  paper2ppt --arxiv 1811.12432 --query "Compare this to prior work" --slides 10 --bullets 4')
    print('  paper2ppt --arxiv "1811.12432,1707.06347" --slides 10 --bullets 4')
    print('  paper2ppt --pdf-dir "/path/to/pdfs" --query "Compare methods" --slides 10 --bullets 4')
    print("")
    print("Defaults:")
    print("  Root runs dir: ~/paper2ppt_runs or $PAPER2PPT_ROOT_DIR")
    print("  Per-run structure: <root>/<paper_title_slug>/{work,outputs}")
    print("")
    print("Common options:")
    print("  -a, --arxiv LIST       One or more arXiv IDs/URLs")
    print("  -p, --pdf LIST         One or more local PDF paths")
    print("  -d, --pdf-dir PATH     Directory of PDFs")
    print("  -u, --pdf-url LIST     One or more direct PDF URLs")
    print("  --root-dir PATH        Override root runs directory")
    print("  --work-dir PATH        Override working directory")
    print("  --out-dir PATH         Override output directory")
    print("  --query TEXT           Guide the presentation theme (web search enabled)")
    print("  --no-web-search        Disable web search")
    print("  --use-figures          Enable figure selection and insertion")
    print("  --with-speaker-notes   Generate speaker notes for each slide")
    print("  --skip-llm-sanity      Skip LLM sanity check")
    print("  --no-approve           Skip outline approval loop")
    print("  --interactive          Prompt at checkpoints to allow aborting")
    print("  --generate-flowcharts  Generate Graphviz flowcharts for key slides")
    print("")
    print("Full options:")
    print("  paper2ppt --help")


def parse_args() -> argparse.Namespace:
    """Parse args.
    
    Returns:
        argparse.Namespace:
    """
    p = argparse.ArgumentParser(description="Generate a Beamer slide deck from arXiv papers or local PDFs.")
    p.add_argument("--version", action="version", version=f"paper2ppt {VERSION}")
    p.add_argument(
        "-a",
        "--arxiv",
        action="append",
        help="arXiv link or ID (repeatable or comma-separated list)",
    )
    p.add_argument(
        "-p",
        "--pdf",
        action="append",
        help="Path to a local PDF file (repeatable or comma-separated list)",
    )
    p.add_argument(
        "-d",
        "--pdf-dir",
        action="append",
        help="Directory containing PDFs (repeatable)",
    )
    p.add_argument(
        "-u",
        "--pdf-url",
        action="append",
        help="Direct PDF URL (repeatable or comma-separated list)",
    )
    p.add_argument("--slides", "-s", type=int, default=12, help="Number of slides to generate")
    p.add_argument("--bullets", "-b", type=int, default=4, help="Number of bullets per slide")
    p.add_argument("--query", "-q", default="", help="User query to guide the presentation theme")
    p.add_argument("--name", "-n", default="", help="Custom run name for output directory")
    p.add_argument("--no-web-search", "-ws", action="store_true", help="Disable web search even if --query is provided")
    p.add_argument("--retry-slides", "-rs", type=int, default=3, help="Retry count for slide generation")
    p.add_argument("--retry-empty", "-re", type=int, default=3, help="Retry count for empty LLM outputs")
    p.add_argument("--interactive", "-I", action="store_true", help="Enable interactive checkpoints to allow aborting")
    p.add_argument(
        "--check-interval",
        "-ci",
        type=int,
        default=5,
        help="How often (in steps) to prompt during interactive runs",
    )
    p.add_argument("--max-llm-workers", "-workers", type=int, default=4, help="Max parallel LLM calls")
    p.add_argument("--generate-flowcharts", "-gf", action="store_true", help="Generate Graphviz flowcharts")
    p.add_argument("--generate-images", "-gi", action="store_true", help="Alias for --generate-flowcharts")
    p.add_argument("--min-flowcharts", "-minf", type=int, default=3, help="Min flowcharts per deck")
    p.add_argument("--max-flowcharts", "-maxf", type=int, default=4, help="Max flowcharts per deck")
    p.add_argument("--resume", "-r", default="", help="Resume from a previous run directory or outputs directory")
    p.add_argument(
        "--root-dir",
        default=None,
        help="Root directory for all runs (default: $PAPER2PPT_ROOT_DIR or ~/paper2ppt_runs)",
    )
    p.add_argument("--work-dir", "-wdir", default=None, help="Working directory (overrides --root-dir)")
    p.add_argument("--out-dir", "-odir", default=None, help="Output directory (overrides --root-dir)")
    p.add_argument("--max-summary-chunks", "-msc", type=int, default=30, help="Max summary chunks to process")
    p.add_argument("--topic", "-t", default="", help="Topic-only mode: research and generate from a topic")
    p.add_argument("--max-web-results", "-maxres", type=int, default=6, help="Max web results to consider in topic mode")
    p.add_argument("--max-web-pdfs", "-maxpdf", type=int, default=4, help="Max PDFs to download in topic mode")
    p.add_argument(
        "--topic-scholarly-only", "-tso",
        action="store_true",
        help="Restrict topic mode to scholarly sources (arXiv/CVPR/ICML/NeurIPS/Scholar)",
    )
    p.add_argument("--must-include", action="append", default=[], help="Keyword(s) that must appear in sources")
    p.add_argument("--exclude", action="append", default=[], help="Keyword(s) to exclude from sources")
    p.add_argument(
        "--domains",
        action="append",
        default=[],
        help="Allowlist domains for topic mode (repeatable or comma-separated)",
    )
    p.add_argument(
        "--diagram-style",
        default="flowchart",
        choices=["flowchart", "block", "sequence", "dag"],
        help="Default diagram style for method slides",
    )
    p.add_argument("--require-evidence", action="store_true", help="Require evidence tags for claims")
    p.add_argument("--diagram-intent-aware", action="store_true", help="Generate intent-driven diagrams after titles")
    p.add_argument("--auto-comparisons", action="store_true", help="Auto-add comparison slides")
    p.add_argument("--baseline-framing", action="store_true", help="Add baseline framing prompts on experiment slides")
    p.add_argument("--quant-results", action="store_true", help="Add quantitative results table slide")
    p.add_argument("--no-approve", "-na", action="store_true", help="Skip outline approval loop")
    p.add_argument("--skip-llm-sanity", "-llms", action="store_true", help="Skip LLM sanity check")
    p.add_argument("--model", "-m", default="nvidia/llama-3.1-nemotron-ultra-253b-v1", help="NVIDIA NIM model name")
    p.add_argument("--use-figures", "-uf", action="store_true", help="Enable figure selection and insertion")
    p.add_argument("--with-speaker-notes", "-wsn", action="store_true", help="Generate speaker notes for each slide")
    p.add_argument("--titles-only", action="store_true", help="Stop after slide titles (skip slide generation)")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p.add_argument("--read", action="store_true", help="Generate reading notes (no slides)")
    p.add_argument("--viva-mode", action="store_true", help="Generate viva prep notes (no slides)")
    p.add_argument("--describe-experiments", action="store_true", help="Generate experiment description (no slides)")
    p.add_argument("--exam-prep", action="store_true", help="Generate exam prep materials (no slides)")
    p.add_argument("--implementation-notes", action="store_true", help="Generate implementation notes (no slides)")
    p.add_argument("--teaching-mode", action="store_true", help="Teaching-optimized slides with pause questions")
    p.add_argument("--index-paper", action="store_true", help="Index a paper into local memory")
    p.add_argument("--search", default="", help="Search the local paper index")
    p.add_argument("--daily-brief", action="store_true", help="Generate a daily research brief")
    return p.parse_args()


def _slugify(s: str, max_len: int = 80) -> str:
    """Slugify.
    
    Args:
        s (str):
        max_len (int):
    
    Returns:
        str:
    """
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
    words = re.findall(r"[A-Za-z0-9]+", query or "")
    if not words:
        return "Query"
    return "_".join(words[:2])


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
            r = requests.get(u, stream=True, timeout=30)
            r.raise_for_status()
            with target.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            downloaded.append(target)
        except Exception as exc:
            print(f"\nFailed to download PDF URL: {u}\nReason: {exc}")
            ans = input("Type 's' to skip this PDF, or 'q' to quit: ").strip().lower()
            if ans in {"q", "quit", "exit"}:
                raise SystemExit(2)
            logger.warning("Skipping PDF URL after failure: %s", u)
    return downloaded

def main() -> int:
    """Function main.
    
    Returns:
        int:
    """
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        print_helper()
        return 0

    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=False)
    args = parse_args()

    mode_daily_brief = bool(args.daily_brief)
    mode_search = bool((args.search or "").strip())
    non_slide_modes = any(
        [
            args.read,
            args.viva_mode,
            args.describe_experiments,
            args.exam_prep,
            args.implementation_notes,
            args.index_paper,
        ]
    )

    ensure_requirements_installed()

    arxiv_inputs = _split_list_args(args.arxiv or [])
    pdf_paths = _collect_pdfs(args.pdf or [], args.pdf_dir or [])
    pdf_urls = _split_list_args(args.pdf_url or [])
    if not args.resume:
        if not mode_daily_brief and not mode_search:
            if not arxiv_inputs and not pdf_paths and not pdf_urls and not (args.topic or "").strip():
                logger.error("Provide sources or use --topic for topic-only mode.")
                return 2

    arxiv_ids: list[str] = []
    if arxiv_inputs:
        for a in arxiv_inputs:
            arxiv_ids.append(extract_arxiv_id(a))
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        logger.warning("NVIDIA_API_KEY is not set; proceeding without a key.")

    paper_title = ""
    if arxiv_ids:
        try:
            meta = get_arxiv_metadata(arxiv_ids[0])
            paper_title = meta.get("title", arxiv_ids[0])
        except Exception:
            paper_title = arxiv_ids[0]
    elif pdf_paths:
        paper_title = pdf_paths[0].stem
    elif pdf_urls:
        paper_title = Path(pdf_urls[0].split("?")[0]).stem or "paper"
    elif mode_daily_brief:
        paper_title = f"DailyBrief_{today_str()}"
    elif mode_search:
        paper_title = f"IndexSearch_{_query_summary(args.search)}"

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        run_dir = resume_path.parent if resume_path.name == "outputs" else resume_path
        resume_out = resume_path if resume_path.name == "outputs" else (run_dir / "outputs")
        progress_path = resume_out / "progress.json"
        if not progress_path.exists():
            logger.error(
                "Resume requested but progress.json not found at %s. "
                "Re-run with sources/topic to start a new run, or resume from a run that contains progress.json.",
                progress_path,
            )
            return 2
    else:
        root_dir = args.root_dir or os.environ.get("PAPER2PPT_ROOT_DIR", "~/paper2ppt_runs")
        run_root = Path(root_dir).expanduser().resolve()
        base_name = _slugify(args.name) if args.name else _slugify(paper_title)
        run_name = base_name
        # If user explicitly set --name, honor it verbatim (no query prefixing).
        if not (mode_daily_brief or mode_search):
            if args.query and not args.name:
                if len(arxiv_ids) + len(pdf_paths) > 1:
                    run_name = f"Q-{_query_summary(args.query)}-{base_name or 'MultiSource'}"
                else:
                    run_name = f"Q-{_query_summary(args.query)}-{base_name}"
        run_dir = run_root / run_name

    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else (run_dir / "work")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (run_dir / "outputs")
    if pdf_urls:
        downloaded = _download_pdfs(pdf_urls, work_dir / "downloads")
        pdf_paths.extend(downloaded)

    cfg = RunConfig(
        arxiv_ids=arxiv_ids,
        pdf_paths=pdf_paths,
        work_dir=work_dir,
        out_dir=out_dir,
        slide_count=args.slides,
        bullets_per_slide=args.bullets,
        max_summary_chunks=args.max_summary_chunks,
        approve=not args.no_approve,
        verbose=args.verbose,
        skip_llm_sanity=args.skip_llm_sanity,
        llm_model=args.model,
        llm_api_key=api_key,
        use_figures=args.use_figures,
        include_speaker_notes=args.with_speaker_notes,
        user_query=(args.query or "").strip(),
        web_search=not args.no_web_search,
        retry_slides=max(1, args.retry_slides),
        retry_empty=max(1, args.retry_empty),
        interactive=args.interactive,
        check_interval=max(1, args.check_interval),
        resume_path=Path(args.resume).expanduser().resolve() if args.resume else None,
        generate_flowcharts=bool(args.generate_flowcharts or args.generate_images),
        min_flowcharts=max(0, args.min_flowcharts),
        max_flowcharts=max(0, args.max_flowcharts),
        flowchart_structure="linear",
        flowchart_depth=8,
        max_llm_workers=max(1, args.max_llm_workers),
        topic=(args.topic or "").strip(),
        max_web_results=max(1, args.max_web_results),
        max_web_pdfs=max(0, args.max_web_pdfs),
        topic_scholarly_only=bool(args.topic_scholarly_only),
        titles_only=bool(args.titles_only),
        diagram_style=args.diagram_style,
        topic_must_include=_split_list_args(args.must_include or []),
        topic_exclude=_split_list_args(args.exclude or []),
        topic_allow_domains=_split_list_args(args.domains or []),
        require_evidence=bool(args.require_evidence),
        diagram_intent_aware=bool(args.diagram_intent_aware),
        auto_comparisons=bool(args.auto_comparisons),
        baseline_framing=bool(args.baseline_framing),
        quant_results=bool(args.quant_results),
        teaching_mode=bool(args.teaching_mode),
        read_mode=bool(args.read),
        viva_mode=bool(args.viva_mode),
        describe_experiments=bool(args.describe_experiments),
        exam_prep=bool(args.exam_prep),
        implementation_notes=bool(args.implementation_notes),
        index_paper=bool(args.index_paper),
        index_search_query=(args.search or "").strip(),
        daily_brief=bool(args.daily_brief),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.user_query:
        (cfg.out_dir / "query.txt").write_text(cfg.user_query + "\n", encoding="utf-8")
    setup_logging(args.verbose, log_path=cfg.out_dir / "run.log")

    if mode_search:
        results = search_index(cfg.index_search_query, limit=10)
        if results:
            lines = ["# Search Results", ""]
            for i, r in enumerate(results, 1):
                lines.append(f"## {i}. {r.get('title','')}")
                lines.append(f"- paper_id: {r.get('paper_id','')}")
                if r.get("summary"):
                    lines.append(f"- summary: {r.get('summary','')}")
                if r.get("key_claims"):
                    lines.append(f"- key_claims: {', '.join(r.get('key_claims', [])[:5])}")
                if r.get("methods"):
                    lines.append(f"- methods: {', '.join(r.get('methods', [])[:5])}")
                if r.get("datasets"):
                    lines.append(f"- datasets: {', '.join(r.get('datasets', [])[:5])}")
                lines.append("")
        else:
            lines = ["# Search Results", "", "No matches found."]
        out_path = cfg.out_dir / "search_results.md"
        out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        print("\nOutput directory:", cfg.out_dir.resolve())
        print("Search results:", out_path.name)
        return 0

    try:
        logger.info("Initializing LLM...")
        llm = init_llm(LLMConfig(model=cfg.llm_model, api_key=cfg.llm_api_key))

        pipeline = Pipeline(cfg, llm)
        if mode_daily_brief:
            date_str = today_str()
            out_path = pipeline.generate_daily_brief(date_str)
            print("\nOutput directory:", cfg.out_dir.resolve())
            print("Daily brief:", out_path.name)
            return 0

        if non_slide_modes:
            outputs = pipeline.run_non_slide()
            summary_excerpt = ""
            if outputs:
                try:
                    summary_excerpt = outputs[0].read_text(encoding="utf-8")[:800]
                except Exception:
                    summary_excerpt = ""
            modes = []
            if args.read:
                modes.append("read")
            if args.viva_mode:
                modes.append("viva")
            if args.describe_experiments:
                modes.append("describe_experiments")
            if args.exam_prep:
                modes.append("exam_prep")
            if args.implementation_notes:
                modes.append("implementation_notes")
            if args.index_paper:
                modes.append("index_paper")
            append_journal(
                {
                    "date": today_str(),
                    "time": time.strftime("%H:%M:%S"),
                    "modes": modes,
                    "source_label": " ".join(arxiv_ids) or (pdf_paths[0].stem if pdf_paths else (args.topic or "")),
                    "outputs": [p.name for p in outputs],
                    "run_dir": str(run_dir),
                    "out_dir": str(cfg.out_dir),
                    "summary_excerpt": summary_excerpt,
                }
            )
            print("\nOutput directory:", cfg.out_dir.resolve())
            if outputs:
                print("Generated:", ", ".join([p.name for p in outputs]))
            else:
                print("Generated: (no files)")
            return 0

        outline, tex_path, pdf_path = pipeline.run()

        logger.info("Saved TeX: %s", tex_path)
        logger.info("Saved PDF: %s", pdf_path)

        slide_titles = [s.title for s in outline.slides][:6]
        summary_excerpt = "Slides: " + "; ".join(slide_titles)
        outputs = []
        if tex_path:
            outputs.append(Path(tex_path).name)
        if pdf_path:
            outputs.append(Path(pdf_path).name)
        append_journal(
            {
                "date": today_str(),
                "time": time.strftime("%H:%M:%S"),
                "modes": ["slides"] + (["teaching"] if args.teaching_mode else []),
                "source_label": outline.arxiv_id,
                "outputs": outputs,
                "run_dir": str(run_dir),
                "out_dir": str(cfg.out_dir),
                "summary_excerpt": summary_excerpt,
            }
        )

        print("\nOutput directory:", cfg.out_dir.resolve())
        print("TeX exists:", tex_path.exists())
        print("PDF exists:", pdf_path.exists() if pdf_path else False)

        return 0
    except Exception:
        logger.exception("Unhandled error in pipeline run")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
Example Run Command:
paper2ppt -a "1811.12432, 2404.04346, 2510.13891, 2503.13139, 2502.21271"\
    -u https://openaccess.thecvf.com/content/CVPR2025/papers/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.pdf\
    -s 15 -b 4 -q "Compare these key frame detection algorithms list their similarities and differences among each other and based on the results from the papers talk about the most efficient approach"\
    -rs 2 -msc 40 -llms -uf -wsn -n "KeyFrameComparisionWithImages" -gi
    
paper2ppt -t "Dataset report for Key Frame Sampling, with best performers in each dataset"\
    -s 16 -b 5\
    -gf -minf 4 -maxf 7\
    -rs 2 -re 2\
    -llms -uf -msc 60\
    -maxres 20 -maxpdf 10 -tso\
    --name "KFSDatasets"
"""
