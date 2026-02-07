"""CLI entrypoint for running the Paper2ppt pipeline from the terminal."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import hashlib
import subprocess
from pathlib import Path

from dotenv import load_dotenv

try:
    from .arxiv_utils import extract_arxiv_id, get_arxiv_metadata
    from .llm import LLMConfig, init_llm
    from .logging_utils import setup_logging
    from .pipeline import Pipeline, RunConfig
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent))
    from arxiv_utils import extract_arxiv_id, get_arxiv_metadata
    from llm import LLMConfig, init_llm
    from logging_utils import setup_logging
    from pipeline import Pipeline, RunConfig

logger = logging.getLogger("paper2ppt")
VERSION = "0.5.6"


def _requirements_path() -> Path | None:
    req = Path(__file__).parent / "requirements.txt"
    return req if req.exists() else None


def _requirements_hash(req_path: Path) -> str:
    h = hashlib.sha256()
    h.update(req_path.read_bytes())
    return h.hexdigest()


def ensure_requirements_installed() -> None:
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
    print("")
    print("Full options:")
    print("  paper2ppt --help")


def parse_args() -> argparse.Namespace:
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
    p.add_argument("--slides", "-s", type=int, required=True, help="Number of slides to generate")
    p.add_argument("--bullets", "-b", type=int, required=True, help="Number of bullets per slide")
    p.add_argument("--query", "-q", default="", help="User query to guide the presentation theme")
    p.add_argument("--no-web-search", "-ws", action="store_true", help="Disable web search even if --query is provided")
    p.add_argument("--retry-slides", "-rs", type=int, default=3, help="Retry count for slide generation")
    p.add_argument(
        "--root-dir",
        default=None,
        help="Root directory for all runs (default: $PAPER2PPT_ROOT_DIR or ~/paper2ppt_runs)",
    )
    p.add_argument("--work-dir", "-wdir", default=None, help="Working directory (overrides --root-dir)")
    p.add_argument("--out-dir", "-odir", default=None, help="Output directory (overrides --root-dir)")
    p.add_argument("--max-summary-chunks", "-msc", type=int, default=30, help="Max summary chunks to process")
    p.add_argument("--no-approve", "-na", action="store_true", help="Skip outline approval loop")
    p.add_argument("--skip-llm-sanity", "-llms", action="store_true", help="Skip LLM sanity check")
    p.add_argument("--model", "-m", default="nvidia/llama-3.1-nemotron-ultra-253b-v1", help="NVIDIA NIM model name")
    p.add_argument("--use-figures", "-uf", action="store_true", help="Enable figure selection and insertion")
    p.add_argument("--with-speaker-notes", "-wsn", action="store_true", help="Generate speaker notes for each slide")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return p.parse_args()


def _slugify(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = s.strip("_")
    return (s or "paper").strip()[:max_len]


def _query_summary(query: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", query or "")
    if not words:
        return "Query"
    return "_".join(words[:2])


def _split_list_args(values: list[str]) -> list[str]:
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
    pdfs: list[Path] = []
    for p in _split_list_args(paths):
        pdfs.append(Path(p).expanduser().resolve())
    for d in _split_list_args(dirs):
        dpath = Path(d).expanduser().resolve()
        if dpath.exists() and dpath.is_dir():
            pdfs.extend(sorted(dpath.glob("*.pdf")))
    return pdfs


def _download_pdfs(urls: list[str], out_dir: Path) -> list[Path]:
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
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        print_helper()
        return 0

    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=False)
    args = parse_args()

    ensure_requirements_installed()

    arxiv_inputs = _split_list_args(args.arxiv or [])
    pdf_paths = _collect_pdfs(args.pdf or [], args.pdf_dir or [])
    pdf_urls = _split_list_args(args.pdf_url or [])
    if not arxiv_inputs and not pdf_paths and not pdf_urls:
        logger.error("Provide at least one source via --arxiv, --pdf, --pdf-dir, or --pdf-url.")
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

    root_dir = args.root_dir or os.environ.get("PAPER2PPT_ROOT_DIR", "~/paper2ppt_runs")
    run_root = Path(root_dir).expanduser().resolve()
    run_name = _slugify(paper_title)
    if args.query:
        if len(arxiv_ids) + len(pdf_paths) > 1:
            run_name = f"Q-{_query_summary(args.query)}-MultiSource"
        else:
            run_name = f"Q-{_query_summary(args.query)}-{run_name}"
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
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.user_query:
        (cfg.out_dir / "query.txt").write_text(cfg.user_query + "\n", encoding="utf-8")
    setup_logging(args.verbose, log_path=cfg.out_dir / "run.log")

    try:
        logger.info("Initializing LLM...")
        llm = init_llm(LLMConfig(model=cfg.llm_model, api_key=cfg.llm_api_key))

        pipeline = Pipeline(cfg, llm)
        outline, tex_path, pdf_path = pipeline.run()

        logger.info("Saved TeX: %s", tex_path)
        logger.info("Saved PDF: %s", pdf_path)

        print("\nOutput directory:", cfg.out_dir.resolve())
        print("TeX exists:", tex_path.exists())
        print("PDF exists:", pdf_path.exists() if pdf_path else False)

        return 0
    except Exception:
        logger.exception("Unhandled error in pipeline run")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
