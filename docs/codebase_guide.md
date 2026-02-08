# Paper2ppt Codebase Guide

This document is a structured walkthrough of the Paper2ppt codebase: modules, configuration variables, key classes/functions, modes, and common use cases. It is intentionally practical and aligned with the current repo layout.

## Purpose
Paper2ppt turns arXiv papers, local PDFs, PDF URLs, or a topic into a Beamer slide deck or structured markdown notes using LLMs. It supports web search grounding, flowchart/diagram generation, figure insertion (arXiv sources), speaker notes, and several non-slide “reading” modes.

## Repository Layout
- `main.py`: CLI entrypoint, argument parsing, run directory setup, and orchestration.
- `pipeline.py`: Public facade that re-exports core pipeline classes from smaller modules.
- `pipeline_core.py`: Main orchestration class `Pipeline` and run flow.
- `pipeline_common.py`: Shared dataclasses and helpers (`RunConfig`, `PaperContext`, `OutlineJSONStore`, `_progress_path`, `logger`).
- `pipeline_arxiv.py`: `ArxivClient` wrapper around arXiv utils.
- `pipeline_outline.py`: `OutlineBuilder` and all slide outline generation logic.
- `pipeline_figures.py`: Figure extraction and figure planning with LLM.
- `pipeline_render.py`: Beamer rendering and `pdflatex` compilation.
- `llm.py`: LLM client init and safe invoke with retries.
- `models.py`: Pydantic schema for deck, slides, tables, flowcharts.
- `arxiv_utils.py`: arXiv download + metadata extraction helpers.
- `pdf_utils.py`: PDF text and image extraction with PyMuPDF.
- `tex_utils.py`: TeX parsing/flattening, text extraction, and Beamer rendering.
- `flowchart_utils.py`: Graphviz graph building and rendering.
- `web_utils.py`: DuckDuckGo HTML search parsing for web grounding.
- `memory_utils.py`: Local persistent index, daily journal, and summary cache.
- `logging_utils.py`: Logging setup utilities.
- `gui_streamlit.py`: Streamlit GUI to configure and run pipelines.
- `docs/`: Mode guides and usage documentation.

## Entry Points
- CLI: `main.py` provides `paper2ppt` command, arguments, and run setup.
- Python API: `pipeline.Pipeline` and `pipeline.RunConfig` can be imported and used programmatically.
- GUI: `gui_streamlit.py` launches Streamlit UI for configuration and execution.

## Core Configuration Variables
All runtime configuration is centralized in `RunConfig` (`pipeline_common.py`). These fields map closely to CLI flags in `main.py` and GUI inputs in `gui_streamlit.py`.

RunConfig fields:
- `arxiv_ids`: list of arXiv IDs or URLs after normalization.
- `pdf_paths`: list of local PDF paths.
- `work_dir`: path used for intermediate artifacts.
- `out_dir`: path used for outputs.
- `slide_count`: number of slides to generate.
- `bullets_per_slide`: bullet count per slide.
- `max_summary_chunks`: max LLM chunks for summarization.
- `approve`: whether to run interactive outline approval loop.
- `verbose`: enable debug logging from LLM retries.
- `skip_llm_sanity`: skip LLM “OK” sanity check.
- `llm_model`: model name for ChatNVIDIA.
- `llm_api_key`: NVIDIA API key.
- `use_figures`: enable figure extraction and insertion (arXiv source only).
- `include_speaker_notes`: include speaker notes per slide.
- `user_query`: user query for query-guided decks and web search.
- `web_search`: enable web search for query-guided runs.
- `retry_slides`: retries for slide JSON generation.
- `retry_empty`: retries for empty LLM output.
- `interactive`: prompt at checkpoints to allow aborting.
- `check_interval`: prompt cadence for `interactive`.
- `resume_path`: resume from a previous run directory.
- `generate_flowcharts`: generate diagrams for slides or reading notes.
- `min_flowcharts`: min diagram count per deck.
- `max_flowcharts`: max diagram count per deck.
- `flowchart_structure`: flowchart structure for simple diagrams.
- `flowchart_depth`: depth for diagram structures.
- `titles_only`: stop after generating slide titles.
- `topic`: topic-only mode (collect sources from web).
- `max_web_results`: max web results for topic mode.
- `max_web_pdfs`: max PDFs downloaded in topic mode.
- `topic_scholarly_only`: restrict topic sources to scholarly domains.
- `max_llm_workers`: max parallel LLM calls.
- `diagram_style`: default diagram style.
- `topic_must_include`: keywords required in topic sources.
- `topic_exclude`: keywords to exclude in topic sources.
- `topic_allow_domains`: allowed domains for topic sources.
- `require_evidence`: enforce evidence tags in slide claims.
- `diagram_intent_aware`: generate intent-aware diagrams after titles.
- `auto_comparisons`: insert comparison-focused slides when enabled.
- `baseline_framing`: encourage baseline framing in slides.
- `quant_results`: generate quantitative results table.
- `teaching_mode`: teaching slide mode with pause questions.
- `read_mode`: reading notes (non-slide).
- `viva_mode`: viva preparation notes (non-slide).
- `describe_experiments`: experiment description notes (non-slide).
- `exam_prep`: exam prep notes (non-slide).
- `implementation_notes`: implementation notes (non-slide).
- `reproduction_checklist`: reproduction checklist (non-slide).
- `index_paper`: index paper into local memory store.
- `index_search_query`: query to search in indexed memory.
- `daily_brief`: generate daily journal summary.
- `cache_summary`: enable summary caching (TTL).
- `chat_mode`: interactive paper Q&A with semantic retrieval.

Environment variables:
- `NVIDIA_API_KEY`: required to use ChatNVIDIA LLM.
- `PAPER2PPT_ROOT_DIR`: default root directory for run outputs.

## Data Models
- `models.FlowchartSpec`: steps, structure, caption for a flowchart.
- `models.TableSpec`: title, columns, rows.
- `models.SlideSpec`: slide title, bullets, speaker notes, figure suggestions, images, diagrams, tables.
- `models.DeckOutline`: deck title, arXiv ID, slides, citations.
- `pipeline_common.PaperContext`: normalized context for the paper(s), summary, sources, and metadata.

## Pipeline Architecture
Main classes:
- `ArxivClient` (`pipeline_arxiv.py`): metadata lookup and source download.
- `OutlineBuilder` (`pipeline_outline.py`): core text processing and LLM prompting, including outline generation, summaries, flowchart planning, and validation.
- `FigurePlanner` (`pipeline_figures.py`): extract figures from TeX, ask LLM to match figures to slides, materialize chosen figures in `outputs/figures`.
- `Renderer` (`pipeline_render.py`): render Beamer `.tex`, run `pdflatex`, return `.pdf`.
- `Pipeline` (`pipeline_core.py`): orchestrates the full flow. Builds paper context, generates outlines, notes, diagrams, and outputs.

Core flow in `Pipeline.run()`:
1. `sanity_checks()` validates inputs, paths, and LLM availability.
2. `prepare_topic_sources()` if `topic` mode is enabled.
3. Build paper context with `build_paper_context()`.
4. Generate either slides or non-slide notes depending on mode flags.
5. If slides, render LaTeX and optionally PDF.

## Modes
All modes are CLI flags in `main.py` and flow through `RunConfig`.

Slide modes:
- Default slide mode: generates a Beamer deck with `slides` and `bullets` counts.
- `--teaching-mode`: generates slides with extra pause questions for teaching.
- `--titles-only`: stops after slide titles (fast iteration).
- `--use-figures`: inserts arXiv figures (TeX source only).
- `--generate-flowcharts`: generates flowchart images and embeds them in slides.

Non-slide modes:
- `--read`: reading notes with sections (Problem, Key Idea, Method, Results, Limitations, What I Learned).
- `--viva-mode`: viva preparation questions and failure cases.
- `--describe-experiments`: experiment setup summaries.
- `--exam-prep`: MCQs and derivations.
- `--implementation-notes`: implementation and training notes.
- `--repro-checklist`: reproduction checklist for a paper.
- `--daily-brief`: journal-style run summary.
- `--chat`: interactive RAG-style Q&A over the paper.

Topic-only mode:
- `--topic`: expand a topic into a query, run web search, collect PDFs, then generate a deck.

Memory modes:
- `--index-paper`: store summary and metadata in local JSON index.
- `--search`: query the index for relevant papers.

## Outputs and Artifacts
All outputs are placed under `out_dir` (usually `<root>/<run>/outputs`).

Typical outputs:
- `slides.tex`: generated Beamer source.
- `slides.pdf`: compiled deck if `pdflatex` is available.
- `outline-*.json`: outline snapshots for debugging and resume.
- `reading_notes.md`: non-slide reading notes when `--read` is set.
- `viva_notes.md`, `exam_prep.md`, `implementation_notes.md`, `reproduction_checklist.md`, `experiment_description.md`.
- `flowcharts/*.png`: generated diagrams and flowcharts.
- `paper_context.json`: cached context, summary, chunks (used for resume and chat).
- `progress.json`: incremental stage output for resume.

## Web Search and Citations
- `web_utils.search_web()` uses DuckDuckGo HTML results to fetch title/snippet/url.
- When `--query` is set and `--no-web-search` is not set, web sources are appended as citations in outlines and outputs.

## Caching and Resume
- `summary_cache.json` in `~/.paper2ppt` stores summaries keyed by hash.
- `progress.json` tracks current stage, summary, and context to support `--resume`.
- `paper_context.json` provides lightweight RAG chunks and summary for `--chat` and resume.

## GUI
`gui_streamlit.py` exposes most CLI settings in a Streamlit UI and runs the same `Pipeline` logic. It supports:
- arXiv IDs, local PDFs, PDF URLs, and uploads.
- Cache and run directory selection.
- Live log streaming during runs.

## Common Use Cases
- Basic arXiv deck: `paper2ppt -a 2401.12345 --slides 12 --bullets 4`
- Multi-source comparison: `paper2ppt -a 1811.12432 -p /path/paper.pdf --query "Compare methods"`
- Topic-based research: `paper2ppt --topic "Video summarization" --slides 12`
- Reading notes: `paper2ppt -a 2401.12345 --read`
- Viva prep: `paper2ppt -a 2401.12345 --viva-mode`
- Teaching mode: `paper2ppt -a 2401.12345 --teaching-mode -s 12 -b 4`
- Index a paper: `paper2ppt --index-paper -a 2401.12345`
- Chat with a paper: `paper2ppt -a 2401.12345 --chat`

## Extension Points
If you want to extend the codebase, the most common entry points are:
- Add a new non-slide mode: implement a new `Pipeline.generate_*` method in `pipeline_core.py`, add a CLI flag in `main.py`, and add a docs page in `docs/`.
- Add a new diagram type: extend `flowchart_utils.py` and update diagram prompts in `pipeline_core.py`.
- Add a new LLM backend: add a new init in `llm.py` and surface it via `LLMConfig`.

## Quick File Reference
- `pipeline_common.py`: shared dataclasses and constants.
- `pipeline_outline.py`: outline generation, text chunking, summary, slide title logic, and approval flow.
- `pipeline_core.py`: orchestrates modes, resume, chat, diagrams, and outputs.
- `tex_utils.py`: Beamer rendering and TeX parsing.

## Known Assumptions
- LLM access requires `langchain_nvidia_ai_endpoints` and `NVIDIA_API_KEY`.
- `pdflatex` and Graphviz are optional but required for PDF and diagram rendering.
- Local PDFs are parsed with PyMuPDF and do not support arXiv figure insertion.
