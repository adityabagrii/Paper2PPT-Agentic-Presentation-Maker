# Changelog

## 0.7.3 - 2026-02-09
- Added research-focused multi-source web search with per-provider logging, dedupe, scoring, and caching/backoff.
- Added `--arxiv-only-search` flag to force arXiv-only search.
- Improved topic expansion and diversified query generation with topic-term anchoring.
- Added LLM-based relevance filtering with arXiv metadata (title/abstract) and detailed keep/drop logs.
- Added semantic chunk ranking via embeddings (fallback to keyword overlap) and capped chunk processing by `--max-summary-chunks`.
- Improved read mode formatting and resume-from-outline support.
- Added source approval flow and slide-title approval with optional slide-count adjustment.
- Prefer arXiv source downloads with PDF fallback after 2 failed attempts.
- Reduced noisy HTTP logs (HF hub/requests/httpx) and added progress bars where needed.
- Added LLM-query web search for `--query` runs with configurable `--max-web-queries`.
- Added web-source approval gate after query search.
- Added post-run revision tools: `--revise-with-figures`, `--revise-with-diagrams`, and `--edit-run`.
- Added targets for post-run figures/diagrams (`--revise-figures-target`, `--revise-diagrams-target`).

## 0.7.2 - 2026-02-08
- Added non-slide modes: `--read`, `--viva-mode`, `--describe-experiments`, `--exam-prep`, `--implementation-notes`.
- Added reproduction checklist mode: `--repro-checklist`.
- Added teaching slide mode with pause questions: `--teaching-mode`.
- Added persistent paper memory (`--index-paper`, `--search`) and daily research brief (`--daily-brief`).
- Added read-mode diagrams embedded per section (with flowcharts/diagram generation) and improved formatting.
- Added chat mode with RAG-style retrieval (`--chat`), chat history logging, and streaming output.
- Added semantic embeddings for chat retrieval (computed only in chat mode).
- Added summary caching with TTL and manual purge (`--cache-summary`, `--purge-cache`).
- Improved resume support for read mode (progress.json and paper_context.json).
- Fixed multiple dict-vs-model rendering crashes and improved figure/diagram slide handling.
- Made LLM responses and read-mode generation more robust with per-section generation and progress indicators.
- Updated docs/README with new modes, chat, and read-mode resume usage.
- CLI version now reads installed package metadata.

## 0.7.1 - 2026-02-07
- Added `--topic-scholarly-only` to restrict topic mode to reputable sources.

## 0.7.0 - 2026-02-07
- Added topic-only research mode with web search and automatic PDF collection.
- Topic is expanded into a focused research query (LLM) before search.
- Downloads PDFs into `work/web_pdfs/` for topic runs.

## 0.6.7 - 2026-02-07
- Added `--titles-only` to stop after slide titles for faster runs.

## 0.6.6 - 2026-02-07
- Added robust slide title JSON extraction with retries and fallback placeholder titles.

## 0.6.5 - 2026-02-07
- Added NVIDIA Flux (black-forest-labs/flux.1-kontext-dev) image generation provider and set as default.

## 0.6.4 - 2026-02-07
- Added diagram/image generation from slide figure ideas with optional OpenAI provider.

## 0.6.3 - 2026-02-07
- Print chunk summary previews and slide titles for better terminal visibility.

## 0.6.2 - 2026-02-07
- Interactive handling for slide title count mismatches: show titles, accept feedback, then auto-fix.

## 0.6.1 - 2026-02-07
- Interactive checkpoints now accept user feedback for titles/slides and include it in prompts.
- Added resume support from `progress.json` via `--resume`.

## 0.6.0 - 2026-02-07
- Added interactive checkpoints to allow users to abort at key stages.
- Added `--check-interval` for interactive prompting cadence.

## 0.5.9 - 2026-02-07
- Added `--retry-empty` to control retries for empty LLM outputs.

## 0.5.8 - 2026-02-07
- Retry empty LLM outputs for chunk summaries and prompt user to skip or quit.

## 0.5.7 - 2026-02-07
- Added `--name` to set a custom run directory name.

## 0.5.6 - 2026-02-07
- Robust slide title generation: retry/fix and fallback padding when LLM returns the wrong count.

## 0.5.5 - 2026-02-07
- GUI: Save a default root directory from the sidebar.
- GUI: Reuse cached LLM client between runs to reduce setup time.
- GUI: De-duplicate uploads and URL downloads to avoid redundant work.
- GUI: Read `NVIDIA_API_KEY` from environment automatically.

## 0.5.4 - 2026-02-07
- Parallelized chunk summarization for multi-source runs to reduce latency.

## 0.5.3 - 2026-02-07
- GUI: Live log streaming during presentation generation.

## 0.5.2 - 2026-02-07
- Added Streamlit GUI (`gui_streamlit.py`) for interactive runs.

## 0.5.1 - 2026-02-07
- Added `--pdf-url` for direct PDF links (repeatable or comma-separated).

## 0.5.0 - 2026-02-07
- Added multi-source support: multiple arXiv IDs, multiple PDFs, and PDF directories.
- Pretty-printed outline display for approval.
- Deck title is generated from the user query and source titles.

## 0.4.5 - 2026-02-07
- Added `--retry-slides` to control slide generation retries.

## 0.4.4 - 2026-02-07
- Retry slide generation on malformed JSON and fall back instead of crashing.

## 0.4.3 - 2026-02-07
- Print top web search results for user reference during query-guided runs.

## 0.4.2 - 2026-02-07
- Logging: fall back to temp log or console-only when file log fails.

## 0.4.1 - 2026-02-07
- Added auto-install of updated dependencies when `requirements.txt` changes.

## 0.4.0 - 2026-02-07
- Added `--query` to steer the presentation goal beyond summarization.
- Added automatic web search when `--query` is set (disable with `--no-web-search`).
- Saved user queries to `outputs/query.txt`.
- Added source citations from web search to outline and final references.

## 0.3.0 - 2026-02-07
- Added local PDF input via `--pdf` with text and image extraction for LLM chunking.
- Added PyMuPDF (`pymupdf`) dependency for PDF parsing.
- Local PDFs disable figure insertion (arXiv-only).
- Fixed progress bar width and live chunk preview.

## 0.2.2 - 2026-02-07
- README updated with steps to obtain and set NVIDIA API keys.
- API key check now warns instead of exiting when missing.
- Added `researchos help` helper command.

## 0.2.0 - 2026-02-07
- Added `RESEARCHOS_ROOT_DIR` for a default runs root.
- Clarified output directory structure and precedence of `--root-dir`, `--work-dir`, and `--out-dir`.

## 0.1.0 - 2026-02-07
- Initial class-based refactor of the original PresentationAgent pipeline.
- CLI entry point `researchos`.
- Outline drafts saved to `outline-*.json` in output directory.
- Run logs saved to `run.log` in output directory.
- Optional speaker notes (`--with-speaker-notes`, default off).
