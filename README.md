# Paper2PPT - An Agentic Workflow for Presentation Generation

# Author
Aditya Bagri  
Email: adityabagrii.work@gmail.com  
Academic: aditya22029@iiitd.ac.in
-----
Agentic CLI that turns arXiv papers, local PDFs and/or PDF URLs into Beamer slide decks using LLMs. It supports query-guided presentations, optional web search with citations, figure insertion (arXiv only), speaker notes, and multi-source synthesis.

It also support Topic-based, where you give a topic as an input, based on web search results for the same, LLM prepares a slide deck on that topic.

## Highlights
- arXiv, local PDF, and PDF URL inputs (single or multiple)
- Query-guided decks that answer a user question (not just summaries)
- Optional web search with citations
- Speaker notes, figure suggestions, and diagram generation (Graphviz)
- Claim→evidence alignment for reviewer-safe slides
- Comparison-first visuals and baseline framing (optional)
- Robust slide generation with retries and interactive fallbacks
- Organized run directories with logs, outlines, and resume support
- Topic based research using web search and slide deck generation using scholarly articles as references for the same.
- New non-slide “chat” modes: reading notes, viva prep, experiment descriptions, exam prep, implementation notes
- Teaching mode for slide pacing with pause questions
- Persistent paper memory (local index + search) and daily research brief

## Requirements
- Python 3.10+
- NVIDIA NIM API key (LLM + flowcharts)
- `pymupdf` for local PDF parsing
- `pdflatex` for PDF output (optional but recommended)
- Graphviz (`dot`) for flowchart rendering

## Installation
From the repository root:
```bash
cd Paper2ppt
pip install -e .
```

Update dependencies after changes:
```bash
pip install -r requirements.txt
```

Verify:
```bash
paper2ppt --version
```

Help:
```bash
paper2ppt help
```

## Get an NVIDIA API Key
1. Create or sign in to your NVIDIA account.
2. Open the NVIDIA NIM portal and generate a new API key.
3. Copy the key and store it securely.

## Set Your NVIDIA API Key
Export the key in your terminal so the CLI can read it:
```bash
export NVIDIA_API_KEY="YOUR_KEY_HERE"
```

To verify it is set:
```bash
echo $NVIDIA_API_KEY
```

To persist across sessions:
```bash
echo 'export NVIDIA_API_KEY="YOUR_KEY_HERE"' >> ~/.zshrc
source ~/.zshrc
```

If you use bash, replace `~/.zshrc` with `~/.bashrc`.

## Install `PDFLaTeX`

### macOS
Option 1 (Homebrew, minimal):
```bash
brew install --cask basictex
```
Then restart your terminal.

Option 2 (full distribution):
```bash
brew install --cask mactex
```

### Windows
Option 1 (recommended, MiKTeX):
1. Download and install MiKTeX from the official site.
2. Make sure the MiKTeX `bin` folder is on your PATH.
3. Open a new terminal and run `pdflatex --version` to verify.

Option 2 (full distribution, TeX Live):
1. Install TeX Live.
2. Add the TeX Live `bin` directory to PATH.

## Install Graphviz (Flowcharts)
Graphviz is required to render flowcharts into PNGs.

### macOS
```bash
brew install graphviz
```

### Windows
1. Download and install Graphviz from the official site.
2. Ensure `dot` is on your PATH.
3. Verify with: `dot -V`

## Quick Start
Basic arXiv run:
```bash
paper2ppt \
  -a 2602.05883 \
  --slides 10 \
  --bullets 4
```

Local PDF run:
```bash
paper2ppt \
  -p "/path/to/paper.pdf" \
  --slides 10 \
  --bullets 4
```

PDF URL run:
```bash
paper2ppt \
  -u "https://example.com/paper.pdf" \
  --slides 10 \
  --bullets 4
```
If a PDF URL fails to download, the CLI will prompt you to either skip that URL or quit.

Query-guided run (web search enabled by default):
```bash
paper2ppt \
  -a 1811.12432 \
  --query "Compare this approach to prior work" \
  --slides 10 \
  --bullets 4
```

Multiple arXiv IDs:
```bash
paper2ppt \
  -a "1811.12432,1707.06347" \
  --query "Compare approaches the different approaches used in these papers" \
  --slides 12 \
  --bullets 4
```

Multiple PDFs from a directory:
```bash
paper2ppt \
  -d "/path/to/pdfs" \
  --query "Compare methods across papers" \
  --slides 12 \
  --bullets 4
```

Mixed sources (arXiv + PDFs):
```bash
paper2ppt \
  -a 1811.12432 \
  -p "/path/to/paper.pdf" \
  --query "Compare approaches" \
  --slides 12 \
  --bullets 4
```

## New Modes (Non-Slide + Teaching + Memory)
Paper2ppt now includes several “chat-style” modes that generate structured markdown instead of slides, plus a teaching mode for slide pacing and a persistent paper memory.

**Mode guides (with tutorials)**
- Reading Mode: 1–2 page structured reading notes. See [docs/reading_mode.md](docs/reading_mode.md)
- Viva Mode: defense-focused questions and failure cases. See [docs/viva_mode.md](docs/viva_mode.md)
- Experiment Description Generator: datasets, baselines, metrics, protocol. See [docs/experiment_description.md](docs/experiment_description.md)
- Exam Prep Generator: MCQs, short answers, derivations, trick questions. See [docs/exam_prep.md](docs/exam_prep.md)
- Implementation Notes: components, training loop, losses, gotchas. See [docs/implementation_notes.md](docs/implementation_notes.md)
- Reproduction Checklist: hyperparams, missing details, compute, traps. See [docs/reproduction_checklist.md](docs/reproduction_checklist.md)
- Teaching Mode (slides): intuition-heavy slides + pause questions. See [docs/teaching_mode.md](docs/teaching_mode.md)
- Persistent Paper Memory: local index + search. See [docs/paper_memory.md](docs/paper_memory.md)
- Daily Research Brief: journal-style summary of runs. See [docs/daily_brief.md](docs/daily_brief.md)
- Chat Mode (RAG): interactive Q&A with semantic retrieval over the paper. See [docs/chat_mode.md](docs/chat_mode.md)

**Quick examples**
```bash
# Reading notes (no slides)
paper2ppt -a 2401.12345 --read

# Reading notes + embedded diagrams
paper2ppt -a 2401.12345 --read --generate-flowcharts

# Reading notes + intent-aware DAG diagrams (must include --generate-flowcharts to embed images)
paper2ppt -a 2401.12345 --read --generate-flowcharts --diagram-intent-aware --diagram-style dag

# Chat mode (RAG over the paper)
paper2ppt -a 2401.12345 --chat

# Resume read mode from a prior run (reuse extracted text + summary)
paper2ppt --read --resume /path/to/run

# Viva prep (no slides)
paper2ppt -a 2401.12345 --viva-mode

# Experiment description (no slides)
paper2ppt -a 2401.12345 --describe-experiments

# Exam prep (no slides)
paper2ppt -a 2401.12345 --exam-prep

# Implementation notes (no slides)
paper2ppt -a 2401.12345 --implementation-notes

# Reproduction checklist (no slides)
paper2ppt -a 2401.12345 --repro-checklist

# Teaching mode (slides)
paper2ppt -a 2401.12345 --teaching-mode -s 12 -b 4

# Index + search
paper2ppt --index-paper -a 2401.12345
paper2ppt --search "keyframe selection efficiency"

# Daily research brief
paper2ppt --daily-brief
```

## Streamlit GUI
Launch the interactive GUI to configure inputs and run the pipeline:
```bash
cd Paper2ppt
streamlit run gui_streamlit.py
```
The GUI supports arXiv IDs, local PDFs, PDF directories, PDF URLs, and file uploads.
You can save a default root directory from the sidebar for future runs.
For flowchart generation, set `NVIDIA_API_KEY` in your environment.

## Flowchart & Diagram Generation (Graphviz)
Paper2ppt can generate **Graphviz flowcharts** for key slides to deepen understanding of methods and system internals.
The LLM decides the flowchart **structure** (linear/branch/cycle) and **step count** per slide, but you can enforce a **diagram style** to keep visuals consistent across decks.
- Linear - A straight one-way flow-chart.
- Branched - Where a cell in the flow-chart can have multiple inputs/outputs.
- Cycle - A flow-chart with loops. 

By default it targets 3–4 flowcharts in a 10‑slide deck (configurable via CLI).

To enable:
```bash
paper2ppt -a 1811.12432 --slides 10 --bullets 4 --generate-flowcharts
```
Flowcharts are saved to `outputs/flowcharts/` and included in slides automatically.

The LLM also proposes **other diagram types** (Graphviz-friendly) per slide, such as:
- Dependency graphs / DAGs
- Hierarchy / taxonomy diagrams
- Decision trees
- Module interaction graphs
- Ablation/result relationship graphs
Use `--diagram-style {flowchart,block,sequence,dag}` to force a predictable diagram style across method slides.
Slides with titles containing keywords like `pipeline`, `architecture`, `framework`, `training`, or `inference` will always receive a diagram.
Use `--diagram-intent-aware` to generate intent-driven, non-linear diagrams (process/comparison/abstraction) after titles are decided.

### New Diagram + Rigor Features
These flags make decks more review-ready and diagram-heavy:

**`--diagram-intent-aware`**
- After slide titles are fixed, the LLM proposes 5–8 diagrams with explicit intent: `process`, `comparison`, or `abstraction`.
- Diagrams are **non-linear** (not just a chain) and are attached to the most relevant slides.
- Great for turning method text into actual diagrams.

**`--require-evidence`**
- Any bullet with a performance/accuracy/efficiency claim must include evidence.
- The system appends `(source: URL)` or `(evidence: Slide N - Results)` tags.
- If evidence is missing, bullets are flagged with `[NEEDS EVIDENCE]`.

**`--auto-comparisons`**
- Ensures key comparison slides exist (e.g., “Full Video vs Key Frames” and “Uniform Sampling vs Learned Selection”).
- Strengthens persuasion by forcing contrast.

**`--baseline-framing`**
- On experiment/result slides, injects:
  - “Why this baseline?”
  - “What does it control for?”
- Improves methodological clarity.

**`--quant-results`**
- Adds a quantitative results table slide.
- Extracts concrete numbers from sources into a structured table (Method, Dataset, Metric, Score).

## Topic-Only Research Mode
You can start from a topic instead of providing sources. Paper2ppt will:
- Expand the topic into a detailed research query
- Search the web for relevant sources
- Download available PDFs (arXiv and direct PDF links)
- Build a presentation that answers the topic query

In this mode, the system behaves like a lightweight research agent:
- It rewrites your topic into a focused query (with sub-questions and keywords).
- It asks for user approval and allows feedback to refine the query.
- It gathers a small set of credible sources (optionally restricted to scholarly domains or a custom allowlist).
- It extracts full text, summarizes, and synthesizes a coherent narrative.
- It then builds slides that start from fundamentals and progress to deep technical content, results, limitations, and future directions.

Example:
```bash
paper2ppt --topic "How has Cross-Attention affected the results generated by Vision Language Models?" --slides 15 --bullets 4 --generate-flowcharts
```

### Topic-Only Workflow (Visual)
```text
User Topic
  |
  v
Expand topic -> focused research query (LLM)
  |
  v
User approves/refines query -> saved to work/query.txt and outputs/query.txt
  |
  v
Web search -> collect candidate sources
  |
  v
If few/no results, ask LLM for search queries -> re-run search
  |
  v
Filter sources:
  - arXiv links -> add as arXiv inputs
  - PDF links -> download to work/web_pdfs/
  |
  v
Extract + flatten text per source
  |
  v
Chunk + summarize (LLM)
  |
  v
Generate slide titles (LLM)
  |
  v
Generate slides (LLM)
  |
  v
Optional flowchart/diagram generation (Graphviz)
  |
  v
Render Beamer LaTeX -> Compile PDF
```

### Notes for Topic Mode
- Use `--max-web-results` to limit search breadth.
- Use `--max-web-pdfs` to cap downloads for speed.
- Topic mode stores the expanded query in `outputs/query.txt` and uses it as the deck’s guiding question.
- The approved query is saved to `work/query.txt` and `outputs/query.txt`.
- Use `--topic-scholarly-only` to reduce noise and keep sources to reputable venues (for example - CVPR, ICML, NeurIPS, arXiv, Google Scholar, OpenReview, ACL).
- Use `--domains` to provide a custom domain allowlist (e.g., `arxiv.org`, `openaccess.thecvf.com`, `openreview.net`).
- Use `--must-include` and `--exclude` to keep results on-topic.
- Debug logs:
  - LLM-suggested search queries are printed to console.
  - Web results are printed and saved to `outputs/topic_web_results.txt` with ranking reasons (venue/recency/citation hints).
  - `progress.json` is updated throughout the run for reliable resume.

### What Happens Under the Hood (Topic Mode)
1. **Topic expansion (LLM):** Your topic is expanded into a research-grade query with key sub-questions and keywords; you can approve or refine it.
2. **Source discovery:** Web search collects candidate sources; if few results, the LLM proposes keyword queries and the search is retried. Optional scholarly-only filtering or custom domain allowlists keep sources reputable and focused.
3. **Source acquisition:** arXiv links are downloaded as LaTeX sources; PDFs are fetched into `work/web_pdfs/`.
4. **Text extraction:** LaTeX is flattened; PDFs are parsed into text (plus image references).
5. **Summarization:** The corpus is chunked and summarized, then merged.
6. **Narrative planning:** Slide titles are generated to cover motivation → methods → results → limitations → future work.
7. **Slide generation:** Each slide is generated with bullets, speaker notes, and flowchart suggestions.
8. **Flowcharts (optional):** Graphviz diagrams are rendered for mechanism-heavy slides.
9. **Render & compile:** Beamer LaTeX is written and compiled to PDF (if `pdflatex` is available).

## Multi-PDF and Multi-Source Workflow
When you provide multiple arXiv IDs and/or multiple PDFs, Paper2ppt:
- Parses each source separately
- Prints a source list with titles and paths/IDs
- Merges all extracted content into a single summarization pipeline
- Generates a unified deck that answers the user query across sources

Source input options:
- Repeatable args: `-p file1.pdf -p file2.pdf`
- Comma-separated lists: `-a "1811.12432,1707.06347"`
- Directory scanning: `-d "/path/to/pdfs"`
- Direct URLs: `-u "https://example.com/paper.pdf"`
- Mixed inputs: any combination of `-a`, `-p`, `-d`, and `-u`

Notes:
- Local PDF parsing uses text extraction (no OCR). Scanned PDFs with no embedded text require OCR.
- Figure insertion is only supported for a single arXiv source.

## Default Directories
By default, Paper2ppt stores all runs under:
`~/paper2ppt_runs/<paper_title_slug>/`

Inside that folder it creates:
- `work/` for intermediate files (downloaded arXiv source, flattened TeX, extracted PDF images)
- `outputs/` for final artifacts

Example output structure:
```text
~/paper2ppt_runs/Adaptive_Frame_Interpolation_for_Fast_Video_Processing/
  work/
    arxiv_1811.12432/...
    pdf_<name>/pdf_images/...
  outputs/
    flowcharts/
    Adaptive_Frame_Interpolation_for_Fast_Video_Processing.tex
    Adaptive_Frame_Interpolation_for_Fast_Video_Processing.pdf
    run.log
    query.txt
    outline-1.json
    outline-2.json
```

Override the default root:
```bash
paper2ppt --root-dir "/path/to/runs" -a 1811.12432 --slides 10 --bullets 4
```

Set a default root once:
```bash
export PAPER2PPT_ROOT_DIR="/path/to/runs"
```

Override work/output directories directly:
```bash
paper2ppt --work-dir "/tmp/p2p_work" --out-dir "/tmp/p2p_outputs" -a 1811.12432 --slides 10 --bullets 4
```

Notes on structure:
- If `--root-dir` or `PAPER2PPT_ROOT_DIR` is used, each run gets its own subfolder named after a slugified paper title.
- If `--work-dir` or `--out-dir` is set, those paths are used directly and no run subfolder is created.
- For local PDFs, extracted images are saved under `work/pdf_images/` and their paths are included in the LLM input.
- For query-guided runs, the user query is saved to `outputs/query.txt` and web sources appear in the References slide.
- For multi-source runs, all PDFs and arXiv titles are listed in the console output, and the deck title is generated from the user query plus source titles.

## Outputs
- `work/` for intermediate files
- `outputs/` for final outputs
  - `<paper_title>.tex`
  - `<paper_title>.pdf` (if `pdflatex` is installed)
  - `run.log` (full run log)
  - `outline-1.json`, `outline-2.json`, ... (all outline drafts)

## CLI Options
- `-a`, `--arxiv` arXiv link or ID (repeatable or comma-separated list)
- `-p`, `--pdf` path to a local PDF (repeatable or comma-separated list)
- `-d`, `--pdf-dir` directory containing PDFs (repeatable)
- `-u`, `--pdf-url` direct PDF URL (repeatable or comma-separated list)
- `-s`, `--slides` number of slides (default `12`)
- `-b`, `--bullets` bullets per slide (default `4`)
- `-q`, `--query` user query to guide the presentation theme (enables web search by default)
- `-n`, `--name` custom run name for the output directory
- `-ws`, `--no-web-search` disable web search even if `--query` is provided
- `-rs`, `--retry-slides` retry count for slide generation (default `3`)
- `-re`, `--retry-empty` retry count for empty LLM outputs (default `3`)
- `-I`, `--interactive` enable interactive checkpoints to allow aborting
- `-ci`, `--check-interval` how often to prompt during interactive runs (default `5`)
- `-r`, `--resume` resume from a previous run directory or outputs directory
- `--titles-only` stop after slide titles (skip slide generation)
- `-t`, `--topic` research a topic and build a deck from web + PDFs
- `-maxres`, `--max-web-results` max web results to consider in topic mode (default `6`)
- `-maxpdf`, `--max-web-pdfs` max PDFs to download in topic mode (default `4`)
- `-tso`, `--topic-scholarly-only` restrict topic mode to scholarly sources (arXiv/CVPR/ICML/NeurIPS/Scholar)
- `--must-include` keyword(s) that must appear in sources (repeatable)
- `--exclude` keyword(s) to exclude from sources (repeatable)
- `--domains` allowlist domains for topic mode (repeatable or comma-separated)
- `-gf`, `--generate-flowcharts` generate Graphviz flowcharts for key slides
- `-gi`, `--generate-images` alias for `--generate-flowcharts`
- `-minf`, `--min-flowcharts` minimum flowcharts per deck (default `3`)
- `-maxf`, `--max-flowcharts` maximum flowcharts per deck (default `4`)
- `--diagram-style` force diagram style for method slides (`flowchart|block|sequence|dag`)
- `--diagram-intent-aware` generate intent-driven diagrams after titles
- `--require-evidence` flag ungrounded claims and require evidence tags
- `--auto-comparisons` auto-add comparison slides (e.g., full video vs key frames)
- `--baseline-framing` add baseline framing bullets on experiment slides
- `--quant-results` add a quantitative results table slide (numbers pulled from sources)
- `--root-dir` root directory for all runs (default `$PAPER2PPT_ROOT_DIR` or `~/paper2ppt_runs`)
- `-wdir`, `--work-dir` working directory (overrides `--root-dir`)
- `-odir`, `--out-dir` output directory (overrides `--root-dir`)
- `-msc`, `--max-summary-chunks` cap for LLM summary chunks (default `30`)
- `-workers`, `--max-llm-workers` max parallel LLM calls (default `4`)
- `-na`, `--no-approve` skip outline approval loop
- `-llms`, `--skip-llm-sanity` skip LLM sanity check
- `-m`, `--model` NVIDIA NIM model name
- `-uf`, `--use-figures` enable figure selection and insertion (single arXiv source only)
- `-wsn`, `--with-speaker-notes` generate speaker notes for each slide
- `-v`, `--verbose` verbose logs
- `--read` generate reading notes (no slides)
- `--viva-mode` generate viva prep notes (no slides)
- `--describe-experiments` generate experiment description (no slides)
- `--exam-prep` generate exam prep materials (no slides)
- `--implementation-notes` generate implementation notes (no slides)
- `--repro-checklist` generate a reproduction checklist (no slides)
- `--teaching-mode` teaching-optimized slides with pause questions
- `--index-paper` index a paper into local memory
- `--search` search the local paper index
- `--daily-brief` generate a daily research brief
- `--chat` chat with the paper using stored context (RAG)
- `--version` show version and exit

## Use Cases
- Quick paper summary for a talk or class
- Comparative literature review across multiple papers
- Query-driven decks like "Compare methods" or "What are the tradeoffs?"
- Generate speaker notes for rehearsals

## Interactive Feedback and Resume
- With `--interactive`, the CLI pauses at key stages and accepts optional guidance text.
- This guidance is injected into slide title and slide generation prompts.
- Resume a stopped run with `--resume /path/to/run` (or `.../outputs`).
- If the model returns the wrong number of slide titles, the CLI can display the current titles and accept user feedback before auto-fixing.

### Title Mismatch Handling
When the LLM returns the wrong number of slide titles:
1. Paper2ppt re-prompts the LLM to fix the count.
2. If still mismatched and `--interactive` is enabled, it prints the current titles and asks for guidance.
3. It then retries with your feedback or falls back to padding/truncation to meet the exact count.

### Resume Flow
When `--resume` is provided, the run loads `outputs/progress.json`, restores titles and slides generated so far, and continues from the next missing slide.

## Workflow Diagram
```text
User CLI
  |
  v
Parse args + Get API Keys from OS Environment
  |
  v
Initialize LLM
  |
  v
Sanity checks for the LLMs
  |
  v
Collect sources (arXiv/PDFs/URLs) or Get the topic -> search queries
  |
  v
Extract + flatten text per source
  |
  v
Chunk + summarize (LLM)
  - Multi-source: chunk summarization
  |
  v
Generate slide titles (LLM)
  |
  v
Generate slides (LLM)
  |
  v
Optional approval loop (user feedback -> LLM updates)
  |
  v
Optional flowcharts (Graphviz)
  |
  v
Render Beamer LaTeX
  |
  v
Compile PDF
```

## Project Directory
```text
Paper2ppt/
  arxiv_utils.py
  llm.py
  logging_utils.py
  main.py
  models.py
  pdf_utils.py
  pipeline.py
  gui_streamlit.py
  tex_utils.py
  web_utils.py
  docs/
    reading_mode.md
    viva_mode.md
    experiment_description.md
    exam_prep.md
    implementation_notes.md
    reproduction_checklist.md
    teaching_mode.md
    paper_memory.md
    daily_brief.md
    chat_mode.md
  requirements.txt
  pyproject.toml
  README.md
  CHANGELOG.md
```

### File Overview
- `arxiv_utils.py`: arXiv ID parsing, metadata lookup, and source download/extraction.
- `llm.py`: LLM wrapper for NVIDIA NIM models with safe invoke utilities.
- `logging_utils.py`: central logging configuration (Rich console if available).
- `main.py`: CLI entrypoint and argument parsing.
- `models.py`: Pydantic models for slides/decks (including flowchart specs).
- `pdf_utils.py`: PDF text and image extraction using PyMuPDF.
- `pipeline.py`: core orchestration (sources, summarization, slides, flowcharts, rendering).
- `gui_streamlit.py`: Streamlit GUI for interactive runs.
- `tex_utils.py`: Beamer LaTeX rendering and output writing.
- `web_utils.py`: web search utility for topic mode.
- `requirements.txt`: pinned Python dependencies.
- `pyproject.toml`: packaging metadata and CLI entrypoint.
- `README.md`: usage and documentation.
- `CHANGELOG.md`: version history.

## Maintenance
- After any version upgrade, run: `pip install -r requirements.txt` from the codebase directory.
- Paper2ppt auto-installs dependencies when `requirements.txt` changes, but manual updates are safer for reproducibility.

## Changelog
See `CHANGELOG.md` for version history and changes.

### Optimization Updates (Summary)
- 0.5.5: GUI caches LLM client, de-duplicates uploads/downloads, and uses saved default root for faster setup.
- 0.4.4: Slide generation retries avoid hard failures when JSON is malformed.
- 0.4.2: Logging falls back to temp/console on filesystem timeouts.

### Recent Additions
- 0.7.0: Topic-only research mode with web search, PDF harvesting, and optional diagram generation.

### Date Last Updated - 8th Feb 2026, by Aditya Bagri
