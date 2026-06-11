# DocAI — Project Review
> Reviewed: 2026-06-09 | Reviewer: Claude Code (deep review, static + compile/import verification)
> Verdict: **Pre-alpha skeleton (~25% complete). Not installable, not runnable end-to-end. Code that exists is high quality but the system's spine (indexing, LLM, CLI) is missing.**

---

## 1. Overview

DocAI is specified (PROJECT_BRIEF.md) as a fully local, offline document intelligence system: scan local/network drives → parse → chunk → embed (BGE) → FAISS index → RAG chat (Ollama/Mistral) → 5 agents → Streamlit dashboard → automations/tracking. Brief status says "PAUSED — Hardware procurement required"; the code state matches a build stopped mid-Phase 3.

**What exists (19 .py files, ~3,400 LOC + schema.sql + config):**
- `docai/core.py` — complete, clean data contracts (enums + dataclasses)
- `docai/ingestion/` — scanner, change_detector, safe_copy, file_router; 3 of 9 parsers (PDF, DOCX, PPTX)
- `docai/processing/` — normalizer, chunker, jsonl_writer, metadata_extractor (complete)
- `docai/retrieval/` — retriever, reranker, citation_builder (written, but unimportable — see Bug B1)
- `docai/agents/base_agent.py` — reason-act loop against Ollama (complete base; zero concrete agents)
- `docai/indexing/schema.sql` — full DDL (no Python to use it)
- `config/settings.yaml`, `requirements.txt`, `pyproject.toml`

**Directory purposes (contents not reviewed per scope):** `data/` (workspace/chunks/db/index runtime artifacts), `models/` (finetuned model weights), `logs/` (automations/decisions/issues/tests) — all empty scaffolds.

**Verification results:**
- `python3 -m compileall docai -q` → **PASS** (Python 3.10.8, zero syntax errors)
- `python3 -m pytest tests -x -q` → **"collected 0 items"** — there are no tests at all (`tests/` contains only an empty `fixtures/` dir)
- Import smoke tests: `docai.core`, `docai.processing.chunker`, `docai.agents.base_agent` import OK; `docai.retrieval.retriever` raises `ModuleNotFoundError: No module named 'docai.indexing.embedder'`

---

## 2. Architecture Assessment

The layering is sound and matches the brief: core data contracts → ingestion → processing → indexing → retrieval → LLM/agents, with config-driven construction (each class takes the parsed settings dict). Good choices:

- Canonical dataclasses in `core.py` shared by all layers — clean contracts, no circular deps.
- Per-file error isolation: `FileRouter.route()` traps all parser exceptions into `ParsedDocument.parse_error`; pipeline never halts on a bad file (brief Section 3 requirement met).
- Atomic JSONL writes (`tmp` + `os.replace`), WAL-mode SQLite, content-addressed workspace, lazy parser imports — all appropriate for the local-first design.
- LLM access is correctly local-only: Ollama at `http://localhost:11434` from config; **no hardcoded API keys, no cloud providers anywhere** (verified by grep).

Structural weaknesses:

- **The dependency spine is broken.** Retrieval imports `docai.indexing.{embedder,faiss_index,sqlite_store}`; agents require a `rag_pipeline`; both depend on layers that were never written. Nothing above ingestion/processing can execute.
- **Not a Python package.** No `__init__.py` in `docai/`, `docai/ingestion/`, `docai/ingestion/parsers/`, `docai/processing/`, `docai/indexing/` (works locally only via implicit namespace packages; breaks packaging — Bug B3).
- Two competing manifests: `change_detector.py` and `safe_copy.py` each open their own SQLite connection/table, while `schema.sql` defines a third `manifest` table in the main DB. Three sources of truth for file state; should be one store.
- Scanner computes SHA-256 of every file on every scan, which makes `ChangeDetector`'s mtime fast path pointless (the expensive read already happened). Contradicts brief Section 2 "hash manifest to avoid redundant I/O".
- Resource lifecycle via `__del__` for SQLite connections (change_detector.py:186, safe_copy.py:157) — nondeterministic; no `close()`/context-manager API.

---

## 3. Implementation vs Brief

| Brief section | Status |
|---|---|
| §2 Scanning + network drives | ~90% — scanner with retry/backoff, rate limit, exclusions, network detection. Missing: partial rescan of only changed dirs; redundant-I/O avoidance defeated (see above) |
| §3 File types | **33%** — only PDF/DOCX/PPTX parsers exist; TXT, MD, HTML, XLSX, CSV, image OCR, MATLAB parsers missing yet are dispatched by file_router (Bug B2) |
| §4 Safe copy workspace | ~85% — works; manifest fields slimmer than spec; silent failure path (B7) |
| §5 Normalization & chunking | ~95% — complete, sentence-aware, token-bounded, JSONL output |
| §6 Embeddings + FAISS + SQLite | **0% Python** — only schema.sql exists. No embedder, faiss_index, sqlite_store, index_manager |
| §7 Local LLM / LoRA | **0%** — `docai/llm/__init__.py` is empty (0 bytes); no model_loader/inference/prompt_templates/rag_pipeline/finetuning |
| §8 CLI (`mytool`) | **0%** — `docai/cli/` does not exist; pyproject entry point dangles (B4) |
| §9 Incremental indexing | ~40% — change detection done; index update orchestration absent |
| §10 Five agents | **15%** — base_agent only; zero concrete agents |
| §11 Linking (6 types) | **0%** — `linking/` has only a docstring `__init__.py` |
| §12 Dashboard (7 pages) | **0%** — `dashboard/pages/` and `components/` are empty dirs |
| §13 Automations | **0%** — `automation/` has only a docstring `__init__.py` (YAML task defs exist in settings.yaml) |
| §15–17 Cost/decision/issue tracking | **0%** — `tracking/` is empty (schema tables exist) |
| §18 Test suite (14 categories) | **0%** — no tests, no conftest, no fixtures |

**Overall completeness: ~25%** (by the brief's own ~55-file checklist: ~20 delivered, and several of those can't run because their dependencies don't exist).

---

## 4. Bugs Found (file:line + severity)

**B1 — CRITICAL — `docai/retrieval/retriever.py:17-19`**
Imports `docai.indexing.embedder`, `docai.indexing.faiss_index`, `docai.indexing.sqlite_store` — none exist (`docai/indexing/` contains only schema.sql). Verified `ModuleNotFoundError` on import. The entire retrieval layer (and anything that would use it) is dead code.

**B2 — HIGH — `docai/ingestion/file_router.py:120-142`**
Dispatches to `txt_parser`, `html_parser`, `xlsx_parser`, `csv_parser`, `image_ocr_parser`, `matlab_parser` — none of these modules exist. Every TXT/MD/HTML/XLSX/CSV/image/MATLAB file silently fails (ImportError swallowed into `parse_error` by `route()` at line 60). 6 of 9 supported file types are non-functional with no loud failure.

**B3 — HIGH — missing `__init__.py` (package-wide) + `pyproject.toml:3`**
No `__init__.py` in `docai/`, `docai/ingestion/`, `docai/ingestion/parsers/`, `docai/processing/`, `docai/indexing/`. `[tool.setuptools.packages.find] include=["docai*"]` uses regular package discovery, so the wheel would contain nothing. Additionally `build-backend = "setuptools.backends.legacy:build"` is not a real backend (correct: `setuptools.build_meta`) — `pip install .` fails outright.

**B4 — HIGH — `pyproject.toml:14`**
`mytool = "docai.cli.main:cli"` — `docai/cli/` does not exist. The advertised CLI entry point can never resolve.

**B5 — MEDIUM — `docai/retrieval/citation_builder.py:79-87`**
`_infer_page_hint()` is called with a `FileType` enum (`retriever.py` builds `SearchResult.filetype` as enum); `str(FileType.PDF).lower()` evaluates to `'filetype.pdf'` (verified at runtime), never matching `"pdf"`/`"pptx"`. Page/slide hints are always empty. Fix: use `.value`.

**B6 — MEDIUM — `requirements.txt`**
Missing `requests` (used by `docai/agents/base_agent.py:17` — the only HTTP client actually used; the pinned `ollama` package is never imported) and missing `pdf2image` (used by `docai/ingestion/parsers/pdf_parser.py:150`). On a clean install, agents crash on import and the scanned-PDF OCR fallback always returns "OCR dependencies not available" — i.e., the brief's scanned-PDF requirement silently fails.

**B7 — MEDIUM — `docai/ingestion/safe_copy.py:90-95`**
On copy failure, `copy_if_needed()` logs and returns the *intended* destination path that does not exist, with no error indication. Downstream parsing of a nonexistent file then fails with a confusing error far from the cause.

**B8 — MEDIUM — `docai/ingestion/scanner.py:157-169`**
Symlinked directories are resolved and recursed: (a) symlink cycles cause repeated re-scanning bounded only by `max_depth=50`; (b) symlinks escape the scan root, contradicting the inline comment "Avoid following symlinks that point outside the scan root" — nothing enforces that.

**B9 — LOW — `docai/ingestion/scanner.py:199` vs `docai/ingestion/change_detector.py:114`**
`_build_record()` hashes every file unconditionally, so the mtime fast path in `get_changes()` saves zero I/O. Full-corpus SHA-256 read on every scan; on network drives this is the dominant cost the brief explicitly wanted avoided.

**B10 — LOW — `docai/indexing/schema.sql:8-9`**
`source_files` PK is `file_hash`: two identical files at different paths collapse into one row (last writer wins on `original_path`). Duplicate-heavy corpora lose provenance.

**B11 — LOW — `docai/agents/base_agent.py:53-56`**
`_TOOL_CALL_RE` arg group `[^)]*` truncates any tool argument containing `)` — e.g. `[TOOL: search(reliability (MTBF) data)]` loses everything after `(MTBF`.

**B12 — LOW — `docai/processing/chunker.py:144-146, 175-177`**
`token_count` is `len(buffer_tokens)` but emitted text is `" ".join(buffer_sentences)` whose re-encoded length differs (join spaces, decoded-overlap re-tokenization). Counts drift slightly from reality; also overlap seeding decodes/re-splits tokens mid-word at chunk boundaries.

**B13 — LOW — hygiene**
`pdf_parser.py:10` unused `import io`; `pptx_parser.py:44-45` unused `Pt`, `MSO_SHAPE_TYPE`; `pdf_parser.py:158,162` hardcode `dpi=300`, `lang="eng"` ignoring `parsing.ocr_dpi`/`ocr_language` config; `base_agent.py` docstring says +0.05/result "capped at 0.95" while code caps the bonus at 0.15 (line 431); `metadata_extractor.py` docstring claims "no file-system reads" but lines 121-124 call `os.path.getsize`; `change_detector.py:186`/`safe_copy.py:157` close DB via `__del__`.

### LLM integration status
- Provider: **Ollama only** (local, `http://localhost:11434` from `settings.yaml`), model tag `mistral`. **No hardcoded keys, no cloud calls** — the offline guarantee holds in code as written.
- Gaps: the whole `docai/llm/` layer is an empty `__init__.py`; `base_agent._call_llm()` (lines 318-364) is the only LLM client — non-streaming despite `stream_response: true` config, 120 s timeout hardcoded, no retry, no health/`ollama list` preflight, ignores `context_window`/`repetition_penalty` config, and uses raw `requests` instead of the pinned `ollama` client. The config's `backend: ollama | transformers` switch has no implementation.

---

## 5. Production Readiness Gaps

Blunt assessment: **not production-ready; not even demo-ready.** A staff engineer would classify this as a partial Phase-2/3 checkpoint.

1. **Cannot install** — broken build backend, missing `__init__.py`, dangling `mytool` entry point (B3/B4).
2. **No executable entry point** — no CLI, no pipeline orchestrator wiring scanner → copy → parse → chunk → (missing) index. There is no way to run DocAI.
3. **No indexing layer** — without embedder/FAISS/sqlite_store, nothing is searchable; retrieval/agents are dead code.
4. **No RAG pipeline / no concrete agents / no dashboard / no automations / no tracking** — the user-visible product surface is 0%.
5. **Zero tests** against a brief that demands a 14-category suite; no CI config.
6. **Dependency manifest doesn't match imports** (B6) — clean-env install would not run even the parts that exist.
7. **Operational gaps** — no logging setup (modules get loggers but nothing configures handlers/`log_file` from config), no settings.yaml loader utility (every class assumes a pre-parsed dict; nothing parses it), no README.
8. **Concurrency** — `check_same_thread=False` SQLite connections shared without locks; fine single-threaded, unsafe once the brief's multiprocessing parsing (Section 14) appears.

---

## 6. Feature Recommendations

Priority order to reach a minimal working product (sequenced so each step is testable):

1. **Fix packaging first** (B3/B4/B6): add all `__init__.py`, correct build backend, add `requests` + `pdf2image` to requirements, point or remove the `mytool` script. Cheap, unblocks everything.
2. **Build `docai/indexing/`** (sqlite_store, embedder, faiss_index, index_manager) — it's the single dependency that revives the already-written retrieval layer. Use the existing schema.sql.
3. **Write the 6 missing parsers** — txt/csv/html are <50 lines each; this triples supported file types instantly.
4. **One thin orchestrator + minimal CLI** (`run`, `search`, `chat`) before dashboard/agents — the brief's `mytool run` is the keystone deliverable.
5. **RAG pipeline in `docai/llm/`** reusing `base_agent._call_llm` logic, lifted into a shared `OllamaClient` with streaming + health check.
6. **Tests with the code, not after** — start with parser fixtures and chunker/normalizer unit tests (pure functions, no model downloads needed).
7. Defer: reranker tuning, LoRA fine-tuning, dashboard polish, 6-type linking — none have value until search works.
8. Worth adding beyond the brief: a `docai doctor` command (checks tesseract/poppler/Ollama availability) — given how many optional deps degrade silently, operators need a preflight.

---

## 7. Cleanup Actions

- Remove unused imports: `pdf_parser.py:10` (`io`), `pptx_parser.py:44-45` (`Pt`, `MSO_SHAPE_TYPE`).
- Delete or implement the dangling `mytool` entry point in `pyproject.toml:14`.
- Either remove the unused `ollama` pin from requirements.txt or migrate `base_agent._call_llm` to it (and then drop the undeclared `requests` dependency).
- Consolidate the three manifest stores (change_detector DB, safe_copy DB, schema.sql `manifest` table) into the single main database.
- Replace `__del__`-based connection cleanup with explicit `close()`/context managers (change_detector.py:186, safe_copy.py:191, safe_copy.py:157).
- Empty placeholder dirs (`docai/tracking/`, `docai/dashboard/pages/`, `docai/dashboard/components/`, `tests/fixtures/`) need `.gitkeep` or content — currently they won't even survive a git clone.
- Align docstrings with behavior: base_agent confidence cap (line 431 vs docstring), metadata_extractor "no file-system reads" claim, change_detector fast-path I/O claim.
- Wire `parsing.ocr_dpi` / `parsing.ocr_language` config into `pdf_parser._ocr_fallback` instead of hardcoded values.
- DocAI is untracked in the parent repo (git status shows `?? DocAI/`); decide whether it gets its own repo or is committed here, and gitignore `data/`, `logs/`, `models/`.
