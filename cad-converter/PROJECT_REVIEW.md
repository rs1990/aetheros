# cad-converter — Deep Review

_Reviewed: 2026-06-09. Scope: full source read, boot test, security & production-readiness audit. No source code was modified._

## Overview

A 2D-CAD → 3D-model converter. FastAPI backend extracts a structured `PartSpec` from uploaded drawings (raster image / PDF via Claude Vision, DXF/DWG via `ezdxf` heuristics), lets the user clarify/edit dimensions, then builds a solid with CadQuery and exports STL/STEP/OBJ plus per-face metadata. Vanilla-JS + Three.js frontend drives a 4-step wizard (upload → clarify → edit spec → preview/download) with a clickable surface inspector.

Pipeline endpoints: `POST /api/extract`, `POST /api/clarify`, `POST /api/build`, `GET /api/model/{id}/faces`, `GET /api/model/{id}/{fmt}`. Sessions are persisted as directories under `uploads/<uuid>/`.

**Boot test:** `python3 -m py_compile backend/*.py` passes. App boots cleanly (`uvicorn backend.main:app`) — `GET /` and `/openapi.json` return 200. Core deps (fastapi, uvicorn, anthropic, trimesh, pydantic, dotenv) are installed; **CadQuery is NOT installed** (`import cadquery` → `ModuleNotFoundError: No module named 'OCP'`), so `/api/build` will fail at runtime in this environment. `ezdxf` and `pypdfium2` import checks were inconclusive/absent — treat as not guaranteed present.

## Architecture Assessment

Solid for a prototype. Good separation: `models.py` (pydantic schema, discriminated unions), `extractor.py` (input → spec), `builder.py` (spec → solid), `main.py` (transport). Correctly runs the non-thread-safe CadQuery work in a `ProcessPoolExecutor` (builder.py:5 documents this). Spec is the single source of truth and is re-validated at each hop with `PartSpec.model_validate`. Frontend is dependency-light and readable.

Weaknesses: no persistence layer beyond loose files; no auth; session state is filesystem-only (won't scale horizontally); deprecated FastAPI `on_event` hooks; CORS wide open; no test suite, no CI, no Dockerfile/render.yaml. The DXF path is a crude heuristic (sorts dimension values by magnitude to guess W/D/H — see Bugs).

## Bugs Found (file:line + severity)

1. **HIGH — Path traversal / arbitrary file write via `file.filename`.** `main.py:82` `(session_dir / file.filename).write_bytes(...)`. The uploaded filename is used unsanitized. A filename like `../../evil.txt` or an absolute path escapes the session directory (`Path("/a/b") / "/etc/x"` yields `/etc/x`). Same untrusted name flows to `extractor.py:352` `Path(f"/tmp/{filename}")`. Sanitize to a basename / fixed name. (Also a security issue — see below.)

2. **HIGH — Unvalidated `session_id` path segments.** `main.py:104,121,149,168` build `UPLOADS_DIR / session_id` (or `/ req.session_id`) directly from client input. `..` segments allow probing/serving files outside `uploads/`. The `.exists()` guards limit impact, but `get_model`/`get_faces` could serve arbitrary `model.stl`/`faces.json` paths if crafted. Validate `session_id` as a UUID before use.

3. **MED — `await file.read()` loads entire upload into memory before the size check.** `main.py:75-77`. The 20 MB cap is enforced only *after* the full body is read, so the limit doesn't actually protect memory; a large body is already resident. Stream and check incrementally, or rely on a server/ingress body-size limit.

4. **MED — Empty / missing filename not handled.** `main.py:82`: if `file.filename` is empty or `None`, `session_dir / ""` writes to the directory path itself (raises) or behaves unexpectedly; `Path(filename).suffix` at extractor.py:344 returns `""` → "Unsupported file type". No explicit 400 for missing filename.

5. **MED — DXF dimension heuristic is unreliable.** `extractor.py:261-264` sorts all DIMENSION measurements descending and assigns `[0]=width, [1]=depth, [2]=height`. Real drawings list dimensions in arbitrary order and include non-extent dims (hole dia, spacing), so W/D/H assignment is frequently wrong. `depth = bbox_w * 0.6` (line 263) is an arbitrary fallback. Functional but low-fidelity; flagged in notes, acceptable for a prototype but not production.

6. **MED — `response.content[0].text` assumes block 0 is text.** `extractor.py:171`. If the model returns a non-text first block (e.g. tool/thinking block) this raises `AttributeError`, which is not caught by the `(json.JSONDecodeError, ValueError)` handler at line 181 and would surface as a 500. Iterate to find the text block.

7. **LOW — Unverified model ID.** `extractor.py:153` uses `model="claude-sonnet-4-6"`. I could not verify this is a valid current model identifier (the claude-api reference skill was not accessible during review). If the ID is wrong, every extract call 404s from the API. Verify against the current Anthropic model list before relying on it.

8. **LOW — Frontend feature/pattern row index desync.** `app.js:308-312` `addFeature` uses `container.children.length` as the new index; after a `removeFeature` + `reindexFeatureRows` (line 319) the dataset indices are renumbered but element `id`s inside rows (`feat[idx].*`, set in `featureFields`) are NOT renumbered. After removing a middle feature and adding a new one, `collectSpec` (app.js:407-422) reads `feat[idx].*` ids that no longer match `row.dataset.idx`, producing wrong/zeroed values. `removePattern` (line 370) doesn't reindex at all.

9. **LOW — `onFeatureTypeChange` string-splitting is fragile.** `app.js:304` rebuilds the row by splitting on the literal `"</select></label>"`. The face `<select>` also ends with `</select></label>`, so for feature types that include a face select this split keeps only the *type* select markup correctly only because it's first — but any future label reordering silently corrupts the row. Brittle DOM manipulation.

10. **LOW — `controls._isDragging` is a private Three.js field.** `app.js:538` relies on an undocumented internal of OrbitControls to suppress click-after-drag. May be `undefined` in three@0.170.0, making the guard a no-op (stray face selections after orbiting).

11. **LOW — Build response URLs are returned before files are confirmed.** `main.py:139-144` returns `stl_url`/etc. after `run_in_executor` completes; fine, but `build_and_export` swallows nothing — if export partially fails the function raises and is caught (500). No bug per se, but the OBJ export depends on `trimesh.load` of the just-written STL (builder.py:306) with no existence check.

12. **LOW — `_cleanup_loop` uses `mtime`, not creation time.** `main.py:62-64`. Re-saving `spec.json` on clarify/build (lines 91/115/125) bumps the session directory's mtime, so active-but-old sessions are kept and the TTL is effectively "time since last activity," which is probably intended — but undocumented and surprising.

## Security Issues

- **Path traversal in uploads (filename + session_id):** see Bugs #1 and #2. This is the most serious finding. Untrusted `file.filename` and `session_id` reach filesystem paths without sanitization. Fix: store uploads under a server-generated name (`session_dir / "source" + safe_ext`); validate `session_id` with `uuid.UUID(session_id)` and reject anything else.
- **No file-content validation / MIME sniffing:** routing is purely by extension (`extractor.py:344`). A `.png` containing arbitrary bytes is sent to Claude; a `.dxf` is written to `/tmp` and parsed. No magic-byte check, no max image dimension. PDF rendering (`pypdfium2`, extractor.py:209) on attacker-controlled PDFs is an untrusted-parser surface.
- **CORS `allow_origins=["*"]` with all methods/headers** (main.py:36-40). Combined with no auth, any site can drive the API. Lock to known origins.
- **No authentication / rate limiting.** `/api/extract` triggers a paid Claude Vision call per request — unauthenticated and unthrottled, this is a direct cost-abuse / DoS vector against the `ANTHROPIC_API_KEY`.
- **`.env` contains a live-looking secret and is NOT git-ignored.** `.env` holds `ANTHROPIC_API_KEY` (108 chars, `sk-ant-` prefix — a real key shape, value not printed here). The project has **no `.gitignore`**, and the parent `/Users/maverick/Desktop/Claude dev/.gitignore` does not cover `.env` (`git check-ignore .env` → not ignored). The whole `cad-converter` dir is currently untracked, so the key isn't committed *yet*, but a `git add .` would commit it. **Action: add `.env` to a `.gitignore` immediately; rotate the key if it was ever shared.**
- **`/tmp` filename collision/predictability:** `extractor.py:352` writes to a predictable `/tmp/<filename>` path (also the traversal sink from #1). Use `tempfile.NamedTemporaryFile`.
- **Stack-trace leakage:** `main.py:89,113,136` interpolate raw exception text into HTTP responses (`f"Extraction failed: {exc}"`). Leaks internal details to clients. Log server-side, return generic messages.

## Production Readiness Gaps

- **No tests.** Zero unit/integration tests for extractor parsing, `apply_clarification_answers` dot-path logic, builder shape construction, or API contracts.
- **No CI** (no GitHub Actions / pipeline config).
- **No Docker / deployment manifest.** No Dockerfile, no render.yaml. CadQuery install is non-trivial (start.sh:11 documents the conda-on-macOS / pip-on-Linux split); without a pinned container, builds are environment-fragile — confirmed by the missing `OCP` module here.
- **Dependency mismatch:** `backend/requirements.txt` lists `numpy` and `aiofiles` (aiofiles is unused in the code) but does **not** list `cadquery` or `ezdxf`, even though both are required at runtime (`builder.py:48`, `extractor.py:223`). A `pip install -r requirements.txt` yields a non-functional app.
- **Deprecated FastAPI lifecycle:** `@app.on_event("startup"/"shutdown")` (main.py:45,53) is deprecated; migrate to the `lifespan` context manager.
- **No structured logging / observability:** logging is ad-hoc (`extractor.py:145`); no request IDs, no metrics, no health endpoint.
- **No input bounds on dimensions:** pydantic models (`models.py`) allow zero/negative/huge floats. A `height: 0` or negative box silently produces a degenerate/failed CadQuery build (caught only as a generic 500). Add `Field(gt=0)` constraints.
- **No concurrency/back-pressure control:** `ProcessPoolExecutor(max_workers=2)` (main.py:49); the 3rd+ concurrent build queues with no timeout — a hanging OCCT operation blocks a worker indefinitely. Add per-build timeouts.
- **Single-instance state:** sessions live on local disk; cannot run multiple replicas behind a load balancer.

## Feature Recommendations

- UUID validation helper + filename sanitization (closes the two path-traversal findings).
- API-key/session auth and per-IP rate limiting on `/api/extract`.
- Pydantic field constraints (`gt=0`) on all dimensions; return 422 with a clear message.
- Build timeout + cancellation; surface CadQuery geometry errors as actionable 4xx messages instead of opaque 500s.
- Pin a Docker image with CadQuery preinstalled (conda-forge base) and a `render.yaml`/compose file.
- Add `cadquery` and `ezdxf` to requirements; drop unused `aiofiles`.
- Test suite: golden-spec fixtures for extractor, dot-path mutation tests for `apply_clarification_answers`, and a build smoke test per base-shape.
- Frontend: fix feature/pattern reindexing; replace `innerHTML` string splicing with element-based row rebuilding; add client-side validation before build.
- Optional: glTF export for web preview (smaller than STL), multi-page PDF handling (currently only page 0, extractor.py:212), and confidence-driven UI cues.

## Cleanup Actions

- Create `cad-converter/.gitignore` with `.env`, `uploads/`, `__pycache__/`, `*.pyc`, `.DS_Store`. (Currently none exists.)
- Remove committed `.DS_Store` files (`cad-converter/.DS_Store` present).
- Remove unused `import math` in `models.py:3` (not referenced).
- Remove unused `aiofiles` from `requirements.txt`; add `cadquery`/`ezdxf`.
- Delete stale `backend/__pycache__/` (build artifacts).
- Migrate `on_event` → `lifespan`.
- Consolidate the inline `import shutil`/`import copy`/`import logging` to module top where reasonable (style).
