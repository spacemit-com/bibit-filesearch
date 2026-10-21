# bibit-filesearch

Local semantic file search service. It indexes a user's knowledge base (default
`~/Documents`) and answers natural-language queries over local documents — **nothing
leaves the machine**.

The service exposes an HTTP contract, so it can be deployed standalone or used as the
data layer of an upstream agent: a caller sends one sentence in, the service locates
local files and returns them. Every logged-in user gets an independent instance
(its own config, data, index and port), so multiple users on one machine do not
interfere.

```
Caller: "search my local docs for GTK"
  → POST /api/search   (port derived from uid, 18700 range by default)
  → 1. gtk_realize_map_visable.md (100%)
```

---

## Features

| Capability | Description |
|---|---|
| Hybrid retrieval | Semantic (embedding cosine, chunk level → file level via max pooling) + BM25 sparse recall + filename channel, fused by real proportions (semantic 0.538462 / filename 0.230769 / BM25 0.230769, sum = 1) |
| Natural-language queries | Deterministic rule-based extraction of time range / directory / file-type filters, no LLM call (millisecond range); `pdfs on my desktop last week` works as-is |
| Multi-format parsing | 20 types: text `txt md docx pdf`; code `py js java cpp c cs go rs ts sql sh json`; spreadsheets and slides `xlsx pptx`; images `png jpg` |
| Images and scans | PP-OCR (det / cls / rec models) with a built-in orientation classifier, so skewed and upside-down text is read directly; inference can be offloaded to the SpaceMIT NPU |
| Automatic indexing | inotify incremental watching plus full scans; scanning can be paused, resumed, and continues after a restart |
| Idempotency | `content_hash` deduplication, unchanged content is skipped; config changes trigger automatic reclamation or rebuild (unchecking a type reclaims, re-checking rebuilds) |
| per-user isolation | One set of config, data, index and port per logged-in user; no conflicts on a shared machine |
| Standalone deployment | deb package plus user systemd; vectorization goes through an external embed endpoint, so this service does not load models itself |

Other retrieval knobs: `search.candidate_pool` (retrieval depth, 1000 chunks by
default), `search.bm25_half_score` (mapping constant K from raw BM25 score to 0~1,
default 2.8), `search.min_similarity` (semantic threshold 0.35; it has little
discriminating power at the current fused score scale and is a debugging knob),
and `top_k` (number of results returned, default 5). `candidate_pool` and `top_k` are
unrelated: the former controls how deep the search looks, the latter how many rows
are shown.

---

## Architecture

```
src/bibit_filesearch/
├── main.py          # Entry point: create_app assembly, serve subcommand, uid-derived port,
│                    # signal handling and shutdown release
├── config.py        # Config loading and factory defaults (config.yaml ⊕ code defaults)
├── settings.py      # Runtime config layer (writes / hot reload / version migration /
│                    # parser and chunk re-queue)
├── db.py            # SQLite WAL storage layer (files / chunks / scan_state)
├── indexer.py       # Executor: file → parse → chunk → embed → store, plus file aggregation
├── pool.py          # Index pool plus resident consumer (single source of truth for state)
├── scanner.py       # Full scan: batch producer (two phases: walk / vectorize)
├── watcher.py       # inotify incremental watching (direct ctypes, zero dependencies)
│                    # plus directory backfill
├── retriever.py     # Retrieval: cosine + BM25 + filename channel, three-way fusion and paging
├── nl_parser.py     # Natural language → structured filters (deterministic rules)
├── parser.py        # Per-format text and author extraction; PARSER_REVISION marks
│                    # the parsing capability version
├── chunker.py       # Text / code chunking (overlapping windows; code uses a character
│                    # budget so it never exceeds the embed limit)
├── bm25.py          # Self-contained BM25 inverted index (jieba tokenization, no
│                    # third-party retrieval dependency)
├── excludes.py      # Single definition of exclusion rules (shared by scanner / watcher /
│                    # reclamation)
├── pathutil.py      # Two path representations (raw filesystem bytes / encodable API string)
├── metadata.py      # File metadata (size, timestamps, author, directory; creation time
│                    # has a fallback chain)
├── state.py         # Scan state persistence (scan_state, resumable after restart)
├── recovery.py      # embed recovery watchdog (re-runs failed files once embed is back)
├── model_client.py  # embed HTTP client (dimension probed from the service, timeout with
│                    # queue bisection retry)
├── ocr_engine.py    # OCR: PP-OCR (the only engine), device auto / npu / cpu
├── ocr_ep.py        # NPU-side client (resident C++ process ep_worker)
└── api/             # FastAPI routes: search / settings / scan / files / meta
```

### Write path (indexing)

```
scanner (walk, seconds)          watcher (inotify, near real time)
        └──── offer / cancel ────┘
                    ↓
     index pool = files table (one row per file, state recorded on the row)
                    ↓ take / commit
        pool.IndexWorker (the single global resident consumer)
                    ↓
   indexer: parser extracts text → chunker splits → embed endpoint computes
            vectors → db persists
```

### Read path (retrieval)

```
query ─┬─ nl_parser: rule-based extraction of time / directory / type filters (no LLM)
       ├─ embed endpoint: query vector → chunk-level cosine → file-level max pooling
       ├─ bm25: inverted-index recall → raw/(raw+K) mapped to 0~1
       └─ filename / path keyword channel
              ↓ three-way weighted fusion (sum = 1) → structured filtering →
                ordering → paging
```

### Key design decisions

- **Single source of truth for state**: index state lives only in the `files` table
  (pool rows), written by the single resident consumer (`pool.IndexWorker`). Producers
  only offer / cancel; the executor only computes and never writes state. The unique
  path constraint on `files` is the physical guarantee of "one row per file".
- **Two-phase scanning**: walking (`os.walk`, seconds) is separated from vectorization
  (serial consumption, slow). Pause / resume apply to the whole task, and state is
  persisted so a restart resumes where it left off.
- **hash idempotency**: identical `content_hash` skips recomputation; when the parsing
  capability changes (`PARSER_REVISION`) or the chunking rules change
  (`CHUNK_REVISION`), affected files are re-queued once under the new rules.
- **Fusion scoring**: the three weights sum to 1, so the numbers in the config really
  are the proportions. Raw BM25 scores are mapped with `raw/(raw+K)` rather than
  per-batch min-max (the latter turns "everything matched" into "one perfect score and
  one zero").
- **per-user**: the port is derived from the uid (uid < 1000 → 18700, otherwise
  18700 + uid - 1000), config and data live under the user's home, and the unit uses
  `ConditionUser=!@system` to exclude system accounts.

---

## Platform and prerequisites

Targeted at **Bianbu (SpaceMiT / riscv64) devices**. The following are "what it takes to
run", not optional optimizations:

1. **Vectorization is an external dependency.** The semantic path calls
   `POST http://127.0.0.1:18790/v1/embed/embeddings` (default
   `qwen3-embedding-0.6b-q4_0`, 1024 dimensions, `--pooling mean`, hence
   `query_instruction` is empty). Without that endpoint retrieval fails; it does not
   degrade to BM25-only. You can run any OpenAI-compatible embed service and point
   `models.embed.{url,path,model,dim}` at it.
2. **Python 3.14** (`requires-python >= 3.14`).
3. **The riscv64 wheels for the OCR dependencies exist only in the SpaceMiT private
   index.** `onnxruntime` and `opencv-python-headless` have no official linux_riscv64
   wheels on PyPI, so `debian/postinst` installs them from that index by default; the
   index can be overridden with the `PIP_INDEX` environment variable. External
   environments need to provide the wheels themselves or build from source.
4. **The NPU is an accelerator, not a dependency.** If
   `/opt/bibit-filesearch/ocr-ep/ep_worker` and `libspacemit_ep.so` are present it uses
   the SpaceMIT NPU, otherwise plain CPU, with identical recognized text.

---

## Building and installing

### Option 1: run from source

```bash
pip install -r requirements.txt

# Start (default port derived from uid; the embed endpoint must already listen on :18790)
PYTHONPATH=src python -m bibit_filesearch.main serve

# Explicit port / config file
PYTHONPATH=src python -m bibit_filesearch.main serve --port 18700 --config /tmp/dev.yaml

# Verify
curl -s http://127.0.0.1:18700/health
curl -s http://127.0.0.1:18700/health/deep        # db / embed / index counts
curl -s -X POST http://127.0.0.1:18700/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"GTK","page":1,"pageSize":5}'

# First indexing run: trigger a full scan and check progress
curl -s -X POST http://127.0.0.1:18700/api/scan -H 'Content-Type: application/json' -d '{}'
curl -s http://127.0.0.1:18700/api/scan/progress
```

The first start does not write a config file: all effective values come from code
defaults, and an empty `index_dirs` is resolved dynamically to the current user's
`~/Documents` (only if the directory exists).

### Option 2: build the deb package

```bash
sudo apt install debhelper g++ python3.14 spacemit-onnxruntime   # build dependencies
dpkg-buildpackage -b -uc -us
# Artifact: ../bibit-filesearch_<version>_riscv64.deb
```

The build does two non-default things:

1. **Compiles the NPU helper** `packaging/ocr-ep/ep_worker.cpp`:
   `g++ -O2 -std=c++17 -lonnxruntime -lspacemit_ep`. `-lspacemit_ep` must be linked
   explicitly (the symbol lives in `libspacemit_ep.so`, not in `libonnxruntime.so`).
   Creating an EP session through Python's `spacemit_ort` segfaults; the C API
   `OrtSessionOptionsSpaceMITEnvInit()` is the only working entry point.
2. **Installs a global enable link for the user-level systemd unit**
   (`default.target.wants/` symlink, shipped as a deb file rather than `postinst
   enable`), so every logged-in user gets the service automatically.

### Option 3: install the deb

```bash
sudo apt install ./bibit-filesearch_<version>_riscv64.deb
```

Install layout:

```
/opt/bibit-filesearch/src/                 source (PYTHONPATH points here)
/opt/bibit-filesearch/.venv/               created by postinst, holds requirements.txt
/opt/bibit-filesearch/config.yaml          factory template (human-readable reference,
                                           not read at runtime)
/opt/bibit-filesearch/requirements.txt
/opt/bibit-filesearch/ocr-ep/ep_worker     NPU inference helper process
/usr/lib/systemd/user/bibit-filesearch.service
/usr/lib/systemd/user/default.target.wants/bibit-filesearch.service -> ../…
/usr/share/doc/bibit-filesearch/           copyright and changelog
```

`postinst` does the following in order: create the venv and install
`requirements.txt` from `PIP_INDEX` (SpaceMiT private index by default) → OCR / NPU
self-check (it actually runs one inference; a missing `libspacemit_ep.so` only logs
that it falls back to CPU and does not block installation) → `daemon-reload` +
`restart` for each currently logged-in non-system user.

Service management:

```bash
systemctl --user status  bibit-filesearch
systemctl --user restart bibit-filesearch
journalctl --user -u bibit-filesearch -n 200 --no-pager
```

The unit is not a root service: it runs as the logged-in user, reads that user's
config, and indexes that user's directories. Ports are derived from the uid, so
multiple users on one machine do not conflict.

Runtime paths (per-user):

| Purpose | Path |
|---|---|
| User config (sparse, only overridden keys) | `~/.config/bibit-filesearch/config.yaml` |
| Index database (SQLite WAL) | `~/.local/share/bibit-filesearch/filesearch.db` |
| Scan error log | `~/.local/share/bibit-filesearch/scan_errors.log` |
| Data directory override | `BIBIT_FILESEARCH_DATA_ROOT` environment variable |

---

## Configuration

The authoritative defaults live in code (`config.DEFAULTS` plus the factory defaults in
`settings`). `~/.config/bibit-filesearch/config.yaml` stores only the keys that differ
from the factory defaults (sparse storage); it is not created on first start, only when
settings are saved from the UI. Settings saved through the UI take effect immediately via
`PUT /api/settings`; editing the file directly requires a service restart.

Top-level keys exposed in the UI:

```yaml
index_dirs: []          # knowledge base paths; empty → resolved to ~/Documents
auto_index: true        # watcher incremental monitoring
indexable_types: [txt, md, docx, pdf, py, js, java, cpp, c, cs, go, rs, ts,
                  sql, sh, json, xlsx, pptx, png, jpg]     # all 20 types
exclude_patterns: [.git, node_modules, __pycache__]
max_depth: 3            # 1-based; 0 = unlimited depth
top_k: 5                # results returned (display only, does not affect depth)
```

Debugging knobs available only by editing the config file:

```yaml
search:
  candidate_pool: 1000    # retrieval depth: how many top chunks take part in
                          # aggregation; 0 = no truncation
  embed_weight: 0.538462
  filename_weight: 0.230769
  bm25_weight: 0.230769   # 0 = disable the BM25 path
  bm25_half_score: 2.8    # mapping constant K from raw BM25 score to 0~1
  min_similarity: 0.35
ocr:
  device: auto            # auto / npu / cpu
  max_side_len: 1400      # main knob for peak memory
  idle_release_secs: 10   # idle time before releasing an OCR worker (0 = off)
  worker_max_runs: 0      # RUN count limit per worker (0 = off)
  worker_max_rss_mb: 0    # RSS limit per worker in MB (0 = off)
  pdf_max_ocr_pages: 20   # max pages OCR'd per PDF (0 = unlimited)
models:
  embed:
    url: http://127.0.0.1:18790
    path: /v1/embed/embeddings
    model: qwen3-embedding-0.6b-q4_0
    dim: 1024
```

---

## HTTP API

Every response is `{code, message, data}` with `code = 0` meaning success.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/health/deep` | Deep health (db / embed / index counts) |
| GET | `/api/meta` | Instance identity: uid / userName / homeDir / derived port / OCR state |
| POST | `/api/search` | Search (query + filters + paging) |
| GET | `/api/files/{fileId}` | File details (metadata plus chunk statistics) |
| POST | `/api/scan` | Trigger a full scan (idempotent) |
| GET | `/api/scan/progress` | Two-phase progress |
| POST | `/api/scan/pause` · `/api/scan/resume` | Pause / resume |
| GET | `/api/settings` | Read config |
| PUT | `/api/settings` | Update config (idempotent, `data.changed` says whether anything changed) |
| POST | `/api/settings/reset` | Restore defaults |

In `/api/search` responses, `score` (the fused score, which can exceed 1) and
`confidence` (a normalized display score) mean different things; do not mix them.

---

## Dependencies

Runtime (`requirements.txt`): `fastapi` / `uvicorn` / `PyYAML` / `jieba` / `numpy` /
`pymupdf` / `python-docx` / `python-pptx` / `openpyxl` / `rank-bm25` / `httpx` /
`rapidocr` / `onnxruntime` / `opencv-python-headless`.

- Vectorization backend: embed endpoint `:18790` → `POST /v1/embed/embeddings`
  (Qwen3-Embedding-0.6B Q4_0, 1024 dimensions, mean pooling).
- OCR: `rapidocr` ships the three PP-OCRv6 det / cls / rec onnx models (about 32MB,
  installed with the wheel, no network download) and runs them on `onnxruntime`. There
  is only one OCR engine: a failed recognition is recorded as `failed` with a reason
  instead of being silently retried on a different engine.
- Verifying locally: `PYTHONPATH=src python -m pytest tests/ -q` (requires `pytest` /
  `pytest-asyncio`; the OCR and embed tests use fake engines, so no real models or NPU
  are needed).

---

## License

Apache-2.0, see `LICENSE`.
