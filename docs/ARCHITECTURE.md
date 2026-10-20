# Architecture Plan — Gated Financial Document Pipeline

> Positioning: this is not a RAG demo. It is a pipeline for unstructured financial data
> with **data quality gates, permission metadata and an audit log**. It applies
> data-warehouse habits (layering like ODS/DWD/ADS, Write-Audit-Publish, incremental
> refresh, agreed metric definitions) to every stage of an LLM pipeline.
>
> Data sources: see §16. Incremental refresh: see §4.5.
>
> Design rule: **every layer ends in a deterministic code gate.** Prompts can only ask
> the model to behave. Gates enforce it.

---

## 1. System overview

```
data/raw/*.pdf
   │
   ▼
① PARSE      MinerU → middle.json → raw blocks (page, bbox, type, html/text)
   │         ── Gate 1: parse quality (row-level) → rejected rows + reject_reason
   ▼
② EVIDENCE   table rebuild + section inheritance → evidence table (staging)
   │         ── Gate 2: schema + permission completeness → PUBLISH (atomic swap)
   ▼
③ RETRIEVE   permission pre-filter → BM25 ∥ BGE-M3 dense → RRF(k=60) → top-K
   │         ── Gate 3a: confidence threshold → refuse (no LLM call)
   │         ── Gate 3b: answerability judge (small LLM JSON call) → refuse
   ▼
④ GENERATE   constrained prompt → {answer, cited_ids, confidence}
   │         ── Gate 4: cited_ids ⊆ retrieved_ids, JSON schema → regenerate once, else refuse
   ▼
⑤ VERIFY     citation existence · numeric consistency · uncited-claim isolation
   │         ── outcome: PASS | NEEDS_HUMAN_REVIEW | REFUSED
   ▼
AUDIT        audit_log.jsonl (source of truth) + Langfuse trace (optional sink)
   ▼
UI           Streamlit: answer · clickable citations (page image + bbox highlight) · audit panel
```

It runs as two separate processes:

| Process | When it runs | Layers | Output |
|---|---|---|---|
| **Offline build** (`make build`) | When a PDF is added or changed | ①② + index build | `warehouse/` (Parquet/DuckDB) + `indexes/<index_version>/` |
| **Online query** (`app` / `eval`) | For each question | ③④⑤ + audit | Answer object + audit record |

MinerU is heavy (GPU-friendly models, large image), so it runs **only** in the offline
build. The demo container gets the already parsed `middle.json` and the built indexes.
This keeps `docker compose up` fast and avoids a model download.

---

## 2. Repository layout

```
Analytics-Agent/
├── config/
│   ├── pipeline.yaml          # thresholds, k, top_k, model ids, index_version
│   └── permissions.yaml       # section/page → permission_level rules (simulated)
├── data/
│   ├── sources.yaml           # source_key → url, sha256, fiscal_year (committed)
│   ├── raw/                   # source PDFs, fetched by `make fetch` (gitignored)
│   └── parsed/<file_sha256>/  # MinerU output cache: middle.json, page images (gitignored)
├── warehouse/                 # "the data warehouse"
│   ├── staging/<run_id>/      # Write: newly parsed, not yet audited
│   ├── evidence/source_key=*/ # Publish: one partition per document (current, served)
│   ├── rejected/source_key=*/ # rejected rows with reject_reason, gate, run_id
│   ├── embed_cache.parquet    # (content_sha, embed_model) → vector
│   └── manifest.parquet       # one row per source_key per run (see §4.5)
├── indexes/<index_version>/   # bm25.pkl, qdrant/ (or faiss), meta.json
├── src/finrag/
│   ├── schemas.py             # pydantic contracts (one place for every data shape)
│   ├── parse/
│   │   ├── mineru_reader.py   # middle.json → RawBlock
│   │   ├── html_cells.py      # span.html → cell text (BeautifulSoup)
│   │   ├── table_rebuild.py   # header detection, merged-cell inheritance, row joins
│   │   └── pdfplumber_fallback.py
│   ├── evidence/
│   │   ├── build.py           # RawBlock → EvidenceRow (sections, permissions, ids)
│   │   ├── gates.py           # Gate 1 & 2 rules (pure functions, unit tested)
│   │   └── publish.py         # Write-Audit-Publish + incremental manifest
│   ├── retrieve/
│   │   ├── bm25.py  dense.py  fusion.py  gate.py
│   ├── generate/
│   │   ├── prompt.py          # evidence rendering with E1..En aliases
│   │   ├── llm.py             # provider-agnostic client, temperature 0, JSON mode
│   │   └── judge.py           # answerability judge (Gate 3b)
│   ├── verify/
│   │   ├── citations.py       # Gate 4 + uncited-claim detection
│   │   └── numbers.py         # numeric normalisation & consistency
│   ├── audit/
│   │   ├── log.py             # append-only JSONL writer
│   │   └── langfuse_sink.py
│   └── pipeline.py            # answer(question, user_role) → AnswerRecord
├── eval/
│   ├── testset.jsonl          # 60 hand-written questions + gold page/bbox/value
│   ├── run_eval.py            # metrics table
│   ├── sweep_threshold.py     # 0.35→0.60 (see §5.3 on the signal scale)
│   └── chunking_ablation.py   # fixed-length vs table-aware recall@5
├── app/streamlit_app.py
├── tests/                     # gate unit tests + adversarial fixtures (fake ids etc.)
├── docker-compose.yml  Dockerfile  Makefile  README.md
```

---

## 3. Data contracts (`schemas.py`)

Every layer exchanges pydantic models, so a schema change shows up in one file and
breaks loudly, the same way a column contract works between warehouse layers.

### 3.1 `EvidenceRow` (the evidence table)

| Field | Type | Notes |
|---|---|---|
| `evidence_id` | str | **Stable hash**: `sha1(source_key, page, block_idx, row, col, content)[:12]`. See below. |
| `source_key` | str | Stable logical name of the document, from `data/sources.yaml` (e.g. `CICT_AR_FY2025`). Says *what* the document is |
| `file_sha256` | str | Which *version* of the file the row came from. Do not use it as the document key: a revised PDF would become a new document and the old rows would never be removed (§4.5) |
| `source_file` | str | |
| `content` | str | For table rows: `"FY2024 \| Net profit \| S$12.34m"` |
| `content_type` | enum | `paragraph`, `title`, `table_header`, `table_row`, `table_cell` |
| `page_number` | int | 1-based |
| `bbox` | [x0,y0,x1,y1] | In PDF points. Store `page_width/page_height` as well so the UI can scale it |
| `table_id` | str \| null | Groups the cells of one table, so the UI can highlight the whole table |
| `cell_bboxes` | list \| null | For a `table_row`: one bbox per cell, so the UI can highlight the exact value |
| `section_title` | str | The nearest heading above, inherited down the heading levels |
| `permission_level` | enum | `public`, `internal`, `confidential` (from `permissions.yaml`) |
| `numeric_values` | list[str] | Raw number strings taken from `content` at build time. Used in §6.2 |
| `parser` | enum | `mineru`, `pdfplumber_fallback` |
| `run_id` | str | The build run that produced the row. This is the data lineage |
| `content_sha` | str | Used for incremental refresh |

**Why not `E_{page}_{row}_{col}`?** It collides across tables on the same page and across
documents. It also changes when the parser changes. Use stable hashes in storage, and in
the prompt use **short per-query aliases `E1..En`** that map back to the hashes. Short
aliases cost fewer tokens and the model copies them more reliably. The validator checks
the aliases against the per-query map, which makes "cited ⊆ retrieved" an exact lookup.

### 3.2 `RejectedRow`
`EvidenceRow` fields (nullable) + `gate` (`G1`/`G2`), `reject_reason` (enum + detail),
`raw_ref` (path into middle.json), `run_id`.

### 3.3 `AnswerRecord` (one per query, the unit of audit)
```
query_id, ts, user_role, question, index_version, config_hash
retrieval: [{alias, evidence_id, bm25_rank, dense_rank, rrf_score, dense_cos}]
gate3a: {signal, value, threshold, passed}
gate3b: {answerable, reason, model, tokens}
generation: {raw_output, parsed, attempts, model, tokens_in, tokens_out, latency_ms}
gate4: {invalid_ids, schema_ok, passed}
verify: {uncited_sentences[], numeric_checks[{alias, model_value, evidence_value, ok}]}
final: {status: PASS|NEEDS_HUMAN_REVIEW|REFUSED, refusal_stage, answer, citations[]}
cost_usd, latency_ms_total
```

---

## 4. Layers ① & ② — parse, rebuild, gate, publish

### 4.1 Parse (`mineru_reader.py`)
- Walk `pdf_info[].para_blocks` / `preproc_blocks` and emit `RawBlock(page, bbox, type, lines/spans)`.
- **Spans that carry only html:** text = `span.get("content")` if present and non-empty,
  else `html_cells.extract(span["html"])`. Never assume `content` exists. Count each path
  in the run stats; the README can quote "N% of table spans were html-only".
- Read headings from `title` blocks to build a section stack (level → text).

### 4.2 Table rebuild (`table_rebuild.py`)
Rule-based and kept to about 100 lines. Coverage does not need to be perfect.
1. Parse the table html into a grid, expanding `rowspan`/`colspan` so each covered cell
   holds a copy of its parent's text. This handles merged-header inheritance.
2. Header rows: the first 1–2 rows in which most cells are non-numeric. `<th>` or bold
   text counts as a strong signal.
3. Row label: the first non-numeric column.
4. For each data cell, emit `"{column header path} | {row label} | {value}"`. Example:
   `"FY2024 | Net profit attributable | 12,340"`. The unit comes from the table caption
   or header (`S$'000`) and is added as a suffix.
5. Emit one `table_row` evidence unit per row, with `cell_bboxes` for each cell. This is
   the retrieval unit. `table_cell` units can be added later if recall@5 needs it.
6. Record `rebuild_status` = `ok | partial | failed`. If it failed, call
   `pdfplumber_fallback` on that page's bbox and mark `parser=pdfplumber_fallback`.
   Fallback rows go through the same gates.

### 4.3 Gate 1 — parse quality (row-level, `gates.py`)
| Rule | reject_reason |
|---|---|
| bbox missing, zero-area or outside the page | `BAD_BBOX` |
| `content.strip()` is empty, or only punctuation | `EMPTY_CONTENT` |
| More than 30% replacement or garbled characters | `GARBLED_TEXT` |
| Table rebuild failed and the fallback also failed | `TABLE_UNRECOVERABLE` |
| Page header, footer or page number (repeated on more than 50% of pages) | `BOILERPLATE` |

### 4.4 Gate 2 — contract and permission check before publish
- Pydantic validation of `EvidenceRow` → failure is `SCHEMA_VIOLATION`.
- `permission_level` is missing, or no rule in `permissions.yaml` matches → `NO_PERMISSION_TAG`.
  The default is **deny, never public**.
- Duplicate `evidence_id` → `DUPLICATE_ID`.
- Batch-level audit: if the rejection rate is above `max_reject_ratio` (for example 15%),
  **the publish is aborted**. The previous `evidence/` stays live. This is WAP at batch level.

### 4.5 Write-Audit-Publish + incremental refresh (`publish.py`)

**Document identity.** The manifest is keyed by `source_key` (what the document is), and
`file_sha256` is its version. Detecting a change means comparing the stored hash with the
current one for the same `source_key`.

**Fingerprint.** Each manifest row stores the inputs that decide how much work a document
needs. Different parts trigger different amounts of work:

| What changed | Classification | Work needed |
|---|---|---|
| Not in manifest | `NEW` | Parse, evidence, gates, embed |
| `file_sha256` | `CHANGED` | Parse, evidence, gates, embed only rows whose `content_sha` is not cached |
| `parser_version` (`table_rebuild.py` etc.) | `CHANGED` | Rebuild evidence from the cached `middle.json`; MinerU does not re-run |
| `permissions_hash` (`permissions.yaml`) | `RETAG` | Re-tag rows, update Qdrant payloads; **no re-embedding** |
| Nothing | `UNCHANGED` | None |
| In manifest, gone from `sources.yaml` | `REMOVED` | Drop partition, delete points, mark manifest row `retired` (not deleted) |
| `embed_model` or chunking strategy | — | Not incremental: new `index_version`, full build |

**Flow (`make build`):**
```
1. diff     sources.yaml + data/raw  vs  manifest.parquet  → classify each source_key
            nothing to do → exit early (a second run must do zero work)
2. parse    NEW/CHANGED only; skip MinerU if data/parsed/<file_sha256>/middle.json exists
3. write    warehouse/staging/<run_id>/source_key=<k>/{evidence,rejected}.parquet
4. audit    per document: Gate 2 + reject ratio + row reconciliation (raw = published + rejected)
            a failing document is skipped and its previous version stays live;
            the other documents in the run still publish
5. publish  replace only that document's partition in evidence/ (atomic directory rename)
6. index    dense:  look up (content_sha, embed_model) in embed_cache; embed only misses
                    (a revised report that is 95% identical reuses 95% of its vectors)
            qdrant: build the next collection, then switch the alias atomically
            bm25:   full rebuild, because IDF depends on the whole corpus (seconds at this size)
7. manifest append: source_key, file_sha256, parser_version, permissions_hash, status,
            run_id, rows published/rejected, embeddings computed/reused, published_at
```

**Why switch the index alias instead of editing in place.** Queries resolve evidence
through the index payload. If the evidence table and the index changed at different
moments, a query could cite a row that no longer exists. Switching the alias in one step
means every query sees one consistent version, and `AnswerRecord.index_version` records
which version served each answer.

**Build log** (one line per run, like a warehouse batch report):
```
run_id=20261020-1432  NEW=1 CHANGED=0 RETAG=0 UNCHANGED=1 REMOVED=0
published=1,842 rejected=97 (5.0%)  embeddings: computed=1,842 reused=0
```

**Tests that prove it (`tests/test_incremental.py`):**
- *Idempotency:* build twice; the second run reports `0 parsed, 0 embedded, 0 published`.
- *Isolation:* add the prior-year report; the FY2025 partition keeps its old `run_id`.
- *Retag only:* change one permission rule; the run reports `0 embedded, N payloads updated`.
- *Quarantine:* add a corrupted PDF; it fails audit, nothing else changes, and the reason is in `rejected/`.

The row-count reconciliation (`raw_blocks == published + rejected`) is the line to show
in an interview: no evidence disappears silently.

---

## 5. Layer ③ — retrieval

### 5.1 Pipeline (`retrieve/`)
1. **Permission pre-filter.** `user_role` maps to the allowed levels. Qdrant applies a
   payload filter, and BM25 applies a mask. Without this step `permission_level` is only
   a label. With it, the demo can show "an internal user sees an answer, a public user
   gets a refusal".
2. BM25 (`rank_bm25`) with a tokenizer that handles mixed Chinese and English (`jieba`
   for CJK, plus a regex that keeps numbers like `1,234.5` and tokens like `FY2024`
   whole). Take the top 50.
3. Dense: `BAAI/bge-m3` dense vectors (normalised), stored in Qdrant local mode. Take the top 50.
4. RRF: `score(d) = Σ 1/(60 + rank_i(d))`. Keep the top-K (K=8) and both raw ranks for the audit.

### 5.2 Gate 3a — retrieval confidence (`gate.py`)
**Signal scale (correction to the original plan):** a raw RRF score is not on a 0–1 scale.
With k=60 the maximum is 2/61 ≈ 0.033, so a threshold of 0.45 cannot be applied to it.
Recommended signal:

- `signal = max dense cosine among the fused top-K` (BGE-M3 cosine is on a 0–1-like scale), **and**
- an exact-match override: if the query contains a number, ticker or year token that
  BM25 matched exactly in the top-3, pass the gate.

The 0.35–0.60 sweep then makes sense, but the sweep is what fixes the range. BGE-M3
cosines often fall around 0.3–0.7. Plot the signal distribution for answerable and
unanswerable questions first, then choose the sweep range. The signal name, the value
and the threshold are all written to the audit record.

### 5.3 Gate 3b — answerability judge (`judge.py`)
This runs only when 3a passes. One small LLM call returns
`{"answerable": bool, "missing": str}` given the question and the top-K evidence.
It catches questions that are on topic but not covered (for example, "the 2026 dividend
policy" retrieves dividend paragraphs but none for 2026). It is cheap and separate from
generation, so the two refusal sources can be measured apart. Report each in the eval
(see §8: refusals by stage).

---

## 6. Layer ④ — generation

- `prompt.py` renders each piece of evidence as
  `[E3] page=12 type=table_row section="Financial Highlights" | FY2024 | Net profit | 12,340 (S$'000)`.
- The prompt is the constrained one from the brief: cite every claim; return
  `INSUFFICIENT_EVIDENCE` if the evidence is not enough; copy numbers exactly; strict JSON.
- `llm.py`: provider-agnostic (the model id lives in `pipeline.yaml`), temperature 0,
  JSON or structured-output mode, and a timeout.
- **Gate 4** (`citations.py`):
  - The JSON parses and matches `GenOutput{answer, cited_ids, confidence}`.
  - `set(cited_ids) ⊆ set(aliases)`, and `set(regex [E\d+] in answer) ⊆ set(cited_ids)`.
  - On failure: **regenerate once**, adding the validator error to the prompt. If it
    fails again, refuse with `refusal_stage=G4`. Both attempts are logged.
- A test fixture (`tests/test_gate4.py`) feeds hand-made outputs with invented ids
  (`E99`) and outputs where `cited_ids` and the inline markers disagree. This is the D6
  acceptance test.

## 7. Layer ⑤ — verification (code first)

### 7.1 Uncited-claim isolation
Split `answer` into sentences (`。！？.!?;` plus newlines). A sentence that has no `[E\d+]`
and is factual (it contains a digit or a named entity, or is longer than about 15
characters and is not a connective like "In summary") goes to `uncited_sentences`. The
UI shows these in a separate box labelled "No explicit evidence; needs human review".
They are **not** deleted. Any uncited sentence sets the status to `NEEDS_HUMAN_REVIEW`.

### 7.2 Numeric consistency (`numbers.py`)
- Extract numbers from each sentence and match them to the evidence that sentence cites.
- Normalise before comparing: thousands separators, full-width digits, `(1,234)` → −1234,
  `%`, currency symbols, and scale words such as `万`, `亿`, `million`, `bn`, `'000`,
  where the scale comes from the evidence unit suffix.
- A value matches if it equals the evidence value **after normalising the format only**.
  A unit conversion done by the model counts as a mismatch, because the prompt says copy
  exactly. This is the "agreed metric definition" stance: one representation, set at source.
- Each mismatch is written to `numeric_checks` as `{model_value, evidence_value, alias}`,
  and the status becomes `NEEDS_HUMAN_REVIEW`.

### 7.3 Final status
`REFUSED` (with stage G3a/G3b/G4/model INSUFFICIENT_EVIDENCE) ·
`NEEDS_HUMAN_REVIEW` (uncited or numeric mismatch) · `PASS`.

---

## 8. Audit and observability

- **`audit/log.py`: an append-only `audit_log.jsonl` is the source of truth.** It has one
  `AnswerRecord` per query and is written even when Langfuse is down. The eval reads this
  same file, so the metrics come from the audit trail.
- **Langfuse** is an optional sink. It holds one trace per query, with spans for
  `retrieve`, `gate3a`, `judge`, `generate`, `gate4` and `verify`. Each span's metadata
  holds that stage's part of the `AnswerRecord`, and generation spans carry token usage.
  - Note: self-hosting Langfuse v3 needs Postgres, ClickHouse, Redis and object storage.
    For this project use **Langfuse Cloud (free tier)**, or a separate
    `docker-compose.langfuse.yml` profile, so the main `docker compose up` stays light.
  - Mask PII in question text with Langfuse's `mask` callback. Use simple regexes for
    NRIC, email and phone. It costs little and shows compliance awareness.
- Put a screenshot of one trace in the README, plus an example of an intercepted query
  for each gate.

---

## 9. UI (Streamlit)

- Sidebar: role selector (public / internal / confidential), index version, and the
  threshold, read-only so the user can see the active config.
- Main panel: the answer with `[E3]` shown as buttons. Clicking one renders the page with
  PyMuPDF at about 150 dpi and draws the bbox. For `table_row` it also draws the specific
  `cell_bbox` whose value matched. Scale the coordinates with
  `img_px = bbox_pt * dpi/72` (or with `page_size` from middle.json).
- Status badge: PASS (green), NEEDS_HUMAN_REVIEW (amber, lists the uncited or mismatched
  items), REFUSED (grey, shows the stage and the gate value, for example
  "retrieval confidence 0.31 < 0.45").
- An expandable panel shows the raw `AnswerRecord` JSON and the Langfuse trace link.

---

## 10. Evaluation design

### 10.1 Test set (`eval/testset.jsonl`, written by hand)
```json
{"qid":"T07","category":"table","question":"What was net profit attributable to shareholders in FY2024?",
 "gold":{"answerable":true,"page":12,"bbox":[..],"value":"12,340","unit":"S$'000"}}
```
Store gold labels as **page + bbox + value, not evidence_id.** Evidence ids change when
the parser or chunking strategy changes. Bbox gold labels stay valid across both chunking
strategies, which the ablation needs.

Split: 25 paragraph · 20 table · 15 unanswerable. Among the unanswerable ones, include
**at least 5 near misses**: on topic, but the specific fact is absent (wrong year, a
segment the report does not split out). Near misses are the questions that test Gate 3b.

### 10.2 Metric definitions
| Metric | How it is computed |
|---|---|
| Table citation correctness | Share of the 20 table questions where a cited unit's `cell_bboxes` (or bbox) overlaps the gold bbox on the gold page with IoU ≥ 0.5 or containment |
| Paragraph citation correctness | Same, using paragraph bboxes |
| Numeric consistency | Share of answered table questions where the normalised answer value equals the gold value |
| Correct refusal rate | Share of the 15 unanswerable questions with `status=REFUSED`, **broken down by stage** |
| Over-refusal rate | Share of the 45 answerable questions with `status=REFUSED` |
| Review rate | Share with `NEEDS_HUMAN_REVIEW` (report it; do not hide it) |
| P95 latency, cost per query | From `AnswerRecord` |
| recall@5 (ablation) | Gold bbox covered by any top-5 unit, before generation |

### 10.3 Experiments
1. **Baseline run.** Fill in the metrics table.
2. **Threshold sweep.** Replay the retrieval logs at each threshold; this needs no LLM
   call for Gate 3a. For each threshold, record the correct-refusal rate, the
   over-refusal rate, and the share of refusals made by 3a and by 3b. Choose the knee and
   write down why.
3. **Chunking ablation.** Build `index_version=fixed512` (512 tokens with overlap,
   ignoring table structure) and `index_version=table_aware`. Run the same 20 table
   questions and compare recall@5 and citation correctness. `index_version` exists so
   that this is one config switch.

---

## 11. Configuration (`config/pipeline.yaml`)
```yaml
index_version: table_aware
retrieval: {bm25_top: 50, dense_top: 50, rrf_k: 60, top_k: 8}
gate3a: {signal: max_dense_cos, threshold: 0.45, exact_match_override: true}
gate3b: {enabled: true, model: <small-model-id>}
generation: {model: <model-id>, temperature: 0, max_regenerations: 1}
evidence: {max_reject_ratio: 0.15, default_permission: deny}
audit: {jsonl_path: audit/audit_log.jsonl, langfuse: false}
```
Each `AnswerRecord` stores `config_hash`, so every logged answer can be reproduced.

---

## 12. Deployment

- `Dockerfile`: python:3.11-slim with the app dependencies and the BGE-M3 weights.
  Either bake the weights in or cache them in a mounted volume. **MinerU is not included.**
- `docker-compose.yml`: `app` (Streamlit) plus optional `qdrant`. Local-mode Qdrant
  inside the app also works and is simpler. A `langfuse` profile is separate.
- `Makefile`: `make fetch` (download from `sources.yaml`, verify sha256), `make parse`
  (MinerU, on the host), `make build`, `make eval`, `make sweep`, `make ablation`, `make app`.
- Do **not** commit the PDFs or the full `middle.json`. Annual reports are public but
  copyrighted, and `middle.json` contains most of the report's text. Commit
  `data/sources.yaml` instead, so a reviewer runs `make fetch && make parse && make build && make app`.
  For a no-GPU path, publish the parsed output as a GitHub release asset or keep it local,
  depending on how comfortable you are redistributing it.

---

## 13. 12-day plan (mapped to modules and acceptance checks)

| Day | Modules | Done when |
|---|---|---|
| D1 | MinerU on one SGX report; inspect `middle.json` | A table bbox can be drawn on the page image (a notebook is fine) |
| D2 | `mineru_reader`, `html_cells` | Prints "Net profit" with its page and bbox; html-only span count is logged |
| D3 | `table_rebuild`, `schemas`, `gates` (G1/G2), `publish` | `rejected.parquet` has rows with reasons; `raw == published + rejected` holds |
| D4 | `bm25`, `dense`, `fusion`, permission filter | 5 manual questions: the right table row appears in the top 5 |
| D5 | `gate` (3a), `judge` (3b), `pipeline` skeleton, `audit/log` | 2 unanswerable questions are refused at 3a or 3b; 3 answerable ones pass |
| D6 | `prompt`, `llm`, `citations` (G4) + adversarial tests | `pytest tests/test_gate4.py` passes; the fake `E99` is blocked |
| D7 | `numbers`, uncited isolation, `langfuse_sink` | A trace shows the evidence list, gate values and verify output |
| D8–9 | `testset.jsonl`, `run_eval` | First version of the metrics table |
| D10 | `sweep_threshold`, `chunking_ablation` | Both comparison tables are filled in |
| D11 | Streamlit, Docker Compose | `docker compose up` shows the full loop, including bbox highlight and a refusal |
| D12 | README (the three required sections), recording | Ready to send |

Line-count budget: about 700–900 lines of Python across `src/`. Most of the effort goes
into the test set, the calibration and the README.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MinerU table recognition fails on complex statements | Mark `rebuild_status`, use the pdfplumber fallback, and keep the failures in `rejected/`. Report the coverage percentage instead of hiding it |
| Gate 3a separates answerable from unanswerable poorly with cosine alone | That is why Gate 3b exists. The sweep shows the share each gate handles; report it honestly |
| Model converts units ("12.34 million") despite the prompt | The numeric check flags it as a mismatch. Count how often this happens and use it as an interview example |
| BGE-M3 is slow on CPU | Embed once at build time. A query embeds one string, about 100 ms |
| Mixed Chinese and English tokenisation hurts BM25 | Use jieba plus a regex that protects numbers. Check on the 5 manual questions on D4 |
| Langfuse setup takes over a day | The JSONL audit log is the source of truth, so Langfuse is optional and can be added after D7 |

---

## 15. Mapping to data-engineering experience

| Warehouse practice | Where it is in this pipeline |
|---|---|
| ODS / DWD / DWS / ADS layering | `data/parsed` (ODS) → `warehouse/evidence` (DWD) → `indexes/` (DWS) → `AnswerRecord` (ADS) |
| Write-Audit-Publish | `staging/` → Gate 2 + batch reject ratio + row reconciliation → atomic publish |
| Row-level quality rules | Gate 1 rules, with `rejected.parquet` holding the reasons |
| Incremental refresh | `manifest.parquet` fingerprint per `source_key` → rebuild only the changed partition, reuse cached embeddings |
| Data dictionary / agreed metric definitions | `schemas.py` contracts; numeric check uses format-only normalisation |
| Lineage | `run_id` on every row, plus `index_version` and `config_hash` on every answer |
| UAT | 60-question hand-labelled set, metric acceptance thresholds |
| Audit logging | `audit_log.jsonl` + Langfuse traces + refusal stage on every refusal |

---

## 16. Data sources

### 16.1 Main corpus: one born-digital SGX annual report
- **Where:** SGXNet company announcements (sgx.com → Securities → Company Announcements,
  category "Annual Report") or the company's investor relations page.
- **Which:** start with an **S-REIT** (e.g. CapitaLand Integrated Commercial Trust,
  Mapletree Pan Asia Commercial Trust, Frasers Centrepoint Trust). It is about 150–250 pages,
  has many regular tables (DPU, net property income, gearing, valuation, lease expiry,
  top tenants) and defines its metrics explicitly. Leave banks (DBS, OCBC, UOB) for
  later: multi-level merged headers and tables that run across pages.
- **Check before choosing:**
  1. Born-digital, not scanned (text is selectable), so MinerU does not need OCR.
  2. Key tables are text, not images. Try selecting a number on the financial highlights page.
  3. Use the latest full year (FY2025), so questions about future years are unanswerable.

### 16.2 Second document (D10+, pick one)
| Goal | Source | Why |
|---|---|---|
| Incremental refresh demo (§4.5) | Same company's prior-year report | Adds one `NEW` partition and leaves FY2025 alone; enables cross-year and "wrong year" near-miss questions |
| Chinese + English (justifies BGE-M3) | HKEX bilingual annual report (hkexnews.hk) | SGX reports are almost all English |
| Check the method on US filings | 10-K from SEC EDGAR | Clean Item 1A / Item 7 structure |

### 16.3 External sanity check (optional)
Run 20–30 questions from the open **FinanceBench** sample (Patronus AI; public 10-Ks with
evidence pages) alongside the hand-built 60, to show the metrics are not only tuned to
self-written questions. FinQA and TAT-QA start from pre-extracted text and tables, so
they do not test the parsing layer.

### 16.4 Permission labels
All annual report content is public, so `permission_level` is simulated by section in
`config/permissions.yaml`. For example, financial highlights and the chair's statement
are public, risk management and remuneration are internal, and related-party transactions
are confidential. The README states that the labels are simulated.

### 16.5 `data/sources.yaml`
```yaml
- source_key: CICT_AR_FY2025
  issuer: CapitaLand Integrated Commercial Trust
  fiscal_year: 2025
  url: <link from SGXNet>
  sha256: <filled in by `make fetch` on first download, then pinned>
```
`make fetch` downloads each file and checks its hash against the pinned value. A mismatch
stops the build: the issuer re-published the file, and the change should go through the
`CHANGED` path in §4.5 on purpose, not by accident.

### 16.6 Findings from the first look at CICT AR FY2025
Checked with `pdftotext`/`pdfinfo` before parsing. These affect design decisions:
- **199 PDF pages, born-digital, text is selectable throughout.** Only pages 1, 196 and
  197 have almost no text (cover and back matter).
- **17 pages are two-page spreads** (1190.55 × 841.89 pt; PDF pages 4–7, 14–17, 22, 109,
  176–182). PDF page 5, for example, holds printed pages 4 and 5. Therefore:
  `page_number` is always the **PDF page index** (that is what bboxes, page rendering and
  gold labels use). Add `printed_page` only as a display field, and read the page size per
  page, never assume A4.
- **Charts carry their numbers as text.** The Financial Highlights spread (PDF page 5)
  has bar-chart labels (`1,619.2`, `1,586.3`, …) next to the five-year table with the same
  values. MinerU may emit these labels as loose text blocks. Expect duplicates that compete
  with the table rows in retrieval. Handle them with a Gate 1 rule (`CHART_LABEL`: a
  block made almost only of numbers inside an image or chart region), and test it on this page.
- **Good first table for D1–D2:** the five-year Key Financial Indicators table on PDF
  page 5 (DPU 11.58¢, aggregate leverage 38.6%, NPI S$1,189.7m, distributable income
  S$860.9m). The same figures appear again on PDF pages 4, 10 and 25/26 and in the financial
  statements (pages 106–107), which makes it a ready-made test of citing the right page
  when several pages agree.
- The terms differ from generic finance terms: the report says "aggregate leverage", not
  "gearing" (the word "gearing" appears once, on page 86), and "Statement of Total Return"
  rather than "income statement". Include synonym questions in the test set (§10.1). They
  test where BM25 alone fails and dense retrieval should help.

### 16.7 D1 result: MinerU 4.0.12 (`flash` tier) on PDF pages 4–10
MinerU is now version 4, which differs from the version 2 layout this plan first assumed:
- **CLI:** `mineru-kit parse <pdf> -o <dir> -f middle_json --tier <flash|basic|standard|advanced> -p 4-10`.
  `flash` reads the PDF text layer and needs **no model download**. On CPU it ran at
  about 2.5 pages/s, so the full report takes roughly 80 s. `basic` needs the ONNX models;
  `standard` also needs a 1.2B VLM (GGUF, via llama.cpp). Both are downloaded from Hugging Face.
- **Schema `docvortex.middle` v2.0:** `pages[].{page_idx, blocks[]}`, with `page_idx` 0-based
  (PDF page = `page_idx + 1`). Block types seen: `text`, `paragraph_title`, `doc_title`,
  `table`, `image`, `footer`, `page_number`, `page_footnote`, `equation`. Page sizes are
  in `extensions.docvortex_layout.pages[].{width_pt, height_pt}`.
- **Bboxes are normalised to 0–1** of the page: `points = bbox × (width_pt, height_pt)`.
  Drawn on PDF page 5, the table bbox fits the table exactly
  (`[652.4, 219.7, 1147.7, 515.2]` pt). See `scripts/draw_blocks.py`.
- **`flash` tables are text, not structure.** `table_body.content` is whitespace-aligned
  text, with no html and no cell bboxes. The **column header row (2021…2025) falls outside
  the table bbox** and is emitted as separate bold `text` blocks just above it. The table
  rebuild (§4.2) must therefore (a) attach header blocks that sit directly above the table
  and line up with its columns, and (b) get cell bboxes from somewhere else.
- **Chart labels are kept inside `image` blocks** (`image_body.content` = "1,619.2 …").
  This makes the `CHART_LABEL` rule (§16.6) simple: do not create evidence from
  `image_body.content` by default, and record them as rejected with that reason.
- Running headers and footers already come typed (`footer`, `page_number`), so the
  `BOILERPLATE` rule can use the type rather than counting repeats across pages.

**Open decision: where cell bboxes come from.**
1. `basic` / `standard` tier: the table model returns cell structure. This needs
   `huggingface.co` allowed in the network settings, plus a model download (size unknown until fetched).
2. `flash` + PDF text layer: inside each MinerU table bbox, read word boxes with
   pdfplumber, cluster them into rows (y) and columns (x, using the header blocks as
   anchors), and give each cell the union of its word boxes. This is exact for a
   born-digital PDF, because the coordinates come from the PDF itself rather than a model.
Recommendation: build option 2 first (deterministic, no GPU), then run `basic` on the same
pages and compare. The comparison belongs in the README's parsing section.

### 16.8 D2–D3 result: table rebuild, evidence table, gates, publish
Run with `make fetch && make build`. Code: `src/finrag/{parse,evidence}`; tests: `make test`
(18 tests; those that need the PDF skip until `make fetch`).

**Table rebuild uses the PDF text layer, not MinerU's table content** (decision from §16.7).
On real pages MinerU's flash table output was unreliable in three ways. The header row sat
outside the bbox (page 5). The bbox covered only the lower half of the values, with the
labels and upper rows split into a `code` block and loose `text` blocks (page 106). And
cells were merged ("333,300227,200", page 45). `table_rebuild.py` therefore uses the MinerU
bbox only as a **seed**:
1. Cluster the numeric words inside the seed by right edge into columns (figures are right-aligned).
2. Grow the region line by line over the text layer. Keep lines whose value-area words are
   mostly values, row-label lines and (going up) header lines; stop at running text.
3. Map header lines onto the columns. A group header ("Group", "Trust") with fewer
   phrases than columns is inherited by each column under it: the merged-cell rule.
4. Classify each body line as a data row, a section header (bold or coloured, no values),
   a wrapped label, a parent of "– sub item" rows, or a subtotal (values, no label).
   Note numbers ("Gross revenue  21") become "(Note 21)"; footnote markers are dropped.
5. **Self-check:** if fewer than 60% of rows have a label and a named column, the
   structure is discarded and the table falls back to its html key/value rows or plain
   text. A wrong table is worse than text, because the numeric check would trust it.

Coverage on CICT AR FY2025 (123 tables after merging): 48 `ok`, 33 `partial` (some words
not placed in a column), 27 `html_fallback`, 15 `text_only`, 0 `failed`. That is **66%
fully structured**, against the 70% that §4.2 said would be enough. The `partial` and
fallback tables are where the eval should be expected to show citation misses.

**Evidence build (first run, 5,496 raw blocks):**

| Outcome | Blocks | Notes |
|---|---|---|
| published | 2,978 | → 4,030 evidence rows: 2,059 paragraph, 1,156 table_row (one bbox per cell), 731 title, 66 footnote, 18 table_text |
| merged | 954 | loose blocks consumed into a rebuilt table |
| rejected | 1,564 | BOILERPLATE 616 · ORPHAN_NUMBER 584 · EMPTY_CONTENT 152 · FRAGMENT 103 · IMAGE_TEXT 95 · CHART_LABEL 14 |

Reconciliation `raw = published + rejected + merged` holds, and the build raises an error
if any block has no outcome. The quality reject ratio is 2.8% (limit 15%); boilerplate,
chart labels and orphan numbers do not count towards it. **`ORPHAN_NUMBER` (584) is the
parse-coverage figure to watch**: table values that ended up outside any rebuilt table,
mostly on pages where MinerU found no table at all.

**Section titles** come from the PDF's bookmarks (38 chapters) plus the latest heading:
`"Statement of Total Return"`, `"Property Details > Bugis Junction"`. A table takes the
heading that sits above it and overlaps it horizontally. The nearest heading in reading
order can be a neighbouring chart title. Permission labels from `config/permissions.yaml`:
3,462 public, 511 internal, 57 confidential.

**Incremental refresh, as built (differences from §4.5):**
- `parser_version` is a hash of the parse and evidence code, not a hand-maintained
  number. During D3 a logic change shipped while the constant stayed the same, and the run
  reported UNCHANGED. Hashing the code makes that impossible.
- A pinned-hash mismatch (`HASH_MISMATCH`) or a missing file (`MISSING`) is reported as an
  error, and the live partition stays as it was.
- Publishing replaces one Parquet file per partition with `os.replace`, which is atomic.
- Checked by `tests/test_publish.py`: NEW → UNCHANGED (zero work) → RETAG (no rebuild) →
  HASH_MISMATCH (live data unchanged) → REMOVED (partition retired). Adding the market
  review as a second document parsed and published only that document, and the annual
  report's rows kept their original `run_id`.
