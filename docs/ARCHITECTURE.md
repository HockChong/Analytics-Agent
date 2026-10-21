# Architecture Plan — Gated Financial Document Pipeline

> Positioning: this is not a RAG demo. It is a pipeline for unstructured financial data
> with **data quality gates, permission metadata and an audit log**. It applies
> data-warehouse habits (layering like ODS/DWD/ADS, Write-Audit-Publish, incremental
> refresh, agreed metric definitions) to every stage of an LLM pipeline.
>
> Data sources: see §16. Incremental refresh: see §4.5.
>
> Design rule: **every layer ends in a deterministic code gate.** Prompts can only ask
> the model to behave. Gates enforce it. There is one LLM check (Gate 3b, answerability).
> It is fail-closed and never the last line: every answer it lets through still has to
> pass the code gates after it (G4, the period gate, numeric and support checks).
>
> What the gates prove, and what they do not: they prove an answer is grounded in the
> served evidence, cites rows that exist and that the user may see, and copies its numbers
> from rows the parser trusts. They do not prove the parser read the PDF correctly; that
> is measured (§16.9, row flags in §3.1), not gated.

---

## 1. System overview

```
data/raw/*.pdf
   │
   ▼
① PARSE      MinerU → middle.json → raw blocks (page, bbox, type, html/text)
   │         ── Gate 1: parse quality (row-level) → rejected rows + reject_reason
   ▼
② EVIDENCE   table rebuild + section inheritance → evidence table (staging), row flags
   │         ── Gate 2: schema + permission + reject ratio + orphan ratio → PUBLISH partition
   ▼
   SNAPSHOT  live partitions → indexes/<strategy>/<build_id>/ → swap CURRENT (§4.6)
   ▼
③ RETRIEVE   role mask on the snapshot → BM25 ∥ BGE-M3 dense (both ranked after the mask)
   │         → RRF(k=60) → top-K
   │         ── Gate 3a: dense top-1 cosine < threshold → refuse (no LLM call)
   │         ── Gate 3b: answerability check (LLM, fail-closed) → refuse
   ▼
④ GENERATE   constrained prompt → {answer, cited_ids}
   │         ── Gate 4: JSON schema, cited ⊆ retrieved, one marker per sentence → regenerate once, else refuse
   ▼
⑤ VERIFY     period gate (refuse) · numeric check · lexical support · uncited residue
   │         ── outcome: PASS | NEEDS_HUMAN_REVIEW | REFUSED
   ▼
AUDIT        audit_log.jsonl (source of truth; Langfuse cut from the plan, §8)
   ▼
UI           Streamlit: answer · clickable citations (page image + cell highlight) · audit panel
```

It runs as two separate processes:

| Process | When it runs | Layers | Output |
|---|---|---|---|
| **Offline build** (`make build`) | When a PDF, the parser or `permissions.yaml` changes | ①② + snapshot | `warehouse/` (Parquet) + `indexes/<strategy>/<build_id>/` + `CURRENT` |
| **Online query** (`app` / `eval`) | For each question | ③④⑤ + audit | Answer object + audit record. Reads **only** the CURRENT snapshot |

MinerU is heavy (GPU-friendly models, large image), so it runs **only** in the offline
build. The demo container gets the already parsed `middle.json` and the built indexes.
This keeps `docker compose up` fast and avoids a model download.

---

## 2. Repository layout

```
Analytics-Agent/
├── config/
│   ├── pipeline.yaml          # paths, parse tier, gate limits, index strategy; thresholds and models from D4
│   └── permissions.yaml       # section/content → permission_level rules, default, role map (simulated)
├── data/
│   ├── sources.yaml           # source_key → url, sha256, fiscal_year (committed)
│   ├── raw/                   # source PDFs, fetched by `make fetch` (gitignored)
│   └── parsed/<file_sha256>/<tier>/  # MinerU output cache: middle.json (gitignored)
├── warehouse/                 # "the data warehouse"
│   ├── staging/<run_id>/      # Write: newly parsed, not yet audited (kept only on audit failure)
│   ├── evidence/source_key=*/ # Publish: one partition per document (current)
│   ├── rejected/source_key=*/ # rejected rows with reject_reason, gate, run_id
│   ├── retired/<run_id>/      # partitions of REMOVED documents, moved aside, never deleted
│   ├── embed_cache.parquet    # (content_sha, embed_model) → vector (D4)
│   └── manifest.parquet       # one row per source_key per run (see §4.5)
├── indexes/<strategy>/        # serving snapshots (§4.6)
│   ├── <build_id>/            # evidence.parquet, meta.json; D4 adds bm25.pkl, vectors.npy
│   └── CURRENT                # the build_id queries read
├── src/finrag/
│   ├── schemas.py             # pydantic contracts (one place for every data shape)
│   ├── numbers.py             # number grammar: sign, scale, years/dates/notes (§7.2)
│   ├── fetch.py               # download sources.yaml entries, check pinned sha256
│   ├── parse/
│   │   ├── mineru_reader.py   # middle.json → RawBlock
│   │   ├── html_cells.py      # span.html → cell text (BeautifulSoup)
│   │   └── table_rebuild.py   # text-layer grid, header paths, html/text fallbacks (§16.8)
│   ├── evidence/
│   │   ├── build.py           # RawBlock → EvidenceRow (sections, permissions, ids, row flags)
│   │   ├── gates.py           # Gate 1 & 2 rules (pure functions, unit tested)
│   │   └── publish.py         # Write-Audit-Publish + incremental manifest + snapshot
│   ├── index/
│   │   └── snapshot.py        # build / publish / load the CURRENT snapshot, role mask
│   ├── retrieve/
│   │   ├── bm25.py  dense.py  fusion.py  gate.py
│   ├── generate/
│   │   ├── prompt.py          # evidence rendering with E1..En aliases
│   │   ├── llm.py             # provider-agnostic client, temperature 0, JSON mode
│   │   └── judge.py           # answerability check (Gate 3b, fail-closed)
│   ├── verify/
│   │   ├── citations.py       # Gate 4, sentence split, uncited residue
│   │   ├── period.py          # period gate (§7.3)
│   │   └── checks.py          # numeric check (uses finrag.numbers) + lexical support
│   ├── audit/
│   │   └── log.py             # append-only JSONL writer
│   └── pipeline.py            # answer(question, user_role) → AnswerRecord
├── eval/
│   ├── testset.jsonl          # hand-written questions + gold locations (§10.1)
│   ├── run_eval.py            # metrics table with intervals
│   ├── sweep_threshold.py     # replay of a 3a-off run (see §5.2 on the signal, §10.3 on the replay)
│   ├── tables_gold.json  score_tables.py  prefill_tables_gold.py  table_gold.py  # extractor benchmark (§16.9)
│   └── chunking_ablation.py   # deferred (§10.3)
├── app/streamlit_app.py
├── tests/                     # test_gates, test_numbers, test_table_rebuild, test_publish, test_score_tables
├── docker-compose.yml  Dockerfile  Makefile  README.md
```

---

## 3. Data contracts (`schemas.py`)

Every layer exchanges pydantic models, so a schema change shows up in one file and
breaks loudly, the same way a column contract works between warehouse layers.

### 3.1 `EvidenceRow` (the evidence table)

| Field | Type | Notes |
|---|---|---|
| `evidence_id` | str | `sha1(source_key, raw_ref, row_idx, content)[:12]`; `raw_ref` = `p{page}/b{block_position}`. See below. |
| `source_key` | str | Stable logical name of the document, from `data/sources.yaml` (e.g. `CICT_AR_FY2025`). Says *what* the document is |
| `file_sha256` | str | Which *version* of the file the row came from. Do not use it as the document key: a revised PDF would become a new document and the old rows would never be removed (§4.5) |
| `source_file` | str | |
| `content` | str | For table rows: `"caption \| row section \| label \| column path: value \| …"`, e.g. `"Net property income \| Group \| 2025 \| $’000: 1,189,749 \| …"` |
| `content_type` | enum | `paragraph`, `title`, `footnote`, `table_row`, `table_text` (a table kept as text) |
| `page_number` | int | PDF page index, 1-based (never the printed page, §16.6) |
| `bbox` | [x0,y0,x1,y1] | In PDF points, plus `page_width/page_height` so the UI can scale it |
| `table_id` | str \| null | Groups the rows of one table, so the UI can highlight the whole table |
| `table_status` | str \| null | `ok`, `partial`, `html_fallback`, `text_only`: how the table was rebuilt (§16.8) |
| `cells` | list \| null | For a `table_row`: `{column, value, bbox, scale}` per cell. `scale` is 1e3 for `'000`, 1e6 for `million`, null when no unit was printed |
| `row_flags` | list[enum] | Why the numeric check must not trust this row: `TABLE_NOT_OK`, `GENERIC_COLUMN`, `MULTI_VALUE`, `NON_NUMERIC`, `TEXT_MISMATCH` (text layer inside the cell bbox ≠ value). Empty = trusted |
| `section_title` | str | `"Chapter > heading"`: chapter from the PDF bookmarks, heading inherited in reading order (tables: the heading above and overlapping) |
| `permission_level` | enum | `public`, `internal`, `confidential` (from `permissions.yaml`, §4.4) |
| `numeric_values` | list[str] | For table rows the cell values; otherwise numbers found in `content`. Used in §7.2 |
| `parser` | str | `evidence-<hash of the parse and evidence code>` |
| `run_id` | str | The build run that produced the content. This is the data lineage; a RETAG does not change it |
| `tag_run_id` | str | The run that last set `permission_level` |
| `content_sha` | str | Key of the embedding cache |

**Why not `E_{page}_{row}_{col}`?** It collides across tables on the same page and across
documents. The hash does not collide, but it is only stable **within one parser version**:
`raw_ref` follows MinerU's block order and `content` includes the header path, so a parser
change produces new ids. That is why gold labels are locations, not ids (§10.1), and why
the embedding cache is keyed by `content_sha`, not `evidence_id`. In the prompt, use
**short per-query aliases `E1..En`** that map back to the hashes. Short aliases cost fewer
tokens and the model copies them more reliably. The validator checks the aliases against
the per-query map, which makes "cited ⊆ retrieved" an exact lookup.

### 3.2 `RejectedRow`
`gate` (`G1`/`G2`), `reject_reason` (enum), `detail`, `source_key`, `file_sha256`,
`page_number`, `block_type`, `bbox`, `content` (first 2,000 characters), `raw_refs` (paths
into middle.json), `run_id`.

### 3.3 `AnswerRecord` (one per query, the unit of audit)
```
query_id, ts, user_role, question, strategy, build_id, config_hash
retrieval: [{alias, evidence_id, permission_level, bm25_rank, dense_rank, rrf_score, dense_cos}]
gate3a: {signal: dense_top1_cos, value, threshold, passed}
gate3b: {answerable, missing, model, tokens, error}      # error → REFUSED at G3b (fail-closed)
generation: {raw_output, parsed, attempts, model, tokens_in, tokens_out, latency_ms}
gate4: {invalid_ids, unmarked_sentences, schema_ok, passed}
verify: {period: {question_periods, cited_periods, passed},
         numeric_checks[{alias, model_value, evidence_value, scale, kind: exact|scaled|derived|sign, ok}],
         support[{sentence, overlap, ok}], uncited_sentences[]}
final: {status: PASS|NEEDS_HUMAN_REVIEW|REFUSED, refusal_stage, review_reasons[], answer, citations[]}
cost_usd, latency_ms_total
```
`build_id` names the snapshot that served the answer (§4.6), so every citation can be
re-resolved later even after the document was re-parsed or retired.

---

## 4. Layers ① & ② — parse, rebuild, gate, publish

### 4.1 Parse (`mineru_reader.py`)
MinerU 4 writes `docvortex.middle` v2.0 (details in §16.7); any other schema is refused.
- Walk `pages[].blocks[]` and emit one `RawBlock(raw_ref, page_number, page size, type,
  bbox, text, html, styles)` per block. `raw_ref = p{pdf_page}/b{position}` points back
  into middle.json.
- Bboxes are normalised to 0–1 of the page; convert them to PDF points here, using the
  per-page size, so every later layer works in one coordinate system.
- **Text that sits only in html:** a block's text is its (possibly nested) `content`;
  for a table part whose content is html, the text comes from `html_cells`. Never assume
  plain `content` exists.
- Section titles come from the PDF bookmarks plus `doc_title` / `paragraph_title` blocks
  (§16.8), not from a heading-level stack: flash output has no reliable heading levels.

### 4.2 Table rebuild (`table_rebuild.py`)
The plan was an html grid from MinerU. As built, the grid comes from the PDF text layer,
with MinerU's bbox as a seed (algorithm and reasons in §16.8). Each table ends with a
status, stored on every row it produces (`table_status`):

| Status | Meaning | Evidence |
|---|---|---|
| `ok` | Every word in the region was placed in a column | `table_row`, one bbox per cell |
| `partial` | Some words could not be placed | `table_row`, flagged `TABLE_NOT_OK` |
| `html_fallback` | No column structure; MinerU's html key/value rows | `table_row` with column `value` and the table bbox, flagged |
| `text_only` | Neither | one `table_text` row |
| `failed` | No text at all | rejected, `TABLE_UNRECOVERABLE` |

Each `table_row` also gets `row_flags` (§3.1). A flag does not stop the row being
retrieved or cited. It stops the numeric check from treating the row as ground truth: an
answer whose numbers come from a flagged row is capped at `NEEDS_HUMAN_REVIEW` (§7.2). On
CICT FY2025, 546 of 1,063 table rows (51%) carry no flag.

### 4.3 Gate 1 — parse quality (row-level, `gates.py`)
| Rule | reject_reason |
|---|---|
| bbox missing, zero-area or outside the page | `BAD_BBOX` |
| `content.strip()` is empty, or has no letter or digit | `EMPTY_CONTENT` |
| More than 30% U+FFFD or control characters | `GARBLED_TEXT` |
| Table block with no text at all | `TABLE_UNRECOVERABLE` |
| Block typed `header`, `footer` or `page_number` by MinerU (§16.7) | `BOILERPLATE` |
| Text inside an `image` block, mostly numbers (chart labels, §16.6) | `CHART_LABEL` |
| Other text inside an `image` block | `IMAGE_TEXT` |
| Under 4 characters or 3 letters, mostly numbers: a table value outside any rebuilt table | `ORPHAN_NUMBER` |
| Under 4 characters or 3 letters, otherwise | `FRAGMENT` |

### 4.4 Gate 2 — contract, permission and coverage check before publish
- Pydantic validation of `EvidenceRow` → failure is `SCHEMA_VIOLATION`.
- **Permissions.** Each rule in `permissions.yaml` matches on `section` (the inherited
  `"Chapter > heading"`) or on `content`. Every matching rule votes and **the most
  restrictive level wins**. Content rules exist because headings are inherited and can
  be wrong: in the D3 build the related-party note on p161 sat under "27 Earnings Per Unit"
  and was tagged public, and a related-party paragraph on p149 sat under a public note
  heading. A row no rule matches gets `default`, which **must not be public**: the build
  refuses to start if it is (`check_permission_config`). With no default, the row is
  rejected as `NO_PERMISSION_TAG`. Public chapters are listed by name, so a new or renamed
  chapter is internal until someone decides otherwise.
- Duplicate `evidence_id` → `DUPLICATE_ID` (all copies rejected, never silently deduplicated).
- **Document-level audit** (per `source_key`, not per batch). A failing document is not
  published, its previous version stays live, and the other documents in the run still
  publish:
  - reconciliation: `raw = published + rejected + merged`, or the build raises;
  - quality reject ratio above `max_reject_ratio` (15%), counting only quality reasons;
  - **orphan value ratio** above `max_orphan_ratio` (35%): printed table values that
    ended up outside any rebuilt table, as a share of all table values. Orphans are
    dropped on purpose and do not count as quality rejects, so without this rule a
    document whose tables were never found would publish with a ratio near 3%. CICT
    FY2025 is at 21.9%.

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
| `parser_version` (hash of the parse/evidence code + extractor + MinerU version + tier) | `CHANGED` | Rebuild evidence from the cached `middle.json` for that tier; MinerU re-runs only for a new tier |
| `permissions_hash` (`permissions.yaml`) | `RETAG` | Re-tag rows in place in the partition (`tag_run_id` changes, `run_id` does not), new snapshot; **no re-embedding** |
| Nothing | `UNCHANGED` | None |
| In manifest, gone from `sources.yaml` | `REMOVED` | Move the partitions to `retired/<run_id>/`, new snapshot without them, manifest row `retired` |
| `embed_model` or chunking strategy | — | Not incremental: a new snapshot series (`indexes/<strategy>/`), full build |

**Flow (`make build`):**
```
1. diff     sources.yaml + data/raw  vs  manifest.parquet  → classify each source_key
            nothing to do → no snapshot is built (a second run does zero work)
2. parse    NEW/CHANGED only; skip MinerU if data/parsed/<file_sha256>/<tier>/middle.json exists
3. write    warehouse/staging/<run_id>/source_key=<k>/{evidence,rejected}.parquet
4. audit    per document: reconciliation + reject ratio + orphan ratio (§4.4)
            a failing document stays in staging and its previous version stays live;
            the other documents in the run still publish
5. publish  os.replace evidence.parquet, then rejected.parquet, into the partition;
            then append that document's manifest row (one row per document, not per run)
6. snapshot if any partition changed: build indexes/<strategy>/<run_id>/ from the live
            partitions, then swap CURRENT (§4.6). D4 adds to this step:
            dense: look up (content_sha, embed_model) in embed_cache; embed only misses
                   (a revised report that is 95% identical reuses 95% of its vectors)
            bm25:  full rebuild, because IDF depends on the whole corpus (seconds at this size)
```

**Build log** (one line per run, like a warehouse batch report):
```
run_id=20261021-044917  CHANGED=1
parsed=0  published_rows=3,971  rejected_rows=1,624  retagged_rows=0  snapshot=table_aware/20261021-044917
```

**Tests that prove it (`tests/test_publish.py`, on the real report in a temporary warehouse):**
- *Idempotency:* build twice; the second run parses, publishes and snapshots nothing, and
  rows keep their `run_id`.
- *Retag only:* change one permission rule; rows change level with no rebuild, `run_id`
  stays, `tag_run_id` changes, a new snapshot is served and the old one is unchanged.
- *Quarantine:* tamper with the PDF; the pinned hash fails, the live partition is unchanged.
- *Removal:* disable the source; the partition moves to `retired/` and the served snapshot
  has no rows from it.
- *Permissions:* in the first snapshot, the public role's rows contain no related-party text.

The row-count reconciliation (`raw_blocks == published + rejected + merged`) is the line
to show in an interview: no evidence disappears silently.

### 4.6 Serving snapshot and consistency (`index/snapshot.py`)
The warehouse partitions are the system of record. **Queries never read them.** They read
one snapshot: an immutable directory `indexes/<strategy>/<build_id>/` holding
`evidence.parquet` (every published row of every live document, with its permission
level), `meta.json` (build id, per-document `file_sha256` and `run_id`, row counts) and,
from D4, `bm25.pkl` and `vectors.npy`, all built from that same `evidence.parquet`.
Publishing is one `os.replace` of the `CURRENT` pointer.

This replaces the plan's mix of per-file partition swaps, a Qdrant alias swap, a BM25
rebuild in place and Qdrant payload updates in place for RETAG. Under that plan, a RETAG
from public to internal updated the partition and the Qdrant payloads while the app still
held the old BM25 in memory, so a public user's BM25 branch could return internal rows. A
REMOVED document could still be returned by an index that had not been rebuilt. Now:

- **One version per query.** The app loads a snapshot once and checks `CURRENT` between
  queries, never during one. Evidence, BM25 and vectors cannot come from different builds.
- **No retired rows.** REMOVED and RETAG build a new snapshot; until it is published,
  queries see the previous one in full. A row disappears from all three structures at once.
- **No vector database.** About 4,000 rows × 1,024 dims is 16 MB of float32; brute-force
  cosine on a masked matrix takes milliseconds. Dropping Qdrant removes the alias, the
  payload filter and the in-place update, and makes the permission mask the same numpy
  boolean array for both branches (§5.1).
- **Citations survive rebuilds.** `AnswerRecord.build_id` names the snapshot, which is
  never modified, so an old answer's citations can be re-resolved. Old snapshots can be
  pruned by age; the audit log records which ones were in use.
- **Page images.** The UI renders a citation from the PDF only if its sha256 equals the
  row's `file_sha256`; otherwise it shows the text without a highlight. This stops an old
  bbox being drawn on a revised PDF.
- **Partition files.** Evidence and rejected are two `os.replace` calls, so a crash
  between them leaves the pair out of step. That affects the audit trail only, not
  queries, and the next build reclassifies the document because its manifest row was
  not written.

---

## 5. Layer ③ — retrieval

### 5.1 Pipeline (`retrieve/`)
1. **Permission mask, before ranking.** `user_role` maps to allowed levels through
   `roles:` in `permissions.yaml`; `Snapshot.allowed(role)` returns one boolean row mask.
   BM25 scores and dense cosines are computed for masked rows only, and *then* each
   branch takes its top 50. Masking after the cut would leave a public user with fewer
   than 50 candidates and different populations in the two branches. Everything
   downstream (fusion, Gate 3a, the judge, the prompt, the audit record) only ever sees
   masked rows, so a refusal or a gate value cannot reveal that restricted rows exist.
   Test (D4): for role `public`, every id in BM25@50, dense@50, the fused top-K, the
   judge prompt and the generation prompt is `public`.
2. BM25 (`rank_bm25`) with a regex tokenizer that keeps numbers like `1,234.5` and tokens
   like `FY2024` whole. The corpus is English, so no CJK tokenizer. IDF is computed over
   the whole snapshot, not per role: the statistics leak nothing a user can read back.
3. Dense: `BAAI/bge-m3` dense vectors (normalised) in `vectors.npy` inside the snapshot.
   Brute-force cosine on the masked rows. Take the top 50.
4. RRF: `score(d) = Σ 1/(60 + rank_i(d))`. Keep the top-K (K=8) and both raw ranks for the audit.

### 5.2 Gate 3a — retrieval confidence (`gate.py`)
A raw RRF score is not on a 0–1 scale: with k=60 its maximum is 2/61 ≈ 0.033. The signal is

- `signal = cosine of the dense top-1 among the masked rows`, computed apart from fusion.
  ("Max cosine in the fused top-K" was undefined for rows that entered the fused list only
  through BM25.)

There is **no exact-match override.** The planned override passed any query whose year or
number token BM25 matched in the top 3. The FY2025 report mentions 2026 in many places
(debt maturities, lease expiries, outlook), so the override passed exactly the wrong-year
near misses that §10.1 is built to test.

**What 3a can and cannot do.** Cosine measures topicality. A near miss ("DPU in FY2026")
is on topic by construction and will score as high as an answerable question, so 3a only
catches off-topic questions. That is its job: it saves an LLM call. Near misses are
handled by 3b and, deterministically, by the period gate (§7.3). Plot the signal per
category (answerable, off-topic, near miss) before choosing the sweep range; BGE-M3
cosines often fall around 0.3–0.7. The signal name, value and threshold are written to
the audit record.

### 5.3 Gate 3b — answerability check (`judge.py`)
This runs only when 3a passes. One small LLM call returns
`{"answerable": bool, "missing": str}` given the question and the top-K (masked) evidence.
It is meant for questions that are on topic but not covered (for example, "the 2026
dividend policy" retrieves dividend paragraphs but none for 2026). It is separate from
generation, so the two refusal sources can be measured apart (§10.2: refusals by stage).

It is **an LLM check, not a code gate**, and it is treated that way:
- **Fail-closed.** A timeout, a parse failure or a non-boolean `answerable` gives
  `REFUSED` with stage `G3b_error`, logged in `gate3b.error`. Nothing gets past 3b
  because 3b broke.
- **Not trusted alone.** An LLM judge tends to say "answerable" too easily. A wrong-year
  question that 3b lets through is still refused by the period gate (§7.3).
- **Replayable.** Results are cached by `(question, evidence ids, model id)`, so an eval
  re-run gives the same 3b outcomes. Temperature 0 alone does not guarantee that across
  provider-side model updates.

---

## 6. Layer ④ — generation

- `prompt.py` renders each piece of evidence as
  `[E3] page=106 type=table_row section="Statement of Total Return" | Net property income | Group | 2025 | $'000: 1,189,749`.
  Curly quotes are normalised (`finrag.numbers.clean`) before rendering.
- The prompt is the constrained one from the brief: one sentence per claim, each ending
  with its marker; return `INSUFFICIENT_EVIDENCE` if the evidence is not enough; copy
  numbers as printed with the unit shown; strict JSON.
- `llm.py`: provider-agnostic (the model id lives in `pipeline.yaml`), temperature 0,
  JSON or structured-output mode, and a timeout.
- **Gate 4** (`citations.py`):
  - The JSON parses and matches `GenOutput{answer, cited_ids}`. There is no model
    `confidence` field: a self-reported score would not feed any gate.
  - Marker grammar: `[E3]`, `[E3, E5]` and `[E3][E5]` are all read as markers (regex
    `\[(E\d+(?:\s*,\s*E\d+)*)\]`). A loose `E3` without brackets is not a marker.
  - `set(cited_ids) ⊆ set(aliases)`, `set(markers) ⊆ set(cited_ids)`, and **every
    sentence (§7.1 splitter) that is not a connective carries at least one marker**.
  - On failure: **regenerate once**, adding the validator error to the prompt. If it
    fails again, refuse with `refusal_stage=G4`. Both attempts are logged.
- A test fixture (`tests/test_gate4.py`) feeds hand-made outputs with invented ids
  (`E99`), outputs where `cited_ids` and the inline markers disagree, `[E3, E5]` markers,
  and a two-sentence paragraph with one marker at the end. This is the D6 acceptance test.

## 7. Layer ⑤ — verification (code first)

Every check below reads only the cited rows from the serving snapshot. Each produces
either a refusal (§7.3) or a review reason. Review reasons are listed in
`final.review_reasons`, so the review rate can be broken down by cause (§10.2).

### 7.1 Sentence split and uncited residue
Split on `(?<=[.!?;])\s+(?=[A-Z(\["'])` and on newlines, never inside a number
(`1,189.7`) and never after an abbreviation (`Pte.`, `Ltd.`, `No.`, `S$`, `approx.`, `e.g.`,
`i.e.`, `vs.`). A plain `.` split would cut "S$1,189.7 million" and "Pte. Ltd." into
fragments with no marker. Because Gate 4 already requires a marker on every sentence, an
uncited sentence here is a residue (for example a connective the G4 rule allowed that
turns out to contain a digit). It goes to `uncited_sentences` and adds the review reason
`UNCITED`. The UI shows it in a separate box; it is **not** deleted.

### 7.2 Numeric check (`checks.py`, grammar in `finrag/numbers.py`)
Built and unit tested now (`tests/test_numbers.py`), because the evidence build already
uses it for cell scales.
- **What is a claim.** Numbers are extracted from each sentence after masking years
  (`2025`, `FY2025`), dates (`31 December 2025`) and note references (`Note 21`). For a
  table row, `numeric_values` holds only the cell values, so without the mask nearly every
  table answer ("In FY2025 …") would fail on the year.
- **Normalise.** Curly `’` → `'` (the report prints `$’000` in 465 of 1,063 table rows),
  full-width digits, thousands separators, `(1,234)` and `-1,234` → −1234, `%`.
- **Compare** each answer number with the numbers of the rows its sentence cites:

  | Result | Rule | Outcome |
  |---|---|---|
  | `exact` | Same value at the printed precision | ok |
  | `scaled` | Equal after applying the scale word in the answer and the cell `scale` (1,189,749 at `$'000` = S$1,189.7 million), to the precision the answer printed | ok. The two representations are equal; the model only chose a unit |
  | `sign` | Same magnitude, opposite sign ("decreased by 7,180" vs `(7,180)`) | review `SIGN` |
  | `derived` | Equals `a−b` or `(a−b)/b` of two cited values, within rounding | review `DERIVED` |
  | none | Anything else | review `NUMERIC_MISMATCH` |

  This relaxes the earlier "copy exactly" stance. A model that writes "S$1,189.7 million"
  from a `$'000` row is correct, and calling it a mismatch would make the review queue
  measure formatting, not risk. A rounded value ("S$1.19 billion") is still a mismatch:
  the answer printed more certainty than the row.
- **Flagged rows.** If a matched number comes from a row with `row_flags`, the outcome is
  at best review `UNTRUSTED_ROW`, whatever the comparison says. The parser may have put
  the right number under the wrong year.
- Every check is written to `numeric_checks`. Before D8, measure the review rate on 10
  hand-written answers; if one rule dominates, fix the rule, not the threshold.

### 7.3 Period gate (`period.py`) — refuses
The only deterministic defence against the wrong-year near miss. Take the periods in the
question (`FY2026`, `2026`, `1H 2025`, `as at 31 December 2024`). Each must appear in a
cited cell's column path or in a cited paragraph. If one does not, the answer is
`REFUSED` with stage `PERIOD`. Example: "What was DPU in FY2026?" passes 3a and maybe 3b;
the model cites the p5 row `2025: 11.58`; there is no 2026 column, so the answer is
refused. A question with no period passes this gate. The same rule applies to
segment names when the question names one from a fixed list (Retail, Office, Integrated
Development).

### 7.4 Lexical support (paragraph answers)
G4 proves a marker exists, not that the cited row supports the sentence. For each
sentence, at least 50% of its content words (stopwords removed, lower-cased, numbers and
proper nouns included) must appear in the rows it cites. Otherwise review
`WEAK_SUPPORT`. This is a cheap floor, not entailment: it catches a sentence about
occupancy that cites a row about leverage.

### 7.5 Final status
`REFUSED` (stage G3a / G3b / G3b_error / G4 / PERIOD / model INSUFFICIENT_EVIDENCE) ·
`NEEDS_HUMAN_REVIEW` (any review reason) · `PASS`.

---

## 8. Audit and observability

- **`audit/log.py`: an append-only `audit_log.jsonl` is the source of truth.** It has one
  `AnswerRecord` per query and is written even when Langfuse is down. The eval reads this
  same file, so the metrics come from the audit trail.
- **Langfuse is cut from the plan** (§13). If D12 has time, it can be added as a sink
  that reads the JSONL. If it is added, these rules apply:
  - Traces carry `evidence_id`, `content_sha` and `permission_level` for each row,
    **never row content or `raw_output`**. Langfuse Cloud is a third party, and anyone
    with project access could read confidential-labelled text in a trace whatever their
    role in the app. Content stays in the local JSONL.
  - Mask PII in question text (simple regexes for NRIC, email and phone).
- **What the audit log may hold.** The JSONL stores evidence content, because it is the
  local record of what the model saw. The UI's audit panel shows only the current query's
  record, which only ever contains rows the current role may see (§5.1).
- **The role is self-asserted.** The sidebar role selector simulates authentication; it
  is not authentication. The claim is that *given* a role, no restricted row reaches
  retrieval, the prompt, the gates or the audit panel. The README says so.
- Put an example of an intercepted query for each gate in the README.

---

## 9. UI (Streamlit)

- Sidebar: role selector (public / internal / confidential, simulated, §8), the serving
  `build_id`, and the thresholds, read-only so the user can see the active config.
- Main panel: the answer with `[E3]` shown as buttons. Clicking one renders the page with
  pypdfium2 at about 150 dpi and draws the bbox. For a `table_row` it also draws the
  `cells[].bbox` whose value matched. Scale the coordinates with
  `img_px = bbox_pt * dpi/72`. The page is rendered only if the PDF's sha256 equals the
  row's `file_sha256` (§4.6).
- Status badge: PASS (green), NEEDS_HUMAN_REVIEW (amber, lists each review reason and the
  items behind it), REFUSED (grey, shows the stage and the gate value, for example
  "retrieval confidence 0.31 < 0.45" or "question asks for 2026; cited rows cover 2021–2025").
- An expandable panel shows the raw `AnswerRecord` JSON.

---

## 10. Evaluation design

### 10.1 Test set (`eval/testset.jsonl`, written by hand)
```json
{"qid":"T07","category":"table","role":"public","split":"test",
 "question":"What was net property income in FY2025?",
 "gold":{"answerable":true,"value":"1,189.7","scale":1e6,
         "locations":[{"page":5,"point":[1080.2,301.5],"bbox":[1062,296,1098,307]},
                      {"page":106,"point":[452.0,233.1],"bbox":[430,228,474,238]}]}}
```
Store gold labels as **locations and values, not evidence_id.** Evidence ids change when
the parser or chunking strategy changes (§3.1). A gold item lists **every acceptable
location**: NPI is printed on p4, 5, 10, 25, 106 and 184, and citing p106 is not wrong.
For a table fact the location is the centre point of the printed value, plus its box.
(The coordinates above are illustrative.)

**Composition** (≈ 93 questions; unanswerable ones need no gold location, so they are cheap):

| Category | n | Notes |
|---|---|---|
| Paragraph | 25 | include synonym questions ("gearing" vs "aggregate leverage", §16.6) |
| Table | 20 | stratified by `table_status` of the gold row: 10 `ok`, 5 `partial`, 5 `html_fallback`, reported separately |
| Unanswerable, off-topic | 15 | tests 3a |
| Unanswerable, wrong period | 15 | "FY2026 DPU", "1H 2024 occupancy" (no half-year table): tests 3b and the period gate |
| Unanswerable, wrong segment or entity | 10 | a split the report does not give, another REIT's figure |
| Permission | 8 | (role, question) pairs: a fact only in a restricted section, asked as `public` (expect `REFUSED`) and as `confidential` (expect an answer). Check by grep that the fact appears nowhere public |

**Split.** Each question is assigned `dev` or `test` by a hash of its qid (about 40/60).
Thresholds and rules are chosen on `dev` only and reported on `test`.

### 10.2 Metric definitions
Every rate is reported with its count and a 95% Wilson interval (for example, 12/15 = 80%
has an interval of 55–93%), because the sets are small.

| Metric | How it is computed |
|---|---|
| Table citation correctness | Share of table questions where some cited row has a cell whose value matches the gold value (§7.2 rules) **and** whose bbox contains a gold location point on that page. Point-in-box, not IoU: a cell is about 30×8 pt, and a 3 pt offset in a hand-drawn box drops IoU below 0.5. A cell from the right row but the wrong year fails the value test |
| Paragraph citation correctness | Some cited paragraph's bbox contains a gold point, or overlaps the gold box by ≥ 50% of the smaller box |
| Citation precision | Cited rows that hit a gold location ÷ cited rows. Stops "cite all 8" from scoring well |
| Numeric consistency | Share of answered table questions where the answer value equals the gold value under §7.2 (`exact` or `scaled`) |
| Correct refusal rate | Share of unanswerable questions with `status=REFUSED`, **by category and by stage** (G3a, G3b, PERIOD, model) |
| Over-refusal rate | Share of answerable questions with `status=REFUSED`, by stage |
| Permission correctness | Share of permission pairs with the expected outcome. A restricted fact in a `public` answer is reported as a leak, separately |
| Review rate | Share with `NEEDS_HUMAN_REVIEW`, broken down by review reason (report it; do not hide it) |
| P95 latency, cost per query | From `AnswerRecord` |
| recall@5 | A gold location is inside any top-5 row, before generation |

### 10.3 Experiments
1. **Baseline run, with Gate 3a off** (threshold 0). Every question runs through 3b,
   generation and verify, so every stage's outcome is logged for every question.
2. **Threshold sweep.** Replay the baseline log on `dev` at each threshold: a question
   whose dense top-1 cosine is under the threshold becomes a 3a refusal; otherwise its
   logged downstream outcome stands. This needs no LLM calls, and over-refusal can be
   computed at any threshold (with 3a on in the baseline, questions it refused would have
   no downstream outcome to replay). Choose the knee, write down why, and report `test`
   at that threshold once.
3. **Chunking ablation (deferred; only if D10 is free).** Build a `fixed512` snapshot
   series beside `table_aware`. A fixed chunk spans several blocks and pages, so it has
   a **list** of block bboxes, and "covered" uses the same point-in-box rule. A single
   union bbox would favour large chunks under any containment test.

---

## 11. Configuration (`config/pipeline.yaml`)
Built (D1–D3):
```yaml
paths: {sources, raw_dir, parsed_dir, warehouse, indexes, permissions}
parse: {tier: flash, table_extractor: textlayer}
evidence: {max_reject_ratio: 0.15, max_orphan_ratio: 0.35}
index: {strategy: table_aware}
```
Added from D4:
```yaml
retrieval: {bm25_top: 50, dense_top: 50, rrf_k: 60, top_k: 8}
gate3a: {signal: dense_top1_cos, threshold: <chosen on dev, §10.3>}
gate3b: {enabled: true, model: <small-model-id>, timeout_s: 20}
generation: {model: <model-id>, temperature: 0, max_regenerations: 1}
verify: {support_min_overlap: 0.5}
audit: {jsonl_path: audit/audit_log.jsonl}
```
Permission rules, the default level and the role map live in `config/permissions.yaml`
(§4.4), whose hash drives RETAG. Each `AnswerRecord` stores `config_hash` and `build_id`,
so every logged answer can be reproduced.

---

## 12. Deployment

- `Dockerfile`: python:3.11-slim with the app dependencies and the BGE-M3 weights.
  Either bake the weights in or cache them in a mounted volume. **MinerU is not included.**
- `docker-compose.yml`: one `app` service (Streamlit). No vector database: the snapshot
  holds the vectors (§4.6). The container mounts `indexes/` and `data/raw/` read-only.
- `Makefile`: `make fetch` (download from `sources.yaml`, verify sha256), `make build`
  (MinerU flash, cached, + evidence + snapshot), `make test`, `make eval`, `make sweep`, `make app`.
- Do **not** commit the PDFs or the full `middle.json`. Annual reports are public but
  copyrighted, and `middle.json` contains most of the report's text. Commit
  `data/sources.yaml` instead, so a reviewer runs `make setup fetch build app`.
  For a no-GPU path, publish the parsed output as a GitHub release asset or keep it local,
  depending on how comfortable you are redistributing it.

---

## 13. 12-day plan (mapped to modules and acceptance checks)

| Day | Modules | Done when |
|---|---|---|
| D1 | MinerU on one SGX report; inspect `middle.json` | A table bbox can be drawn on the page image (a notebook is fine) |
| D2 | `mineru_reader`, `html_cells` | Prints "Net profit" with its page and bbox; html-only span count is logged |
| D3 | `table_rebuild`, `schemas`, `gates` (G1/G2), `publish` | `rejected.parquet` has rows with reasons; `raw == published + rejected` holds |
| D3+ | Design review fixes: permission rules, row flags, cell scale, `numbers`, coverage gate, serving snapshot | Done: `make test` passes 31 tests; no public row mentions related parties |
| D4 | Write the test set questions first (§10.1, no gold locations yet). Then `bm25`, `dense` (numpy), `fusion`, role mask | The all-ids-public test passes; 5 dev questions find the right row in the top 5 |
| D5 | `gate` (3a), `judge` (3b, fail-closed), `pipeline` skeleton, `audit/log` | 2 off-topic questions are refused at 3a, 1 near miss at 3b; 3 answerable ones pass |
| D6 | `prompt`, `llm`, `citations` (G4, marker grammar, one marker per sentence) + adversarial tests | `pytest tests/test_gate4.py` passes; `E99` and `[E3, E5]` are handled |
| D7 | `period` gate, `checks` (numeric + support), sentence splitter | The FY2026 DPU question is refused at PERIOD; review rate measured on 10 hand answers |
| D8–9 | Gold locations, `run_eval` with a 3a-off baseline | First metrics table, with intervals |
| D10 | `sweep_threshold` on dev; report test once | Threshold chosen and written down |
| D11 | Streamlit, Docker Compose | `docker compose up` shows the full loop, including a cell highlight, a PERIOD refusal and a permission refusal |
| D12 | README (the three required sections), recording | Ready to send |

**Cut:** Qdrant (numpy instead), Langfuse (JSONL only), the jieba tokenizer (the corpus
is English), the 3a exact-match override, the HKEX and FinanceBench extensions, and the
MinerU basic / Docling extractor benchmark. **Deferred:** the chunking ablation (D10 only
if free).

Line-count budget: `src/finrag` is already about 1,650 lines after D3 (the table rebuild
alone is about 460). Retrieval, generation, verification, audit and the pipeline add about
500–700, so the realistic total is about 2,200–2,400. Most of the remaining effort goes
into the test set, the calibration and the README.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Table rebuild gets the structure wrong | `table_status` and `row_flags` on every row; flagged rows cap answers at review (§7.2). Report results per table status instead of averaging them (§10.1) |
| Tables not found at all | Orphan value ratio gate (§4.4). Before writing gold locations, list the pages with the most `ORPHAN_NUMBER`s and check whether test questions land on them |
| Gate 3a separates answerable from unanswerable poorly with cosine alone | Expected for near misses (§5.2). 3b and the period gate handle them; the sweep reports each stage's share |
| Gate 3b lets near misses through | The period gate refuses them in code (§7.3); 3b failures are fail-closed |
| Review rate is dominated by formatting, not risk | §7.2 accepts scaled equality and masks years; review reasons are counted per cause, and 10 hand answers are measured before D8 |
| Permission labels inherited from the wrong heading | Content rules, most-restrictive-wins, non-public default (§4.4); permission questions in the eval (§10.1) |
| BGE-M3 is slow on CPU | Embed once at build time, cached by `content_sha`. A query embeds one string, about 100 ms |

---

## 15. Mapping to data-engineering experience

| Warehouse practice | Where it is in this pipeline |
|---|---|
| ODS / DWD / DWS / ADS layering | `data/parsed` (ODS) → `warehouse/evidence` (DWD) → `indexes/` (DWS) → `AnswerRecord` (ADS) |
| Write-Audit-Publish | `staging/` → Gate 2 + per-document reject and orphan ratios + row reconciliation → partition publish → snapshot pointer swap |
| Row-level quality rules | Gate 1 rules, with `rejected.parquet` holding the reasons |
| Incremental refresh | `manifest.parquet` fingerprint per `source_key` → rebuild only the changed partition, reuse cached embeddings |
| Data dictionary / agreed metric definitions | `schemas.py` contracts; `numbers.py` as the one number grammar; scale stored per cell |
| Lineage | `run_id` (content) and `tag_run_id` (permissions) on every row, `build_id` and `config_hash` on every answer |
| UAT | Hand-labelled test set with a dev/test split, metrics with intervals |
| Audit logging | `audit_log.jsonl` + refusal stage and review reasons on every answer |

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

### 16.2 Second document (D10+, only if time allows)
The candidate is the market review already listed in `data/sources.yaml`
(`CICT_IMR_2025`, disabled). It adds one `NEW` partition, leaves FY2025 alone, and
exercises the orphan value gate (§4.4) on a document of a different shape. The same
company's prior-year report would also enable cross-year questions. Cut from the plan
(§13): an HKEX bilingual report and a US 10-K.

### 16.3 External sanity check (cut)
A FinanceBench sample (Patronus AI; public 10-Ks with evidence pages) would show the
metrics are not tuned only to self-written questions, but it needs a second parsing
target and does not fit the 12 days. The README names it as the next step. FinQA and
TAT-QA start from pre-extracted text and tables, so they do not test the parsing layer.

### 16.4 Permission labels
All annual report content is public, so `permission_level` is simulated in
`config/permissions.yaml`, by section and by content (§4.4). Named chapters such as
financial highlights and the chair's statement are public; risk management and corporate
governance (including remuneration) are internal; related-party and interested-person
disclosures are confidential wherever they appear; anything else is internal by default.
The README states that the labels, and the role selector, are simulated.

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

**Decision (made in D2): cell bboxes come from the PDF text layer.** The options were
(1) the `basic` / `standard` tier, whose table model returns cell structure but needs
`huggingface.co` access and a model download, or (2) `flash` + the PDF text layer: inside
each MinerU table bbox, read word boxes with pdfplumber, cluster them into rows and
columns, and give each cell the union of its word boxes. Option 2 was built (§16.8). It
is deterministic, needs no GPU, and its coordinates come from the PDF itself. Comparing
against `basic` is cut from the plan (§13).

### 16.8 D2–D3 result: table rebuild, evidence table, gates, publish
Run with `make fetch && make build`. Code: `src/finrag/{parse,evidence}`; tests: `make test`
(31 tests after the design review; those that need the PDF skip until `make fetch`).

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

Coverage on CICT AR FY2025, first D3 run (123 tables after merging): 48 `ok`, 33
`partial` (some words not placed in a column), 27 `html_fallback`, 15 `text_only`, 0
`failed`, so 66% structured. After the §16.9 self-check fix ("Subtotal" no longer counts
as a label), the current build has **119 tables: 44 `ok`, 19 `partial`, 33
`html_fallback`, 23 `text_only`, 53% structured**. There was never an agreed target. The
`partial` and fallback tables are where the eval should be expected to show citation
misses, which is why the table questions are stratified by status (§10.1).

**Evidence build (current, 5,496 raw blocks):**

| Outcome | Blocks | Notes |
|---|---|---|
| published | 3,004 | → 3,971 evidence rows: 2,076 paragraph, 1,063 table_row (one bbox per cell), 740 title, 66 footnote, 26 table_text |
| merged | 868 | loose blocks consumed into a rebuilt table |
| rejected | 1,624 | ORPHAN_NUMBER 619 · BOILERPLATE 616 · EMPTY_CONTENT 167 · FRAGMENT 113 · IMAGE_TEXT 95 · CHART_LABEL 14 |

Reconciliation `raw = published + rejected + merged` holds, and the build raises an error
if any block has no outcome. The quality reject ratio is 3.0% (limit 15%); boilerplate,
chart labels and orphan numbers do not count towards it. **`ORPHAN_NUMBER` is the
parse-coverage figure to watch**: table values that ended up outside any rebuilt table,
mostly on pages where MinerU found no table at all. It now has its own gate: the orphan
value ratio is 21.9% (limit 35%, §4.4).

**Row flags** (§3.1) on the 1,063 table rows: 546 trusted, 517 `TABLE_NOT_OK`, 257
`GENERIC_COLUMN`, 203 `MULTI_VALUE` (for example "Asia Square Tower 2 | value: 2,245.0
2,252.0 …"), 157 `NON_NUMERIC` ("Property In | value: formation"), 2 `TEXT_MISMATCH`.
The re-read finds little for text-layer tables because their cells *are* text-layer
words; it exists for the Camelot grid, where cell boundaries come from elsewhere. 1,285
cells have scale `'000`, 109 `million`, and 1,179 have no printed unit (cents, %, sq ft,
prices).

**Section titles** come from the PDF's bookmarks (38 chapters) plus the latest heading:
`"Statement of Total Return"`, `"Property Details > Bugis Junction"`. A table takes the
heading that sits above it and overlaps it horizontally. The nearest heading in reading
order can be a neighbouring chart title.

Permission labels, first D3 run: 3,462 public, 511 internal, 57 confidential, with
`default: public` contradicting §4.4. The design review found 7 of 31 rows that mention
related parties or interested persons tagged public, because their inherited heading was
public (p149, p161). With the rules in §4.4 (non-public default, content rules,
most-restrictive-wins): **3,396 public, 506 internal, 69 confidential, and all 36
related-party/IPT rows are confidential.**

**Incremental refresh, as built (differences from §4.5):**
- `parser_version` is a hash of the parse and evidence code (plus `numbers.py`), the
  table extractor, the MinerU version and the parse tier, not a hand-maintained number.
  During D3 a logic change shipped while the constant stayed the same, and the run
  reported UNCHANGED. Hashing the code makes that impossible. The tier is in both the
  version and the cache path (`data/parsed/<sha>/<tier>/`), so switching flash → basic
  cannot silently reuse flash output.
- A pinned-hash mismatch (`HASH_MISMATCH`) or a missing file (`MISSING`) is reported as an
  error, and the live partition stays as it was.
- Publishing replaces each Parquet file in the partition with `os.replace`, and writes that
  document's manifest row straight after. Query consistency comes from the snapshot
  pointer (§4.6), not from the partition files.
- Checked by `tests/test_publish.py`: NEW → UNCHANGED (zero work) → RETAG (no rebuild,
  `run_id` kept, new snapshot) → HASH_MISMATCH (live data unchanged) → REMOVED (partition
  moved to `retired/`, no longer served). Adding the market review as a second document
  parsed and published only that document, and the annual report's rows kept their
  original `run_id`.

### 16.9 Table extractor benchmark (set up; waiting on the verified answer key)
Libraries tried on PDF pages 5 and 106 without a model download:
- **pdfplumber `extract_tables`:** ruled-line mode finds nothing (the tables use dotted
  lines). Text mode splits words across columns.
- **PyMuPDF `find_tables`:** finds the structure but **scrambles the numbers**
  ("1,305.1" → `13051 , .`). A table that looks right with wrong numbers: the reason
  numbers are checked in code (§7.2).
- **Camelot stream:** clean grids, but it needs to be told where the table is. On a whole
  page it mixes chart labels into the table.

As a result, `table_rebuild.py` has a switchable grid step (`parse.table_extractor:
textlayer | camelot`). Locating the table and the rules on top are shared. Camelot gets
the same region and defines rows and column boundaries; words and bboxes still come from
the text layer. The benchmark therefore measures one thing: who gets the grid right.

Two fixes came out of comparing them. The self-check counted "Subtotal" (the label for
rows that had none) as a real label, so the portfolio valuation table (p23), where every
property name had been lost, passed as structured. And Camelot's separate Note column is
now recognised by its last header word.

Whole report after the fixes: text layer 47 ok + 20 partial = 67 structured; Camelot
40 + 16 = 56. **They succeed on different tables** (Camelot gets p23, p25 and p154; the
text layer gets p27).

**Answer key:** `eval/tables_gold.json`, 10 tables and 216 cells, drafted by
`eval/prefill_tables_gold.py` and **to be verified by hand** (`eval/TABLES_GOLD.md`).
Scored on the unverified draft, each extractor gets close to 100% on what it found
itself, including Camelot's header "31" counted as a value. That shows why a key drafted
by the system is not evidence. `eval/score_tables.py` scores only verified tables and
counts wrong row / wrong column / wrong value separately from missing, because a number
attributed to the wrong year is worse than no number.

Next, after verification: score both extractors and report the result in the README.
"Run both, keep the one that passes the self-check" is worth trying only if it fits in
D10. MinerU basic and Docling are cut from the plan (§13).

### 16.10 Design review before D4: what changed
A design review before D4 checked whether each gate is really enforceable. It found
three problems: the permission default was public and leaked related-party text (§16.8),
nothing tied evidence, BM25 and the index to one version, and nothing deterministic
stopped a wrong-year answer. Built in response (D3+):

| Change | Where | Section |
|---|---|---|
| Non-public default, content rules, most-restrictive-wins, role map | `config/permissions.yaml`, `gates.resolve_permission` | §4.4 |
| `table_status`, `row_flags`, per-cell `scale` on every table row | `schemas.py`, `build.py` | §3.1, §4.2 |
| One number grammar (sign, scale, curly `’000`, years/dates/notes masked) | `numbers.py` | §7.2 |
| Orphan value ratio as a publish gate | `gates.orphan_value_ratio`, `publish.audit` | §4.4 |
| Serving snapshot behind a `CURRENT` pointer; Qdrant dropped | `index/snapshot.py` | §4.6 |
| RETAG keeps `run_id` and sets `tag_run_id`; REMOVED moves partitions to `retired/`; manifest row per document; tier in parser version and cache path | `publish.py` | §4.5 |

Specified for D4–D7 instead of the original plan:
- role mask applied before ranking in both branches (§5.1);
- 3a on dense top-1 cosine with no exact-match override (§5.2);
- 3b fail-closed (§5.3);
- G4 marker grammar and one marker per sentence (§6);
- the period gate (§7.3), scaled numeric equality (§7.2) and lexical support (§7.4);
- an eval with ≈93 questions, a dev/test split, intervals and point-in-box citation
  scoring, with the baseline run with 3a off (§10).
