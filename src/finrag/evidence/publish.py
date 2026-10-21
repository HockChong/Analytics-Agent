"""Write-Audit-Publish with incremental refresh (docs/ARCHITECTURE.md §4.5).

    python -m finrag.evidence.publish [--config config/pipeline.yaml]

1. diff     sources.yaml + files on disk vs manifest → NEW | CHANGED | RETAG | UNCHANGED | REMOVED
2. parse    NEW/CHANGED only; MinerU output is cached by file hash and tier
3. write    warehouse/staging/<run_id>/source_key=<k>/{evidence,rejected}.parquet
4. audit    per document: reconciliation + quality reject ratio + orphan value ratio; a
            failing document is skipped and its previous version stays live
5. publish  os.replace each staged file into warehouse/{evidence,rejected}/source_key=<k>/,
            then append that document's manifest row
6. snapshot if anything changed: write indexes/<strategy>/<run_id>/ from the live partitions
            and swap the CURRENT pointer (finrag.index.snapshot). Queries read only that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd
import yaml

from finrag.evidence import gates
from finrag.evidence.build import PARSER_VERSION, BuildResult, Source, build_document
from finrag.index import snapshot
from finrag.parse.mineru_reader import load_middle
from finrag.schemas import ContentType, EvidenceRow, RejectedRow, RejectReason

MANIFEST_COLUMNS = [
    "run_id", "source_key", "status", "action", "file_sha256", "parser_version", "permissions_hash",
    "raw_blocks", "published_blocks", "rejected_blocks", "merged_blocks", "evidence_rows",
    "rejected_rows", "quality_reject_ratio", "orphan_value_ratio", "detail", "finished_at",
]


@dataclass
class Plan:
    source_key: str
    action: str  # NEW | CHANGED | RETAG | UNCHANGED | REMOVED
    source: dict | None
    file_sha256: str | None
    reason: str = ""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parser_version(table_extractor: str = "textlayer", tier: str = "flash") -> str:
    return f"{PARSER_VERSION}+{table_extractor}+mineru-{metadata.version('mineru')}-{tier}"


class Warehouse:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.parquet"

    def partition(self, kind: str, source_key: str) -> Path:
        return self.root / kind / f"source_key={source_key}" / f"{kind}.parquet"

    def manifest(self) -> pd.DataFrame:
        if self.manifest_path.exists():
            return pd.read_parquet(self.manifest_path)
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    def live_state(self) -> dict[str, dict]:
        """Latest published (or retired) manifest row per source_key."""
        m = self.manifest()
        m = m[m["status"].isin(["published", "retired"])]
        state = {}
        for _, row in m.iterrows():
            state[row["source_key"]] = row.to_dict()  # rows are appended in run order
        return {k: v for k, v in state.items() if v["status"] == "published"}

    def append_manifest(self, rows: list[dict]) -> None:
        if not rows:
            return
        m = pd.concat([self.manifest(), pd.DataFrame(rows, columns=MANIFEST_COLUMNS)], ignore_index=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        self.root.mkdir(parents=True, exist_ok=True)
        m.to_parquet(tmp, index=False)
        os.replace(tmp, self.manifest_path)


# ---------------------------------------------------------------- 1. diff
def plan_runs(sources: list[dict], live: dict[str, dict], perm_hash: str, pversion: str) -> list[Plan]:
    plans = []
    enabled = {s["source_key"]: s for s in sources if s.get("enabled")}
    for key, src in enabled.items():
        path = src["local_path"]
        if not Path(path).exists():
            plans.append(Plan(key, "MISSING", src, None, f"{path} not found; run make fetch"))
            continue
        sha = sha256_file(path)
        if src.get("sha256") and src["sha256"] != sha:
            plans.append(Plan(key, "HASH_MISMATCH", src, sha,
                              f"pinned {src['sha256'][:12]}, on disk {sha[:12]}: update sources.yaml on purpose"))
            continue
        prev = live.get(key)
        if prev is None:
            plans.append(Plan(key, "NEW", src, sha))
        elif prev["file_sha256"] != sha or prev["parser_version"] != pversion:
            why = "file changed" if prev["file_sha256"] != sha else "parser changed"
            plans.append(Plan(key, "CHANGED", src, sha, why))
        elif prev["permissions_hash"] != perm_hash:
            plans.append(Plan(key, "RETAG", src, sha))
        else:
            plans.append(Plan(key, "UNCHANGED", src, sha))
    for key in live:
        if key not in enabled:
            plans.append(Plan(key, "REMOVED", None, None))
    return plans


# ---------------------------------------------------------------- 2. parse
def ensure_parsed(pdf_path: str, sha: str, parsed_dir: Path, tier: str) -> tuple[Path, bool]:
    out_dir = parsed_dir / sha / tier  # a tier switch must not reuse another tier's output
    middle = out_dir / "middle.json"
    if middle.exists():
        return middle, False
    out_dir.mkdir(parents=True, exist_ok=True)
    mineru = Path(sys.executable).with_name("mineru-kit")
    subprocess.run([str(mineru), "parse", pdf_path, "-o", str(out_dir), "-f", "middle_json",
                    "--tier", tier, "-p", "all"], check=True, capture_output=True)
    produced = out_dir / f"{Path(pdf_path).stem}.json"
    os.replace(produced, middle)
    return middle, True


# ---------------------------------------------------------------- 3-5. write, audit, publish
def _rows_df(rows: list[EvidenceRow] | list[RejectedRow]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump(mode="json") for r in rows])


def audit(result: BuildResult, max_ratio: float, max_orphan_ratio: float) -> tuple[bool, float, float, str]:
    s = result.stats
    if s["raw_blocks"] != s["published_blocks"] + s["rejected_blocks"] + s["merged_blocks"]:
        return False, 0.0, 0.0, "reconciliation failed: raw != published + rejected + merged"
    # Count each rejected block once, by its first reason.
    first_reason: dict[str, RejectReason] = {}
    for r in result.rejected:
        first_reason.setdefault(r.raw_refs[0], r.reject_reason)
    rejected_refs = {ref for ref, o in result.outcomes.items() if o == "rejected"}
    ratio = gates.quality_reject_ratio([first_reason[r] for r in rejected_refs if r in first_reason], s["raw_blocks"])
    orphan = gates.orphan_value_ratio(
        [r.content for r in result.rejected if r.reject_reason == RejectReason.ORPHAN_NUMBER],
        sum(len(e.cells or []) for e in result.evidence if e.content_type == ContentType.table_row))
    if ratio > max_ratio:
        return False, ratio, orphan, f"quality reject ratio {ratio:.1%} > {max_ratio:.0%}"
    if orphan > max_orphan_ratio:
        return False, ratio, orphan, f"orphan value ratio {orphan:.1%} > {max_orphan_ratio:.0%}: tables were not found"
    return True, ratio, orphan, "ok"


def stage_and_publish(wh: Warehouse, run_id: str, key: str, evidence_df: pd.DataFrame,
                      rejected_df: pd.DataFrame, publish: bool) -> None:
    staging = wh.root / "staging" / run_id / f"source_key={key}"
    staging.mkdir(parents=True, exist_ok=True)
    staged = {"evidence": staging / "evidence.parquet", "rejected": staging / "rejected.parquet"}
    evidence_df.to_parquet(staged["evidence"], index=False)
    rejected_df.to_parquet(staged["rejected"], index=False)
    if not publish:
        return  # stays in staging for inspection; the live partition is untouched
    for kind, path in staged.items():
        target = wh.partition(kind, key)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)


def run(config_path: str = "config/pipeline.yaml", run_id: str | None = None) -> dict:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    paths = cfg["paths"]
    sources = yaml.safe_load(open(paths["sources"], encoding="utf-8"))
    perm_text = Path(paths["permissions"]).read_text(encoding="utf-8")
    permissions = yaml.safe_load(perm_text)
    gates.check_permission_config(permissions)
    perm_hash = hashlib.sha256(perm_text.encode()).hexdigest()[:16]
    extractor = cfg["parse"].get("table_extractor", "textlayer")
    pversion = parser_version(extractor, cfg["parse"]["tier"])
    wh = Warehouse(paths["warehouse"])
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    max_ratio = cfg["evidence"]["max_reject_ratio"]
    max_orphan = cfg["evidence"].get("max_orphan_ratio", 1.0)
    strategy = cfg.get("index", {}).get("strategy", "table_aware")

    plans = plan_runs(sources, wh.live_state(), perm_hash, pversion)
    log = {"run_id": run_id, "actions": {}, "parsed": 0, "published_rows": 0,
           "rejected_rows": 0, "retagged_rows": 0, "snapshot": None, "errors": []}
    live_changed = False

    def manifest_row(plan: Plan, status: str, stats: dict | None = None, ratio: float = 0.0,
                     orphan: float = 0.0, detail: str = ""):
        stats = stats or {}
        return {
            "run_id": run_id, "source_key": plan.source_key, "status": status, "action": plan.action,
            "file_sha256": plan.file_sha256, "parser_version": pversion, "permissions_hash": perm_hash,
            **{k: stats.get(k, 0) for k in ("raw_blocks", "published_blocks", "rejected_blocks",
                                             "merged_blocks", "evidence_rows", "rejected_rows")},
            "quality_reject_ratio": ratio, "orphan_value_ratio": orphan, "detail": detail or plan.reason,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

    for plan in plans:
        log["actions"][plan.action] = log["actions"].get(plan.action, 0) + 1
        # Each document's manifest row is written right after its partition, so a crash
        # mid-run never leaves a published partition the manifest does not know about.
        if plan.action in ("MISSING", "HASH_MISMATCH"):
            log["errors"].append(f"{plan.source_key}: {plan.reason}")
            wh.append_manifest([manifest_row(plan, "error")])
        elif plan.action in ("NEW", "CHANGED"):
            src = plan.source
            middle_path, parsed = ensure_parsed(src["local_path"], plan.file_sha256, Path(paths["parsed_dir"]),
                                                cfg["parse"]["tier"])
            log["parsed"] += int(parsed)
            result = build_document(
                Source(plan.source_key, plan.file_sha256, src["local_path"], src["file_name"]),
                load_middle(middle_path), permissions, run_id, extractor)
            ok, ratio, orphan, detail = audit(result, max_ratio, max_orphan)
            stage_and_publish(wh, run_id, plan.source_key, _rows_df(result.evidence), _rows_df(result.rejected), ok)
            wh.append_manifest([manifest_row(plan, "published" if ok else "audit_failed", result.stats,
                                             ratio, orphan, detail)])
            if ok:
                live_changed = True
                log["published_rows"] += len(result.evidence)
                log["rejected_rows"] += len(result.rejected)
            else:
                log["errors"].append(f"{plan.source_key}: {detail}; previous version stays live")
        elif plan.action == "RETAG":
            prev = wh.live_state()[plan.source_key]
            ev = pd.read_parquet(wh.partition("evidence", plan.source_key))
            rj = pd.read_parquet(wh.partition("rejected", plan.source_key))
            new_levels = pd.Series(
                [(p.value if (p := gates.resolve_permission(s, c, permissions)) else None)
                 for s, c in zip(ev["section_title"], ev["content"])], index=ev.index)
            untagged = new_levels.isna()
            changed = int((new_levels[~untagged] != ev.loc[~untagged, "permission_level"]).sum())
            if untagged.any():  # Gate 2 again: a rule change must not leave rows untagged
                extra = ev[untagged].assign(gate="G2", reject_reason=RejectReason.NO_PERMISSION_TAG.value,
                                            detail="retag: no rule matched", block_type=None)
                rj = pd.concat([rj, extra[[c for c in rj.columns if c in extra.columns]]], ignore_index=True)
            before = {refs[0] for refs in ev["raw_refs"]}
            # run_id stays: it names the build that produced the content. tag_run_id records the retag.
            ev = ev[~untagged].assign(permission_level=new_levels[~untagged], tag_run_id=run_id)
            lost = len(before - {refs[0] for refs in ev["raw_refs"]})  # blocks left with no published row
            stage_and_publish(wh, run_id, plan.source_key, ev, rj, True)
            stats = {k: int(prev[k]) for k in ("raw_blocks", "published_blocks", "rejected_blocks", "merged_blocks")}
            stats["published_blocks"] -= lost
            stats["rejected_blocks"] += lost
            stats |= {"evidence_rows": len(ev), "rejected_rows": len(rj)}
            wh.append_manifest([manifest_row(plan, "published", stats, float(prev["quality_reject_ratio"]),
                                             float(pd.Series([prev.get("orphan_value_ratio")]).fillna(0.0)[0]),
                                             f"retag: {changed} rows changed level")])
            live_changed = True
            log["retagged_rows"] += changed
        elif plan.action == "REMOVED":
            # Retired partitions are moved aside, not deleted: the rejected rows are part of
            # the audit trail ("no evidence disappears silently").
            for kind in ("evidence", "rejected"):
                part = wh.partition(kind, plan.source_key).parent
                if part.exists():
                    dest = wh.root / "retired" / run_id / kind / part.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(part, dest)
            wh.append_manifest([manifest_row(plan, "retired")])
            live_changed = True

    indexes = paths.get("indexes", "indexes")
    if live_changed or snapshot.current_id(indexes, strategy) is None:
        snapshot.build(wh.root, indexes, strategy, run_id, wh.live_state())
        snapshot.publish(indexes, strategy, run_id)
        log["snapshot"] = f"{strategy}/{run_id}"
    staging = wh.root / "staging" / run_id
    if staging.exists() and not any(p.is_file() for p in staging.rglob("*")):
        shutil.rmtree(staging)  # everything was published; keep staging only for failed audits
    return log


def format_log(log: dict) -> str:
    acts = " ".join(f"{k}={v}" for k, v in sorted(log["actions"].items()))
    line = (f"run_id={log['run_id']}  {acts}\n"
            f"parsed={log['parsed']}  published_rows={log['published_rows']:,}  "
            f"rejected_rows={log['rejected_rows']:,}  retagged_rows={log['retagged_rows']:,}  "
            f"snapshot={log['snapshot'] or 'unchanged'}")
    for e in log["errors"]:
        line += f"\nERROR {e}"
    return line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--json", action="store_true", help="print the run log as JSON")
    args = ap.parse_args()
    log = run(args.config)
    print(json.dumps(log, indent=1) if args.json else format_log(log))
    sys.exit(1 if log["errors"] else 0)


if __name__ == "__main__":
    main()
