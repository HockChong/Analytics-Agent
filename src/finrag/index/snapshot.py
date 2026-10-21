"""Versioned serving snapshot (docs/ARCHITECTURE.md §4.6).

The query side reads exactly one thing: the snapshot named by `indexes/<strategy>/CURRENT`.
A snapshot is a directory written once and never changed:

    indexes/<strategy>/<build_id>/
        evidence.parquet   every published row of every live document, with permission_level
        meta.json          build_id, strategy, per-document file_sha256 and run_id, row counts
        (D4 adds bm25.pkl and vectors.npy here, built from evidence.parquet)

Publishing a snapshot is one os.replace of the CURRENT pointer. A RETAG, a REMOVED document
or a rebuilt document produces a new snapshot, so a query can never mix evidence, BM25 and
vectors from different builds, or retrieve a row that was retired after its snapshot was
taken. `AnswerRecord.build_id` records which snapshot served each answer.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class Snapshot:
    path: Path
    meta: dict
    evidence: pd.DataFrame

    @property
    def build_id(self) -> str:
        return self.meta["build_id"]

    def allowed(self, role: str, roles: dict[str, list[str]]) -> pd.Series:
        """Row mask for a user role. Retrieval applies it before ranking in both branches."""
        if role not in roles:
            raise ValueError(f"unknown role {role!r}; expected one of {sorted(roles)}")
        return self.evidence["permission_level"].isin(roles[role])


def build(warehouse_root: str | Path, indexes_root: str | Path, strategy: str, build_id: str,
          live: dict[str, dict]) -> Path:
    """Write a new snapshot from the live evidence partitions. Does not publish it."""
    root = Path(indexes_root) / strategy
    final = root / build_id
    tmp = root / f".{build_id}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    parts = [pd.read_parquet(Path(warehouse_root) / "evidence" / f"source_key={k}" / "evidence.parquet")
             for k in sorted(live)]
    evidence = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    evidence.to_parquet(tmp / "evidence.parquet", index=False)
    meta = {
        "build_id": build_id,
        "strategy": strategy,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": {k: {"file_sha256": v["file_sha256"], "run_id": v["run_id"]} for k, v in sorted(live.items())},
        "rows": len(evidence),
        "permission_levels": evidence["permission_level"].value_counts().to_dict() if len(evidence) else {},
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return final


def publish(indexes_root: str | Path, strategy: str, build_id: str) -> None:
    root = Path(indexes_root) / strategy
    if not (root / build_id / "meta.json").exists():
        raise FileNotFoundError(f"no snapshot {build_id} under {root}")
    tmp = root / "CURRENT.tmp"
    tmp.write_text(build_id, encoding="utf-8")
    os.replace(tmp, root / "CURRENT")


def current_id(indexes_root: str | Path, strategy: str) -> str | None:
    pointer = Path(indexes_root) / strategy / "CURRENT"
    return pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None


def load(indexes_root: str | Path, strategy: str, build_id: str | None = None) -> Snapshot:
    """Load the CURRENT snapshot (or a named one). A long-running app calls current_id()
    between queries and reloads when it changes; it never reads warehouse/ directly."""
    build_id = build_id or current_id(indexes_root, strategy)
    if build_id is None:
        raise FileNotFoundError(f"no CURRENT snapshot for strategy {strategy!r}; run make build")
    path = Path(indexes_root) / strategy / build_id
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return Snapshot(path=path, meta=meta, evidence=pd.read_parquet(path / "evidence.parquet"))
