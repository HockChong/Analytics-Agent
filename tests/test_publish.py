"""Incremental refresh behaviour (docs/ARCHITECTURE.md §4.5) on a temporary warehouse."""
import shutil
from pathlib import Path

import pandas as pd
import yaml

from finrag.evidence.publish import Warehouse, run
from finrag.index import snapshot
from tests.conftest import AR_MIDDLE, AR_PDF, AR_SHA, needs_report


def _setup(tmp_path: Path, extra_sources=()):
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(AR_PDF, raw / AR_PDF.name)
    parsed = tmp_path / "parsed" / AR_SHA / "flash"
    parsed.mkdir(parents=True)
    shutil.copy(AR_MIDDLE, parsed / "middle.json")
    sources = [{"source_key": "CICT_AR_FY2025", "file_name": AR_PDF.name, "local_path": str(raw / AR_PDF.name),
                "sha256": AR_SHA, "enabled": True}, *extra_sources]
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(sources))
    shutil.copy("config/permissions.yaml", tmp_path / "permissions.yaml")
    cfg = {
        "paths": {"sources": str(tmp_path / "sources.yaml"), "raw_dir": str(raw), "parsed_dir": str(tmp_path / "parsed"),
                  "warehouse": str(tmp_path / "wh"), "permissions": str(tmp_path / "permissions.yaml"),
                  "indexes": str(tmp_path / "indexes")},
        "parse": {"tier": "flash"},
        "evidence": {"max_reject_ratio": 0.15, "max_orphan_ratio": 0.35},
        "index": {"strategy": "table_aware"},
    }
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(cfg))
    return str(tmp_path / "pipeline.yaml"), Warehouse(tmp_path / "wh")


@needs_report
def test_incremental_refresh(tmp_path):
    cfg, wh = _setup(tmp_path)
    indexes = tmp_path / "indexes"

    first = run(cfg, run_id="r1")
    assert first["actions"] == {"NEW": 1} and first["published_rows"] > 3000
    m = wh.manifest().iloc[-1]
    assert m.raw_blocks == m.published_blocks + m.rejected_blocks + m.merged_blocks  # reconciliation
    snap = snapshot.load(indexes, "table_aware")
    assert snap.build_id == "r1" and len(snap.evidence) == first["published_rows"]
    public = snap.evidence[snap.allowed("public", {"public": ["public"]})]
    assert set(public.permission_level) == {"public"}
    assert not public.content.str.contains("(?i)related part").any()

    # Idempotency: a second run does no work and keeps serving the same snapshot.
    second = run(cfg, run_id="r2")
    assert second["actions"] == {"UNCHANGED": 1} and second["snapshot"] is None
    assert second["parsed"] == second["published_rows"] == 0
    assert set(pd.read_parquet(wh.partition("evidence", "CICT_AR_FY2025"))["run_id"]) == {"r1"}
    assert snapshot.current_id(indexes, "table_aware") == "r1"

    # Retag only: a permission rule change re-tags rows without a rebuild, keeps the
    # content lineage (run_id) and serves a new snapshot; the old one is left untouched.
    perms = yaml.safe_load(open(tmp_path / "permissions.yaml"))
    perms["rules"].append({"section": "^Financial Highlights", "level": "internal"})
    (tmp_path / "permissions.yaml").write_text(yaml.safe_dump(perms))
    third = run(cfg, run_id="r3")
    assert third["actions"] == {"RETAG": 1} and third["published_rows"] == 0 and third["retagged_rows"] > 0
    ev = pd.read_parquet(wh.partition("evidence", "CICT_AR_FY2025"))
    assert set(ev.loc[ev.section_title.str.startswith("Financial Highlights"), "permission_level"]) == {"internal"}
    assert set(ev.run_id) == {"r1"} and set(ev.tag_run_id) == {"r3"}
    assert snapshot.current_id(indexes, "table_aware") == "r3"
    old = snapshot.load(indexes, "table_aware", "r1").evidence
    assert set(old.loc[old.section_title.str.startswith("Financial Highlights"), "permission_level"]) == {"public"}

    # Quarantine: a file that no longer matches its pinned hash is refused; the live
    # partition is untouched.
    pdf = Path(yaml.safe_load(open(tmp_path / "sources.yaml"))[0]["local_path"])
    pdf.write_bytes(pdf.read_bytes() + b"tampered")
    fourth = run(cfg, run_id="r4")
    assert fourth["actions"] == {"HASH_MISMATCH": 1} and fourth["errors"]
    assert len(pd.read_parquet(wh.partition("evidence", "CICT_AR_FY2025"))) == len(ev)

    # Removal: disabling a source retires its partition.
    srcs = yaml.safe_load(open(tmp_path / "sources.yaml"))
    srcs[0]["enabled"] = False
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(srcs))
    fifth = run(cfg, run_id="r5")
    assert fifth["actions"] == {"REMOVED": 1}
    assert not wh.partition("evidence", "CICT_AR_FY2025").exists()
    assert (wh.root / "retired" / "r5" / "rejected" / "source_key=CICT_AR_FY2025" / "rejected.parquet").exists()
    assert wh.manifest().iloc[-1].status == "retired"
    assert len(snapshot.load(indexes, "table_aware").evidence) == 0  # retired rows are no longer served
