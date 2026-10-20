"""Download every source in data/sources.yaml that is not on disk, then check pinned hashes.

    python -m finrag.fetch
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import yaml

from finrag.evidence.publish import sha256_file


def main(sources_path: str = "data/sources.yaml") -> int:
    bad = 0
    for s in yaml.safe_load(open(sources_path)):
        path = Path(s["local_path"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(s["url"], headers={"User-Agent": "Mozilla/5.0"})
            path.write_bytes(urllib.request.urlopen(req, timeout=120).read())
            print(f"fetched {s['source_key']}")
        sha = sha256_file(path)
        ok = s.get("sha256") in (None, sha)
        bad += not ok
        print(f"{s['source_key']:<16} {sha[:12]}  {'ok' if ok else 'HASH MISMATCH (pinned ' + s['sha256'][:12] + ')'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
