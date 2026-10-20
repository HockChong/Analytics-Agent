from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AR_PDF = ROOT / "data/raw/2_CICT_AR_2025_Final.pdf"
AR_SHA = "03b4ffd55c33cce440cdf5ed27143f61473a8f874d97b41d5e1d6c01217e7ad0"
AR_MIDDLE = ROOT / f"data/parsed/{AR_SHA}/middle.json"

# Source PDFs are not committed (copyright); tests that need them skip until `make fetch`.
needs_report = pytest.mark.skipif(
    not (AR_PDF.exists() and AR_MIDDLE.exists()), reason="run `make fetch parse` first"
)
