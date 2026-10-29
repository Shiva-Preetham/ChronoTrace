"""bench: manifest parsing and the offline --synthetic run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chronotrace.bench import bench_synthetic, read_manifest  # noqa: E402

WORK = ROOT / "tests" / "_work" / "bench"


def test_read_manifest_skips_comments(tmp_path):
    m = tmp_path / "manifest.csv"
    m.write_text("case_id,packing_file,return_file,order_id,item_prompt,expected_status,notes\n"
                 "# example,a.mp4,b.mp4,FK-1,box,GENUINE_RETURN,\n"
                 "\n"
                 "c1,p.mp4,r.mp4,FK-2,earbuds case,SWAP_DETECTED,\n")
    cases = read_manifest(m)
    assert [c.case_id for c in cases] == ["c1"]
    assert cases[0].packing == tmp_path / "p.mp4" and cases[0].item_prompt == "earbuds case"


def test_read_manifest_rejects_missing_fields(tmp_path):
    m = tmp_path / "manifest.csv"
    m.write_text("case_id,packing_file,return_file,order_id,item_prompt,expected_status,notes\n"
                 "c1,p.mp4,,FK-2,box,GENUINE_RETURN,\n")
    with pytest.raises(ValueError, match="return_file"):
        read_manifest(m)


def test_template_manifest_has_no_cases():
    assert read_manifest(ROOT / "data" / "clips" / "manifest.csv") == []


def test_bench_synthetic():
    csv_path, summary = bench_synthetic(WORK, log=lambda *_: None)
    text = summary.read_text(encoding="utf-8")
    assert "Correct verdicts: 3 of 3" in text
    assert "Estimated API cost: $0.000" in text
    assert "not real-world accuracy" in text
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4 and all(r.split(",")[3] == "True" for r in rows[1:])
