"""End-to-end on synthetic clips (mock SAM, fake Spark) plus unit tests for the verdict.

Run:  python -m pytest -q      (or: python tests/test_pipeline.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from chronotrace.config import Settings  # noqa: E402
from chronotrace.events import Check, VideoFindings  # noqa: E402
from chronotrace.pipeline import run  # noqa: E402
from chronotrace.sam import CachedSam, parse_sam_output  # noqa: E402
from chronotrace.verdict import decide  # noqa: E402
from synthetic import FakeSpark, MockSam, build_all  # noqa: E402

WORK = ROOT / "tests" / "_work"


def _setup():
    clips = WORK / "clips"
    truth = build_all(clips) if not (clips / "truth.json").exists() else json.loads(
        (clips / "truth.json").read_text())
    s = Settings(cache_dir=WORK / "cache")
    prompts = {s.container_prompt: "box", s.item_prompt: "item", s.seal_prompt: "seal"}
    return clips, truth, s, prompts


def _run(name: str, spark_result: dict):
    clips, truth, s, prompts = _setup()
    s.out_dir = WORK / "out" / name
    sam = CachedSam(MockSam(truth, prompts), s.cache_dir, refresh=True)
    return run(clips / f"{name}_pack.mp4", clips / f"{name}_return.mp4", f"TEST-{name}", s, sam,
               FakeSpark(spark_result), log=lambda *_: None).report


def test_parser_documented_example():
    line = ("<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,60,"
            "~!!!!M!0c[0o91w?q1!pIH4|>,1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
            "<|mask;x=0;y=0;data=27,48,~!!!!J!0c|>")
    dets = parse_sam_output([line])
    assert [(d.frame, d.obj_id, d.x1) for d in dets] == [(0, 0, 211), (0, 1, 155)]


def test_genuine_return():
    rep = _run("genuine", {"same_unit": "yes", "visible_differences": [],
                           "new_damage_on_return": False})
    assert rep["status"] == "GENUINE_RETURN", rep
    assert rep["qr"]["shipped"] and rep["qr"]["shipped"] == {k: v for k, v in rep["qr"]["shipped"].items()}
    assert "CT-A7F3K" in rep["qr"]["returned"]
    ins = rep["packing"]["events"]["inserted"]["frame"]
    seal = rep["packing"]["events"]["sealed"]["frame"]
    assert 85 <= ins <= 110, ins          # item reaches the box ~frame 90-100
    assert 148 <= seal <= 160, seal       # tape appears at frame 150


def test_swap_detected():
    rep = _run("swap", {"same_unit": "no", "visible_differences": ["red stripe on returned item"],
                        "new_damage_on_return": True})
    assert rep["status"] == "SWAP_DETECTED", rep
    assert "CT-A7F3K" in rep["qr"]["shipped"] and "CT-B2M9Q" in rep["qr"]["returned"]


def test_seller_at_fault():
    rep = _run("fault", {"same_unit": "yes", "visible_differences": [],
                         "new_damage_on_return": False})
    assert rep["status"] == "PACKING_NOT_PROVEN", rep


def _findings(role, checks):
    f = VideoFindings(role, 30.0)
    f.checks = [Check(n, p, n) for n, p in checks]
    return f


GOOD_PACK = [("item_inserted", True), ("inserted_before_sealed", True), ("not_removed_after_seal", True)]
GOOD_RET = [("seal_on_arrival", True), ("item_from_box", True)]


def test_verdict_branches():
    p, r = _findings("packing", GOOD_PACK), _findings("return", GOOD_RET)
    assert decide(p, r, {"A": 1}, {"A": 5}, None).status == "GENUINE_RETURN"
    assert decide(p, r, {"A": 1}, {"B": 5}, None).status == "SWAP_DETECTED"
    assert decide(p, r, {"A": 1}, {"A": 5}, {"same_unit": "yes", "new_damage_on_return": True,
                                             "visible_differences": ["crack"]}).status == "RETURNED_WITH_DAMAGE"
    assert decide(p, r, {"A": 1}, {}, {"same_unit": "no", "visible_differences": ["colour"]}).status == "SWAP_LIKELY"
    assert decide(p, r, {"A": 1}, {}, {"same_unit": "uncertain"}).status == "NEEDS_REVIEW"
    assert decide(p, r, {}, {}, {"same_unit": "no", "visible_differences": ["x"]}).status == "SWAP_LIKELY"
    assert decide(p, r, {}, {}, None).status == "NEEDS_REVIEW"
    bad = _findings("packing", [("item_inserted", False)])
    assert decide(bad, r, {"A": 1}, {"A": 1}, None).status == "PACKING_NOT_PROVEN"
    unsealed = _findings("return", [("seal_on_arrival", False), ("item_from_box", True)])
    v = decide(p, unsealed, {"A": 1}, {"A": 1}, None)
    assert v.status == "GENUINE_RETURN" and any("seal" in n.lower() for n in v.review_notes)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
