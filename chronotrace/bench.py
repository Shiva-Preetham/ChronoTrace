"""Run every case in a manifest and score the verdicts against the expected ones.

  python -m chronotrace bench data/clips/manifest.csv
  python -m chronotrace bench --synthetic        # offline: generated clips + mock SAM

Writes results.csv and summary.md. Real-clip numbers in the README come only from here.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import api
from .config import Settings
from .pipeline import run

FIELDS = ["case_id", "packing_file", "return_file", "order_id", "item_prompt",
          "expected_status", "notes"]


@dataclass
class Case:
    case_id: str
    packing: Path
    returned: Path
    order_id: str
    item_prompt: str
    expected: str
    notes: str = ""


@dataclass
class Outcome:
    case: Case
    got: str
    reason: str
    cost: float

    @property
    def correct(self) -> bool:
        return self.got == self.case.expected


def read_manifest(path: Path) -> list[Case]:
    """Parse manifest.csv. Blank lines and lines starting with # are skipped."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    cases = []
    for row in csv.DictReader(io.StringIO("\n".join(lines))):
        missing = [f for f in FIELDS[:6] if not (row.get(f) or "").strip()]
        if missing:
            raise ValueError(f"{path}: row {row.get('case_id') or '?'} is missing {', '.join(missing)}")
        cases.append(Case(
            case_id=row["case_id"].strip(),
            packing=path.parent / row["packing_file"].strip(),
            returned=path.parent / row["return_file"].strip(),
            order_id=row["order_id"].strip(),
            item_prompt=row["item_prompt"].strip(),
            expected=row["expected_status"].strip(),
            notes=(row.get("notes") or "").strip(),
        ))
    return cases


def _miss_reason(rep: dict) -> str:
    failed = [c["detail"] for side in ("packing", "return") for c in rep[side]["checks"]
              if c["passed"] is False]
    parts = rep["reasons"] + [f"failed check: {d}" for d in failed]
    return "; ".join(parts) or rep["headline"]


def run_cases(cases: list[Case], out_dir: Path, make_sam: Callable, make_spark: Callable,
              base: Settings | None = None, log=print) -> tuple[list[Outcome], str | None]:
    """Run each case. Returns outcomes and, if the budget stopped the run, why."""
    outcomes: list[Outcome] = []
    for case in cases:
        log(f"[{len(outcomes) + 1}/{len(cases)}] {case.case_id} (expect {case.expected})")
        s = Settings(item_prompt=case.item_prompt, out_dir=out_dir / case.case_id)
        if base is not None:
            s.container_prompt, s.seal_prompt = base.container_prompt, base.seal_prompt
            s.cache_dir, s.use_spark = base.cache_dir, base.use_spark
        sam = make_sam(s)
        spark = make_spark(s, case)
        try:
            res = run(case.packing, case.returned, case.order_id, s, sam, spark, log=lambda *_: None)
            rep = res.report
            out = Outcome(case, rep["status"], "" if rep["status"] == case.expected
                          else _miss_reason(rep), rep["estimated_api_cost_usd"])
        except api.BudgetExceeded as e:
            return outcomes, str(e)
        except Exception as e:  # noqa: BLE001 - one broken case shouldn't stop the bench
            out = Outcome(case, "ERROR", f"{type(e).__name__}: {e}", sam.estimated_spend)
        log(f"    got {out.got}{'' if out.correct else '  <-- MISS'}")
        outcomes.append(out)
    return outcomes, None


def write_results(outcomes: list[Outcome], out_dir: Path, title: str,
                  stopped: str | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "expected_status", "got_status", "correct", "reason_if_wrong",
                    "estimated_cost_usd"])
        for o in outcomes:
            w.writerow([o.case.case_id, o.case.expected, o.got, o.correct, o.reason, f"{o.cost:.4f}"])

    n, ok = len(outcomes), sum(o.correct for o in outcomes)
    cost = sum(o.cost for o in outcomes)
    expected = sorted({o.case.expected for o in outcomes})
    got = sorted({o.got for o in outcomes})
    md = [f"# {title}", "",
          f"- Cases run: {n}",
          f"- Correct verdicts: {ok} of {n}",
          f"- Estimated API cost: ${cost:.3f}", ""]
    if stopped:
        md += [f"**Stopped early:** {stopped}", ""]
    md += ["## Confusion table (rows: expected, columns: got)", "",
           "| expected \\ got | " + " | ".join(got) + " |",
           "|---|" + "---|" * len(got)]
    for e in expected:
        counts = [sum(1 for o in outcomes if o.case.expected == e and o.got == g) for g in got]
        md.append(f"| {e} | " + " | ".join(str(c) for c in counts) + " |")
    misses = [o for o in outcomes if not o.correct]
    md += ["", "## Misses", ""]
    md += [f"- **{o.case.case_id}**: expected {o.case.expected}, got {o.got}. {o.reason}"
           for o in misses] or ["None."]
    summary = out_dir / "summary.md"
    summary.write_text("\n".join(md) + "\n", encoding="utf-8")
    return csv_path, summary


def bench_manifest(manifest: Path, out_dir: Path, use_spark: bool = True, log=print):
    from .identity import SparkComparator
    from .sam import CachedSam, MetaSam

    cases = read_manifest(manifest)
    if not cases:
        raise ValueError(f"{manifest} has no cases yet. Film clips and fill it in first.")
    base = Settings(use_spark=use_spark)
    base.cache_dir.mkdir(parents=True, exist_ok=True)
    backend = MetaSam(base.cache_dir, log=log)
    spark = SparkComparator(base, base.cache_dir, log=log) if use_spark else None

    def make_sam(s: Settings):
        return CachedSam(backend, s.cache_dir)

    outcomes, stopped = run_cases(cases, out_dir, make_sam, lambda s, c: spark, base, log)
    return write_results(outcomes, out_dir, f"Benchmark: {manifest}", stopped)


def bench_synthetic(out_dir: Path, log=print):
    """The three demo scenarios on generated clips with a mock SAM and no Spark.
    This tests the plumbing and the rules, not real-world accuracy."""
    from .sam import CachedSam
    from .synthetic import SCENARIOS, MockSam, build_all

    clips = out_dir / "clips"
    truth = build_all(clips)
    base = Settings(cache_dir=out_dir / "cache", use_spark=False)
    prompts = {base.container_prompt: "box", base.item_prompt: "item", base.seal_prompt: "seal"}
    cases = [Case(sc.name, clips / f"{sc.name}_pack.mp4", clips / f"{sc.name}_return.mp4",
                  sc.order_id, base.item_prompt, sc.expected, sc.description) for sc in SCENARIOS]

    def make_sam(s: Settings):
        return CachedSam(MockSam(truth, prompts), s.cache_dir, refresh=True, budget=api.Budget())

    outcomes, stopped = run_cases(cases, out_dir, make_sam, lambda s, c: None, base, log)
    title = "Synthetic benchmark (generated clips, mock SAM, no Spark; not real-world accuracy)"
    return write_results(outcomes, out_dir, title, stopped)
