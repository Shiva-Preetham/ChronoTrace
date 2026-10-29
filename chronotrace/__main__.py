"""CLI.

  python -m chronotrace check
  python -m chronotrace verify packing.mp4 return.mp4 --order FK-88314 --item "chocolate bar"
  python -m chronotrace verify packing.mp4 return.mp4 --no-spark      # SAM + QR only
  python -m chronotrace verify a.mp4 b.mp4 --refresh-prompt "packing tape"   # re-query one prompt
  python -m chronotrace bench data/clips/manifest.csv
  python -m chronotrace bench --synthetic                              # offline, no key needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import SAM_MODEL, TRANSCRIBE_MODEL, Settings, load_dotenv, spark_model


def check_report(ids: list[str], spark: str) -> list[str]:
    """Lines saying which models ChronoTrace needs are present, and what to do if not."""
    from .api import pick_spark_model

    lines = []
    for want in (SAM_MODEL, spark, TRANSCRIBE_MODEL):
        lines.append(f"{want:28s} {'present' if want in ids else 'MISSING'}")
    if SAM_MODEL not in ids:
        lines.append(f"{SAM_MODEL} is required for tracking. Nothing works without it.")
    if spark not in ids:
        alt = pick_spark_model(ids, spark)
        lines.append(f"Spark fallback: set CHRONOTRACE_SPARK_MODEL={alt} (it will also be picked "
                     "automatically)." if alt else
                     "No muse-spark-* model on this key. Use --no-spark (SAM + QR only).")
    return lines


def cmd_check() -> None:
    from . import api

    try:
        c = api.client()
        ids = sorted(m.id for m in api.with_retries(c.models.list, "List models"))
    except api.ApiError as e:
        sys.exit(str(e))
    print("API access OK. Models on your key:")
    for i in ids:
        print("  ", i)
    print()
    for line in check_report(ids, spark_model()):
        print(line)


def cmd_bench(args) -> None:
    from . import api, bench

    try:
        if args.synthetic:
            csv_path, summary = bench.bench_synthetic(args.out or Path("out/bench_synthetic"))
        else:
            csv_path, summary = bench.bench_manifest(args.manifest, args.out or Path("out/bench"),
                                                     use_spark=not args.no_spark)
    except (api.ApiError, ValueError, FileNotFoundError) as e:
        sys.exit(f"Stopped: {e}")
    print(f"\nResults: {csv_path}\nSummary: {summary}\n")
    print(summary.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(prog="chronotrace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="verify API access and model names")
    v = sub.add_parser("verify", help="compare a packing video with a return video")
    v.add_argument("packing", type=Path)
    v.add_argument("returned", type=Path)
    v.add_argument("--order", default="UNKNOWN")
    v.add_argument("--item", default=Settings.item_prompt, help="SAM noun phrase for the item")
    v.add_argument("--box", default=Settings.container_prompt)
    v.add_argument("--seal", default=Settings.seal_prompt)
    v.add_argument("--out", type=Path, default=Path("out"))
    v.add_argument("--no-spark", action="store_true")
    v.add_argument("--refresh", action="store_true", help="ignore all cached SAM output")
    v.add_argument("--refresh-prompt", action="append", default=[], metavar="PROMPT",
                   help="re-query SAM for this prompt only (repeatable)")
    b = sub.add_parser("bench", help="run every case in a manifest and score the verdicts")
    b.add_argument("manifest", type=Path, nargs="?", default=Path("data/clips/manifest.csv"))
    b.add_argument("--synthetic", action="store_true",
                   help="generated clips + mock SAM, no API calls")
    b.add_argument("--out", type=Path, default=None)
    b.add_argument("--no-spark", action="store_true")
    args = ap.parse_args()
    load_dotenv()

    if args.cmd == "check":
        cmd_check()
        return
    if args.cmd == "bench":
        cmd_bench(args)
        return

    from .identity import SparkComparator
    from .pipeline import run
    from .sam import CachedSam, MetaSam

    s = Settings(container_prompt=args.box, item_prompt=args.item, seal_prompt=args.seal,
                 out_dir=args.out / args.order, use_spark=not args.no_spark)
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    sam = CachedSam(MetaSam(s.cache_dir), s.cache_dir, refresh=args.refresh)
    spark = SparkComparator(s, s.cache_dir) if s.use_spark else None
    res = run(args.packing, args.returned, args.order, s, sam, spark)
    print(f"\nReport: {res.report_path}\nEvidence card: {res.card_path}")


if __name__ == "__main__":
    main()
