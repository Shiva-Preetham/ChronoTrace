"""CLI.

  python -m chronotrace check
  python -m chronotrace verify packing.mp4 return.mp4 --order FK-88314 --item "chocolate bar"
  python -m chronotrace verify packing.mp4 return.mp4 --no-spark      # SAM + QR only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import BASE_URL, Settings


def cmd_check() -> None:
    from openai import OpenAI

    key = os.environ.get("MODEL_API_KEY")
    if not key:
        sys.exit("MODEL_API_KEY is not set.")
    ids = sorted(m.id for m in OpenAI(base_url=BASE_URL, api_key=key).models.list())
    print("API access OK. Models on your key:")
    for i in ids:
        print("  ", i)
    for want in ("sam-3.1", "muse-spark-1.3", "muse-voice-transcribe-1.0"):
        print(f"{want:28s} {'present' if want in ids else 'MISSING'}")


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
    v.add_argument("--refresh", action="store_true", help="ignore cached SAM output")
    args = ap.parse_args()

    if args.cmd == "check":
        cmd_check()
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
