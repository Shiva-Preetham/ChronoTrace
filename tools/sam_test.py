"""
ChronoTrace - SAM 3.1 first-contact test.

What it does:
  1. (--check) Confirms your API key works and lists available model IDs.
  2. Uploads a video once (cached by file hash), runs sam-3.1 once PER concept prompt
     (the API supports one concept per request), and caches the raw output in cache/,
     the same cache the pipeline uses, so re-runs and later `verify` runs cost nothing.
  3. Prints a per-object track summary (first/last frame, seconds, frames seen) and
     writes an overlay MP4 so you can eyeball the tracking.

Usage (from the repo root):
  python tools/sam_test.py --check
  python tools/sam_test.py clip.mp4 --prompts "cardboard box" "earbuds case" "packing tape"
  python tools/sam_test.py clip.mp4 --prompts "box" --refresh   # ignore cache for these prompts

Docs this follows: dev.meta.ai/docs/sam/segmenting and /sam/reading-segmentation
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chronotrace import api, video  # noqa: E402
from chronotrace.__main__ import cmd_check  # noqa: E402
from chronotrace.config import load_dotenv  # noqa: E402
from chronotrace.sam import CachedSam, MetaSam, file_sha256  # noqa: E402
from chronotrace.tracks import Track  # noqa: E402

CACHE = Path("cache")


def summarize(track: Track, fps: float, n_frames: int) -> None:
    print(f"\n=== '{track.prompt}' ===")
    if not track:
        print("  ZERO matches. Try a plainer noun phrase (e.g. 'box' instead of 'shipping carton').")
        return
    idx = track.frames()
    print(f"  frames with detections: {len(idx)} / {n_frames} "
          f"(range {idx[0]}..{idx[-1]}; gaps mean nothing visible on those frames)")
    by_id: dict[int, list] = {}
    for d in track.all():
        by_id.setdefault(d.obj_id, []).append(d)
    for oid, ds in sorted(by_id.items()):
        confs = [d.conf for d in ds if d.conf is not None]
        avg_c = f"{sum(confs) / len(confs):.2f}" if confs else "n/a"
        print(f"  id {oid}: frames {ds[0].frame}..{ds[-1].frame} "
              f"({ds[0].frame / fps:.2f}s..{ds[-1].frame / fps:.2f}s), seen on {len(ds)} frames, "
              f"avg conf {avg_c}")
    if idx[-1] > n_frames:
        print("  NOTE: frame indices exceed OpenCV's frame count - check fps/frame mapping.")


def write_overlay(path: Path, tracks: list[Track], out: Path) -> None:
    colors = [(255, 120, 0), (0, 200, 0), (180, 0, 255), (0, 200, 255), (0, 0, 255)]
    vi = video.info(path)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), vi.fps, (vi.width, vi.height))
    warned = False
    for f, frame in video.frames(path):     # already rotated to display orientation
        for pi, t in enumerate(tracks):
            for d in t.at(f):
                if not warned and (d.w > d.h) != (vi.width > vi.height):
                    print("WARNING: SAM frame size and the rotated video differ in aspect. "
                          "Overlay may be misplaced.")
                    warned = True
                x1, y1, x2, y2 = d.scaled(vi.width, vi.height)
                col = colors[pi % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{t.prompt}#{d.obj_id}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(frame, f"frame {f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--prompts", nargs="+", default=["cardboard box"])
    ap.add_argument("--check", action="store_true", help="verify API access and list models")
    ap.add_argument("--refresh", action="store_true", help="ignore cached SAM output for these prompts")
    args = ap.parse_args()
    load_dotenv()

    if args.check:
        cmd_check()
        if not args.video:
            return
    if not args.video:
        ap.error("give a video path, or use --check")

    path = Path(args.video)
    vi = video.info(path)
    print(f"{path.name}: {vi.width}x{vi.height} (rotation {vi.rotation}), {vi.fps:.2f} fps, "
          f"{vi.n_frames} frames ({vi.seconds:.1f}s)")
    print(f"sha256: {file_sha256(path)}")

    try:
        sam = CachedSam(MetaSam(CACHE), CACHE, refresh=args.refresh)
        tracks = []
        for prompt in args.prompts:
            cached = sam.cache_path(path, prompt).exists() and not args.refresh
            print(f"[{'cache' if cached else 'api'}] '{prompt}'")
            t = sam.track(path, prompt, vi.n_frames)
            tracks.append(t)
            summarize(t, vi.fps, vi.n_frames)
    except (api.ApiError, api.BudgetExceeded) as e:
        sys.exit(f"Stopped: {e}")

    overlay = path.with_name(path.stem + "_overlay.mp4")
    write_overlay(path, tracks, overlay)
    print(f"\nOverlay video: {overlay}")
    print(f"Estimated API spend this run: ${sam.estimated_spend:.3f}")


if __name__ == "__main__":
    main()
