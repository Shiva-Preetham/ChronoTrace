"""
ChronoTrace - SAM 3.1 first-contact test.

What it does:
  1. (--check) Confirms your API key works from India and lists available model IDs.
  2. Uploads a video once (cached by file hash), runs sam-3.1 once PER concept prompt
     (the API supports one concept per request), and caches the raw output to disk
     so re-runs cost nothing.
  3. Parses boxes per frame, prints a per-object track summary (first/last frame,
     seconds, frames seen), saves tracks JSON, and writes an overlay MP4 so you can
     eyeball the tracking.

Setup:
  pip install openai opencv-python
  set MODEL_API_KEY=...            (Windows)   |   export MODEL_API_KEY=...  (Mac/Linux)

Usage:
  python sam_test.py --check
  python sam_test.py clip.mp4 --prompts "cardboard box" "earbuds case" "packing tape"
  python sam_test.py clip.mp4 --prompts "cardboard box" --refresh   # ignore cache

Docs this follows: dev.meta.ai/docs/sam/segmenting and /sam/reading-segmentation
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import cv2
from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
CACHE = Path("sam_cache")
PRICE_PER_FRAME = 0.20 / 1000   # SAM 3.1 video, USD
MIN_BILLED_FRAMES = 50          # per request, per docs

FRAME_RE = re.compile(r"^<(\d+)f>")
# Box fields are strictly numeric, so this can't false-match inside base85 mask payloads.
REC_RE = re.compile(
    r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>"
)


def client() -> OpenAI:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        sys.exit("MODEL_API_KEY is not set.")
    return OpenAI(base_url=BASE_URL, api_key=key)


def check_access():
    c = client()
    ids = sorted(m.id for m in c.models.list())
    print("API access OK. Models available to your key:")
    for i in ids:
        print("  ", i)
    for want in ("sam-3.1", "muse-spark-1.3", "muse-voice-transcribe-1.0"):
        print(f"{want:28s} {'present' if want in ids else 'MISSING'}")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_once(c: OpenAI, path: Path, digest: str) -> str:
    id_file = CACHE / f"{digest[:16]}.file_id"
    if id_file.exists():
        return id_file.read_text().strip()
    with open(path, "rb") as f:
        fid = c.files.create(purpose="user_data", file=f).id
    id_file.write_text(fid)
    print(f"Uploaded {path.name} -> {fid}")
    return fid


def run_sam(c: OpenAI, file_id: str, prompt: str, out: Path) -> list[str]:
    """Stream one concept prompt; each output_text delta is one frame line."""
    deltas = []
    with c.responses.stream(
        model="sam-3.1",
        input=[{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": file_id},
            ],
        }],
        metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                deltas.append(event.delta)
    out.write_text(json.dumps(deltas))
    return deltas


def parse(deltas: list[str]) -> dict[int, list[dict]]:
    """frame_index -> [{id, x1, y1, x2, y2, w, h, c}]"""
    frames: dict[int, list[dict]] = {}
    buf = "".join(deltas)
    # Each frame line starts with <Nf>; split on line boundaries the stream gives us.
    lines = deltas if all(FRAME_RE.match(d) for d in deltas if d.strip()) else buf.splitlines()
    for line in lines:
        m = FRAME_RE.match(line.strip())
        if not m:
            continue
        f = int(m.group(1))
        for r in REC_RE.finditer(line):
            frames.setdefault(f, []).append({
                "id": int(r.group(1)),
                "x1": int(r.group(2)), "y1": int(r.group(3)),
                "x2": int(r.group(4)), "y2": int(r.group(5)),
                "w": int(r.group(6)), "h": int(r.group(7)),
                "c": float(r.group(8)) if r.group(8) else None,
            })
    return frames


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def summarize(prompt: str, frames: dict, fps: float, n_frames: int):
    print(f"\n=== '{prompt}' ===")
    if not frames:
        print("  ZERO matches. Try a plainer noun phrase (e.g. 'box' instead of 'shipping carton').")
        return
    idx = sorted(frames)
    print(f"  frames with detections: {len(idx)} / {n_frames} "
          f"(range {idx[0]}..{idx[-1]}; gaps mean nothing visible on those frames)")
    tracks: dict[int, list[int]] = {}
    for f in idx:
        for r in frames[f]:
            tracks.setdefault(r["id"], []).append(f)
    for oid, fs in sorted(tracks.items()):
        confs = [r["c"] for f in fs for r in frames[f] if r["id"] == oid and r["c"] is not None]
        avg_c = f"{sum(confs)/len(confs):.2f}" if confs else "n/a"
        print(f"  id {oid}: frames {fs[0]}..{fs[-1]} "
              f"({fs[0]/fps:.2f}s..{fs[-1]/fps:.2f}s), seen on {len(fs)} frames, avg conf {avg_c}")
    if idx[-1] > n_frames:
        print("  NOTE: frame indices exceed OpenCV's frame count - check fps/frame mapping.")


def write_overlay(path: Path, results: dict[str, dict], out: Path):
    colors = [(255, 120, 0), (0, 200, 0), (180, 0, 255), (0, 200, 255), (0, 0, 255)]
    cap = cv2.VideoCapture(str(path))
    fps, _, W, H = video_info(path)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    f = 0
    warned = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for pi, (prompt, frames) in enumerate(results.items()):
            for r in frames.get(f, []):
                sx, sy = W / r["w"], H / r["h"]
                if not warned and abs(sx - sy) > 0.05:
                    print("WARNING: SAM frame size and OpenCV frame size differ in aspect "
                          "(phone rotation flag?). Overlay may be misplaced.")
                    warned = True
                p1 = (int(r["x1"] * sx), int(r["y1"] * sy))
                p2 = (int(r["x2"] * sx), int(r["y2"] * sy))
                col = colors[pi % len(colors)]
                cv2.rectangle(frame, p1, p2, col, 2)
                cv2.putText(frame, f"{prompt}#{r['id']}", (p1[0], max(p1[1] - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(frame, f"frame {f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        f += 1
    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--prompts", nargs="+", default=["cardboard box"])
    ap.add_argument("--check", action="store_true", help="verify API access and list models")
    ap.add_argument("--refresh", action="store_true", help="ignore cached SAM output")
    args = ap.parse_args()

    if args.check:
        check_access()
        if not args.video:
            return
    if not args.video:
        ap.error("give a video path, or use --check")

    path = Path(args.video)
    CACHE.mkdir(exist_ok=True)
    c = client()
    digest = file_hash(path)
    fps, n_frames, W, H = video_info(path)
    print(f"{path.name}: {W}x{H}, {fps:.2f} fps, {n_frames} frames ({n_frames/fps:.1f}s)")
    print(f"sha256: {digest}")

    file_id = upload_once(c, path, digest)
    results, spent = {}, 0.0
    for prompt in args.prompts:
        slug = re.sub(r"\W+", "_", prompt.lower()).strip("_")
        raw = CACHE / f"{digest[:16]}_{slug}.json"
        if raw.exists() and not args.refresh:
            deltas = json.loads(raw.read_text())
            print(f"[cache] '{prompt}'")
        else:
            print(f"[api]   '{prompt}' ...")
            deltas = run_sam(c, file_id, prompt, raw)
            spent += max(n_frames, MIN_BILLED_FRAMES) * PRICE_PER_FRAME
        frames = parse(deltas)
        results[prompt] = frames
        summarize(prompt, frames, fps, n_frames)

    tracks_out = CACHE / f"{digest[:16]}_tracks.json"
    tracks_out.write_text(json.dumps({p: {str(k): v for k, v in fr.items()} for p, fr in results.items()}))
    overlay = path.with_name(path.stem + "_overlay.mp4")
    write_overlay(path, results, overlay)
    print(f"\nTracks: {tracks_out}\nOverlay video: {overlay}")
    print(f"Estimated API spend this run: ${spent:.3f}")


if __name__ == "__main__":
    main()
