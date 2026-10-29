"""SAM 3.1 over the Meta Model API: upload once, one request per concept, cache everything.

Wire format (dev.meta.ai/docs/sam/reading-segmentation): one line per frame,
  <Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..[;c=..]|><|mask;...|>,id<|box;...|>...
Frames with nothing visible emit no line, so gaps in frame indices are normal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Protocol

from .config import BASE_URL, SAM_MIN_BILLED_FRAMES, SAM_MODEL, SAM_PRICE_PER_FRAME
from .tracks import Det, Track

FRAME_RE = re.compile(r"^<(\d+)f>")
# Box fields are strictly numeric, so this can't false-match inside base85 mask payloads.
REC_RE = re.compile(
    r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>"
)


def parse_sam_output(deltas: list[str]) -> list[Det]:
    """Turn streamed output_text deltas into detections."""
    # Normally each delta is exactly one frame line. If the transport ever splits
    # or merges them, fall back to splitting the joined text on newlines.
    if all(FRAME_RE.match(d.strip()) for d in deltas if d.strip()):
        lines = deltas
    else:
        lines = "".join(deltas).splitlines()
    dets = []
    for line in lines:
        line = line.strip()
        m = FRAME_RE.match(line)
        if not m:
            continue
        frame = int(m.group(1))
        for r in REC_RE.finditer(line):
            dets.append(Det(
                frame=frame, obj_id=int(r.group(1)),
                x1=int(r.group(2)), y1=int(r.group(3)), x2=int(r.group(4)), y2=int(r.group(5)),
                w=int(r.group(6)), h=int(r.group(7)),
                conf=float(r.group(8)) if r.group(8) else None,
            ))
    return dets


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SamBackend(Protocol):
    def segment(self, video: Path, prompt: str) -> list[str]:
        """Return the raw output_text deltas for one concept prompt."""


class MetaSam:
    """Real backend: Meta Model API."""

    def __init__(self, cache_dir: Path, api_key: str | None = None):
        from openai import OpenAI  # imported here so offline tests don't need a key

        key = api_key or os.environ.get("MODEL_API_KEY")
        if not key:
            raise RuntimeError("MODEL_API_KEY is not set")
        self.client = OpenAI(base_url=BASE_URL, api_key=key)
        self.cache_dir = cache_dir

    def _file_id(self, video: Path) -> str:
        digest = file_sha256(video)
        id_file = self.cache_dir / f"{digest[:16]}.file_id"
        if id_file.exists():
            return id_file.read_text().strip()
        with open(video, "rb") as f:
            fid = self.client.files.create(purpose="user_data", file=f).id
        id_file.write_text(fid)
        return fid

    def segment(self, video: Path, prompt: str) -> list[str]:
        file_id = self._file_id(video)
        deltas: list[str] = []
        with self.client.responses.stream(
            model=SAM_MODEL,
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
        return deltas


class CachedSam:
    """Wraps any backend so each (video, prompt) pair is paid for once.

    SAM results vary slightly run to run, so the cache also keeps the demo reproducible.
    """

    def __init__(self, backend: SamBackend, cache_dir: Path, refresh: bool = False):
        self.backend = backend
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.estimated_spend = 0.0
        cache_dir.mkdir(parents=True, exist_ok=True)

    def track(self, video: Path, prompt: str, n_frames: int) -> Track:
        digest = file_sha256(video)[:16]
        slug = re.sub(r"\W+", "_", prompt.lower()).strip("_")
        path = self.cache_dir / f"{digest}_{slug}.json"
        if path.exists() and not self.refresh:
            deltas = json.loads(path.read_text())
        else:
            deltas = self.backend.segment(video, prompt)
            path.write_text(json.dumps(deltas))
            self.estimated_spend += max(n_frames, SAM_MIN_BILLED_FRAMES) * SAM_PRICE_PER_FRAME
        return Track(prompt, parse_sam_output(deltas))
