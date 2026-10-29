"""Is the returned item the same unit that was shipped?

Two independent signals:
  1. QR code on the item, decoded deterministically with OpenCV (no model guessing).
  2. Muse Spark comparing the sharpest crops from both videos (condition, marks, damage).
The verdict logic trusts the QR first and uses Spark for condition and as a fallback.
"""

from __future__ import annotations

import base64
import heapq
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import video
from .config import BASE_URL, SPARK_MODEL, SPARK_PRICE_IN, SPARK_PRICE_OUT, Settings
from .tracks import Det, Track


@dataclass
class Keyframe:
    frame: int
    sharpness: float
    crop: np.ndarray = field(repr=False)


def _crop(frame: np.ndarray, d: Det, pad: float = 0.15) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = d.scaled(W, H)
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(W, x2 + px), min(H, y2 + py)
    return frame[y1:y2, x1:x2].copy()


def _sharpness(crop: np.ndarray, size: int) -> float:
    """Laplacian variance on a fixed-size grayscale crop, so bigger crops don't win
    just by having more edges."""
    if crop.size == 0:
        return 0.0
    g = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def select_keyframes(path: Path, item: Track, s: Settings, frame_range: tuple[int, int] | None = None,
                     keep: int = 8) -> list[Keyframe]:
    """Sharpest, least-occluded item crops. Returns up to `keep`, best first."""
    largest: dict[int, Det] = {}
    for f in item.frames():
        if frame_range and not (frame_range[0] <= f <= frame_range[1]):
            continue
        largest[f] = max(item.at(f), key=lambda d: d.area)
    if not largest:
        return []
    max_area = max(d.area for d in largest.values())
    candidates = {f: d for f, d in largest.items() if d.area >= s.keyframe_min_area_ratio * max_area}

    heap: list[tuple[float, int, np.ndarray]] = []
    last = max(candidates)
    for i, frame in video.frames(path):
        if i in candidates:
            crop = _crop(frame, candidates[i])
            score = _sharpness(crop, s.keyframe_crop_size)
            item_ = (score, i, crop)
            if len(heap) < keep:
                heapq.heappush(heap, item_)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item_)
        if i >= last:
            break
    return [Keyframe(f, sc, c) for sc, f, c in sorted(heap, key=lambda t: -t[0])]


def decode_qr(crops: list[Keyframe]) -> dict[str, int]:
    """QR text -> first frame it was read on. Tries each crop at 1x and 2x."""
    det = cv2.QRCodeDetector()
    found: dict[str, int] = {}
    for kf in crops:
        for scale in (1, 2):
            img = kf.crop if scale == 1 else cv2.resize(kf.crop, None, fx=2, fy=2,
                                                         interpolation=cv2.INTER_CUBIC)
            try:
                text, _, _ = det.detectAndDecode(img)
            except cv2.error:
                text = ""
            if text:
                found.setdefault(text, kf.frame)
                break
    return found


# --- Muse Spark comparison ---------------------------------------------------------

SPARK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["same_unit", "visible_differences", "condition_shipped",
                 "condition_returned", "new_damage_on_return", "explanation"],
    "properties": {
        "same_unit": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "visible_differences": {"type": "array", "items": {"type": "string"}},
        "condition_shipped": {"type": "string"},
        "condition_returned": {"type": "string"},
        "new_damage_on_return": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
}

SPARK_PROMPT = """You are checking an e-commerce return.
The first {na} image(s) are crops of the item as it was packed by the seller (SHIPPED).
The next {nb} image(s) are crops of the item that came back in the return package (RETURNED).

Decide only from what is visible:
- same_unit: "yes" if nothing visible suggests a different unit, "no" if you see a clear
  difference (different model, colour, labels, markings, wear), "uncertain" if the images
  don't show enough to tell. Identical-looking products are common, so say "uncertain"
  rather than guess when there is no distinguishing detail.
- visible_differences: short concrete phrases, empty if none.
- condition_shipped / condition_returned: one short phrase each.
- new_damage_on_return: true only if the RETURNED item shows damage, tape, scuffs or
  marks not present on the SHIPPED item.
- explanation: at most two sentences.
Ignore differences in lighting, angle, blur and hands."""


def _data_url(img: np.ndarray, size: int = 512) -> str:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class SparkComparator:
    def __init__(self, s: Settings, cache_dir: Path, api_key: str | None = None):
        from openai import OpenAI

        key = api_key or os.environ.get("MODEL_API_KEY")
        if not key:
            raise RuntimeError("MODEL_API_KEY is not set")
        self.client = OpenAI(base_url=BASE_URL, api_key=key)
        self.s = s
        self.cache_dir = cache_dir
        self.estimated_spend = 0.0

    def compare(self, shipped: list[Keyframe], returned: list[Keyframe], cache_key: str) -> dict:
        path = self.cache_dir / f"spark_{cache_key}.json"
        if path.exists():
            return json.loads(path.read_text())
        content = [{"type": "text", "text": SPARK_PROMPT.format(na=len(shipped), nb=len(returned))}]
        for kf in shipped + returned:
            content.append({"type": "image_url", "image_url": {"url": _data_url(kf.crop)}})
        kwargs = dict(
            model=SPARK_MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "return_check", "strict": True,
                                             "schema": SPARK_SCHEMA}},
        )
        if self.s.spark_reasoning_effort:
            kwargs["reasoning_effort"] = self.s.spark_reasoning_effort
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # unsupported reasoning_effort -> retry without it
            if "reasoning" in str(e).lower() and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort")
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        result = json.loads(resp.choices[0].message.content)
        if resp.usage:
            self.estimated_spend += (resp.usage.prompt_tokens * SPARK_PRICE_IN
                                     + resp.usage.completion_tokens * SPARK_PRICE_OUT)
        path.write_text(json.dumps(result, indent=2))
        return result
