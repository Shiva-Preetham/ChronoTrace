"""Evidence packet: JSON report plus a side-by-side card built from real frames.

The card is composed with OpenCV from the actual video frames. A forensics tool shouldn't
use a generative image model to draw its own evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .events import VideoFindings
from .identity import Keyframe
from .verdict import Verdict

GREEN, RED, AMBER, GREY = (60, 160, 60), (50, 50, 210), (30, 150, 220), (110, 110, 110)
STATUS_COLOR = {
    "GENUINE_RETURN": GREEN, "RETURNED_WITH_DAMAGE": AMBER, "SWAP_DETECTED": RED,
    "SWAP_LIKELY": RED, "PACKING_NOT_PROVEN": AMBER, "NEEDS_REVIEW": GREY,
}


def _fit(img: np.ndarray | None, w: int, h: int) -> np.ndarray:
    canvas = np.full((h, w, 3), 235, np.uint8)
    if img is None or img.size == 0:
        cv2.putText(canvas, "no clear frame", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)
        return canvas
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))))
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
    return canvas


def _text(img, s, xy, scale=0.6, color=(30, 30, 30), thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _wrap(s: str, n: int = 95) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def card(verdict: Verdict, order_id: str, pack: VideoFindings, ret: VideoFindings,
         kf_a: list[Keyframe], kf_b: list[Keyframe], qr_a: dict, qr_b: dict, out: Path) -> Path:
    W, panel_w, panel_h = 960, 460, 300
    lines = []
    for r in verdict.reasons:
        lines += _wrap("- " + r)
    for c in pack.checks + ret.checks:
        mark = "PASS" if c.passed else ("FAIL" if c.passed is False else " ?? ")
        lines += _wrap(f"[{mark}] {pack.role if c in pack.checks else ret.role}: {c.detail}")
    for n in verdict.review_notes:
        lines += _wrap("REVIEW: " + n)
    H = 110 + panel_h + 50 + 24 * len(lines) + 30
    img = np.full((H, W, 3), 250, np.uint8)

    color = STATUS_COLOR[verdict.status]
    cv2.rectangle(img, (0, 0), (W, 70), color, -1)
    _text(img, verdict.headline, (20, 45), 0.9, (255, 255, 255), 2)
    _text(img, f"Order {order_id}   |   ChronoTrace evidence", (20, 95), 0.55, GREY)

    y0 = 110
    for i, (label, kfs, qr, f) in enumerate([
        ("SHIPPED (packing video)", kf_a, qr_a, pack),
        ("RETURNED (return video)", kf_b, qr_b, ret),
    ]):
        x0 = 20 + i * (panel_w + 20)
        best = kfs[0] if kfs else None
        img[y0:y0 + panel_h, x0:x0 + panel_w] = _fit(best.crop if best else None, panel_w, panel_h)
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (200, 200, 200), 1)
        sub = f"{label}  frame {best.frame} ({f.t(best.frame)})" if best else label
        _text(img, sub, (x0, y0 + panel_h + 22), 0.5)
        _text(img, "QR: " + (", ".join(qr) if qr else "none read"), (x0, y0 + panel_h + 42), 0.5,
              GREY)

    y = y0 + panel_h + 75
    for ln in lines:
        c = RED if "[FAIL]" in ln else ((20, 110, 20) if "[PASS]" in ln else (40, 40, 40))
        _text(img, ln, (20, y), 0.5, c)
        y += 24
    cv2.imwrite(str(out), img)
    return out


def report(verdict: Verdict, order_id: str, pack: VideoFindings, ret: VideoFindings,
           qr_a: dict, qr_b: dict, spark: dict | None, file_hashes: dict, spend: float,
           kf_a: list[Keyframe], kf_b: list[Keyframe], videos: dict | None = None) -> dict:
    def findings(f: VideoFindings) -> dict:
        return {
            "events": {k: (None if v is None else {"frame": v, "time": f.t(v),
                                                   "seconds": round(v / f.fps, 3)})
                       for k, v in f.events.items()},
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail,
                        "time": f.t(c.frame) if c.frame is not None else None,
                        "seconds": round(c.frame / f.fps, 3) if c.frame is not None else None}
                       for c in f.checks],
            "warnings": f.warnings,
            "warning_marks": [{"frame": fr, "seconds": round(fr / f.fps, 3), "label": label}
                              for fr, label in f.warning_frames],
        }
    return {
        "order_id": order_id,
        "status": verdict.status,
        "headline": verdict.headline,
        "reasons": verdict.reasons,
        "needs_review": verdict.review_notes,
        "packing": findings(pack),
        "return": findings(ret),
        "qr": {"shipped": qr_a, "returned": qr_b},
        "visual_comparison": spark,
        "keyframes": {"shipped": [k.frame for k in kf_a], "returned": [k.frame for k in kf_b]},
        "sha256": file_hashes,
        "videos": videos or {},
        "estimated_api_cost_usd": round(spend, 4),
    }


def save_crops(kf_a: list[Keyframe], kf_b: list[Keyframe], out_dir: Path) -> dict[str, str]:
    """Best real crop from each video, as JPEG files next to the report."""
    saved = {}
    for name, kfs in (("shipped", kf_a), ("returned", kf_b)):
        if kfs:
            path = out_dir / f"{name}_crop.jpg"
            cv2.imwrite(str(path), kfs[0].crop)
            saved[name] = path.name
    return saved


def save(rep: dict, out_dir: Path) -> Path:
    path = out_dir / "report.json"
    path.write_text(json.dumps(rep, indent=2))
    return path
