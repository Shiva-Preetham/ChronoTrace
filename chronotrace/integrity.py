"""Edit warnings and file hashes.

Cut detection is the most error-prone check, so it only ever produces a
"needs review" warning, never a fraud verdict. It catches hard cuts, not deepfakes.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from . import video
from .config import Settings
from .events import VideoFindings
from .sam import file_sha256
from .tracks import Track


def find_cuts(path: Path, s: Settings) -> list[int]:
    """Frames where the whole picture changes abruptly versus the previous frame."""
    cuts, prev = [], None
    for i, frame in video.frames(path):
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev is not None and cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL) < s.cut_hist_threshold:
            cuts.append(i)
        prev = hist
    return cuts


def box_gaps(box: Track, fps: float, min_seconds: float = 0.5) -> list[tuple[int, int]]:
    """Stretches where the box vanished mid-clip. On a fixed stand the box shouldn't
    disappear; a gap can mean a hand covered it, or the recording was cut."""
    fs = box.frames()
    gaps = []
    for a, b in zip(fs, fs[1:]):
        if b - a > min_seconds * fps:
            gaps.append((a, b))
    return gaps


def add_integrity_warnings(findings: VideoFindings, path: Path, box: Track, s: Settings) -> None:
    for c in find_cuts(path, s):
        findings.warnings.append(
            f"Possible edit at {findings.t(c)}: the whole picture changed in one frame. Needs review.")
    for a, b in box_gaps(box.main_object(), findings.fps):
        findings.warnings.append(
            f"Box not visible from {findings.t(a)} to {findings.t(b)} "
            "(hand in the way, or a cut). Needs review.")


def hashes(*paths: Path) -> dict[str, str]:
    return {p.name: file_sha256(p) for p in paths}
