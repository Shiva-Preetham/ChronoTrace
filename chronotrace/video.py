"""Video reading helpers. Read sequentially; seeking in phone MP4s is unreliable."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def seconds(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(fps, n, w, h)


def frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


def grab(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Collect only the frames we need in one sequential pass."""
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    last = max(wanted)
    for i, frame in frames(path):
        if i in wanted:
            out[i] = frame
        if i >= last:
            break
    return out
