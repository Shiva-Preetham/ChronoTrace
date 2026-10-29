"""Video reading helpers. Read sequentially; seeking in phone MP4s is unreliable.

Phone videos are often stored landscape with a rotation flag. SAM reports boxes in the
displayed (rotated) frame, so every frame we hand out is rotated the same way. We turn
OpenCV's own auto-rotation off and apply the flag ourselves, because whether OpenCV
auto-rotates (and which size it reports) differs between versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    n_frames: int
    width: int          # displayed width, after applying the rotation flag
    height: int
    rotation: int = 0   # degrees clockwise from the stored frame

    @property
    def seconds(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def _open(path: Path) -> tuple[cv2.VideoCapture, int]:
    """Open with auto-rotation off; return the capture and the rotation flag in degrees."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open video: {path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rot = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
    return cap, rot if rot in ROTATIONS else 0


def rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    return cv2.rotate(frame, ROTATIONS[degrees]) if degrees in ROTATIONS else frame


def info(path: Path) -> VideoInfo:
    cap, rot = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if rot in (90, 270):
        w, h = h, w
    return VideoInfo(fps, n, w, h, rot)


def frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Every frame, in display orientation."""
    cap, rot = _open(path)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, rotate(frame, rot)
            i += 1
    finally:
        cap.release()


def match_aspect(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Last resort when the rotation flag couldn't be read: if the frame is landscape and
    SAM's reported size is portrait (or the reverse), turn the frame 90 degrees clockwise,
    the usual phone portrait case."""
    fh, fw = frame.shape[:2]
    if (fw > fh) != (w > h) and fw != fh and w != h:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


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
