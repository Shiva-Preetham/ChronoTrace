"""Per-frame detections from SAM, plus small geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Det:
    """One object on one frame. Box corners are in source-video pixels."""
    frame: int
    obj_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    w: int          # source frame width
    h: int          # source frame height
    conf: float | None = None

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def scaled(self, W: int, H: int) -> tuple[int, int, int, int]:
        """Box in a frame of size W x H (handles SAM vs OpenCV size differences)."""
        sx, sy = W / self.w, H / self.h
        return int(self.x1 * sx), int(self.y1 * sy), int(self.x2 * sx), int(self.y2 * sy)


def overlap_fraction(inner: Det, outer: Det) -> float:
    """Share of `inner`'s area that lies inside `outer`."""
    ix1, iy1 = max(inner.x1, outer.x1), max(inner.y1, outer.y1)
    ix2, iy2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / inner.area if inner.area else 0.0


class Track:
    """All detections for one concept prompt across a video, keyed by frame."""

    def __init__(self, prompt: str, dets: list[Det]):
        self.prompt = prompt
        self.by_frame: dict[int, list[Det]] = {}
        for d in dets:
            self.by_frame.setdefault(d.frame, []).append(d)

    def frames(self) -> list[int]:
        return sorted(self.by_frame)

    def at(self, frame: int) -> list[Det]:
        return self.by_frame.get(frame, [])

    def all(self) -> list[Det]:
        return [d for f in self.frames() for d in self.by_frame[f]]

    def __bool__(self) -> bool:
        return bool(self.by_frame)

    def main_object(self) -> Track:
        """The object id seen on the most frames (e.g. the box, ignoring a second
        box in the background). Returned as its own Track."""
        counts: dict[int, int] = {}
        for d in self.all():
            counts[d.obj_id] = counts.get(d.obj_id, 0) + 1
        if not counts:
            return Track(self.prompt, [])
        best = max(counts, key=counts.get)
        return Track(self.prompt, [d for d in self.all() if d.obj_id == best])

    def to_json(self) -> dict:
        return {
            "prompt": self.prompt,
            "frames": {
                str(f): [d.__dict__ for d in ds] for f, ds in sorted(self.by_frame.items())
            },
        }
