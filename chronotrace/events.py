"""Turn SAM tracks into events: item inserted, box sealed, seal on arrival, item removed.

Key design choice: we never rely on SAM object ids surviving occlusion. Meta's docs warn
ids aren't reliable across an object leaving the frame or being fully hidden, and an item
going into a box is exactly that. So "inserted" means: the item's last observations are
inside the box region and it is never seen outside again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .tracks import Det, Track, overlap_fraction


@dataclass
class Check:
    name: str
    passed: bool | None          # None = could not be determined
    detail: str
    frame: int | None = None


@dataclass
class VideoFindings:
    role: str                    # "packing" or "return"
    fps: float
    events: dict[str, int | None] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # (frame, short label) for each warning, so a timeline can place it.
    warning_frames: list[tuple[int, str]] = field(default_factory=list)

    def check(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    def t(self, frame: int | None) -> str:
        return "n/a" if frame is None else f"{frame / self.fps:05.2f}s"


def container_at(container: Track, frame: int, window: int) -> Det | None:
    """Container box on this frame, or the nearest one within `window` frames.
    Hands often hide the box for a moment; it doesn't move in that time."""
    if container.at(frame):
        return max(container.at(frame), key=lambda d: d.area)
    for off in range(1, window + 1):
        for f in (frame - off, frame + off):
            if container.at(f):
                return max(container.at(f), key=lambda d: d.area)
    return None


def runs(frames: list[int], max_gap: int) -> list[tuple[int, int]]:
    """Group sorted frame indices into (start, end) runs, tolerating small gaps."""
    out: list[tuple[int, int]] = []
    for f in sorted(frames):
        if out and f - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def _on_box_frames(track: Track, container: Track, frac: float, window: int) -> list[int]:
    hits = []
    for f in track.frames():
        box = container_at(container, f, window)
        if box and any(overlap_fraction(d, box) >= frac for d in track.at(f)):
            hits.append(f)
    return hits


def analyze_packing(container: Track, item: Track, seal: Track, fps: float,
                    s: Settings) -> VideoFindings:
    out = VideoFindings("packing", fps)
    min_frames = max(2, round(s.min_event_seconds * fps))
    window = round(0.5 * fps)
    gap = round(0.25 * fps)
    box = container.main_object()

    if not box:
        out.checks.append(Check("box_detected", False,
                                f"No '{container.prompt}' found. Try a simpler prompt."))
        return out
    out.checks.append(Check("box_detected", True, f"'{container.prompt}' tracked"))

    # --- insertion -------------------------------------------------------------
    obs: list[tuple[int, bool]] = []
    for f in item.frames():
        b = container_at(box, f, window)
        for d in item.at(f):
            obs.append((f, b is not None and overlap_fraction(d, b) >= s.inside_fraction))

    insertion = None
    if not obs:
        out.checks.append(Check("item_inserted", False,
                                f"'{item.prompt}' never detected in the packing video"))
    else:
        last_outside = max((f for f, inside in obs if not inside), default=-1)
        inside_after = sorted({f for f, inside in obs if inside and f > last_outside})
        last_obs = max(f for f, _ in obs)
        # Disappeared while inside the box, and the box is still visible afterwards.
        vanished_inside = bool(inside_after) and inside_after[-1] == last_obs and any(
            f > last_obs + min_frames for f in box.frames())
        if inside_after and (len(inside_after) >= min_frames or vanished_inside):
            insertion = inside_after[0]
            how = "then hidden inside the box" if vanished_inside else "and stayed inside"
            out.checks.append(Check("item_inserted", True,
                                    f"Item entered the box at {out.t(insertion)} {how}",
                                    insertion))
        else:
            out.checks.append(Check(
                "item_inserted", False,
                "Item was seen, but never placed and kept inside the box"
                + (f" (last seen outside at {out.t(last_outside)})" if last_outside >= 0 else ""),
                last_outside if last_outside >= 0 else None))
    out.events["inserted"] = insertion

    # --- sealing ---------------------------------------------------------------
    seal_hits = _on_box_frames(seal, box, s.seal_on_box_fraction, window)
    seal_runs = [r for r in runs(seal_hits, gap) if r[1] - r[0] + 1 >= min_frames]
    sealed = seal_runs[0][0] if seal_runs else None
    out.events["sealed"] = sealed
    out.checks.append(Check("box_sealed", sealed is not None,
                            f"Seal applied on the box at {out.t(sealed)}" if sealed is not None
                            else f"No '{seal.prompt}' seen on the box", sealed))

    # --- order of events ---------------------------------------------------------
    if insertion is not None and sealed is not None:
        ok = insertion < sealed
        out.checks.append(Check(
            "inserted_before_sealed", ok,
            f"Inserted {out.t(insertion)}, sealed {out.t(sealed)}" if ok
            else f"Seal at {out.t(sealed)} came before insertion at {out.t(insertion)}",
            sealed))
        b_after = [(f, ins) for f, ins in obs if f > sealed and not ins]
        out.checks.append(Check(
            "not_removed_after_seal", not b_after,
            "Item not seen outside the box after sealing" if not b_after
            else f"Item seen outside the box at {out.t(b_after[0][0])}, after sealing",
            b_after[0][0] if b_after else None))
    else:
        out.checks.append(Check("inserted_before_sealed", None if sealed is None else False,
                                "Can't confirm the item was packed before sealing"))
    return out


def analyze_return(container: Track, item: Track, seal: Track, fps: float,
                   s: Settings) -> VideoFindings:
    out = VideoFindings("return", fps)
    window = round(0.5 * fps)
    box = container.main_object()
    if not box:
        out.checks.append(Check("box_detected", False, f"No '{container.prompt}' found"))
        return out
    out.checks.append(Check("box_detected", True, f"'{container.prompt}' tracked"))

    # Seal present on arrival: seen on the box early in the clip.
    arrival_end = round(s.arrival_window_seconds * fps)
    seal_hits = _on_box_frames(seal, box, s.seal_on_box_fraction, window)
    early = [f for f in seal_hits if f <= arrival_end]
    need = max(2, round(0.3 * fps))
    on_arrival = len(early) >= need
    out.events["seal_seen"] = early[0] if early else None
    out.checks.append(Check(
        "seal_on_arrival", on_arrival,
        f"Seal visible on the box at {out.t(early[0])}" if on_arrival
        else f"No seal seen on the box in the first {s.arrival_window_seconds:.1f}s",
        early[0] if early else None))

    # Item comes out of the box: first seen inside, later seen outside.
    obs = []
    for f in item.frames():
        b = container_at(box, f, window)
        for d in item.at(f):
            obs.append((f, b is not None and overlap_fraction(d, b) >= s.inside_fraction))
    first_inside = min((f for f, ins in obs if ins), default=None)
    removed = None
    if first_inside is not None:
        removed = min((f for f, ins in obs if not ins and f > first_inside), default=None)
    out.events["first_inside"] = first_inside
    out.events["removed"] = removed
    if not obs:
        out.checks.append(Check("item_from_box", False, "Item never detected in the return video"))
    elif first_inside is None:
        out.checks.append(Check("item_from_box", False,
                                "Item only seen outside the box; it may not have come from this package",
                                min(f for f, _ in obs)))
    else:
        out.checks.append(Check(
            "item_from_box", True,
            f"Item seen inside the box at {out.t(first_inside)}"
            + (f", taken out at {out.t(removed)}" if removed is not None else ""),
            first_inside))
    return out
