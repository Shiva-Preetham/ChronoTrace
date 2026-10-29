"""Packing video + return video -> verdict, report.json, evidence card."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import evidence, integrity, video
from .config import Settings
from .events import analyze_packing, analyze_return
from .identity import decode_qr, select_keyframes
from .sam import CachedSam, file_sha256
from .verdict import decide


@dataclass
class Result:
    report: dict
    report_path: Path
    card_path: Path


def run(packing: Path, returned: Path, order_id: str, s: Settings, sam: CachedSam,
        spark=None, log=print) -> Result:
    s.out_dir.mkdir(parents=True, exist_ok=True)
    vi_a, vi_b = video.info(packing), video.info(returned)
    log(f"Packing: {packing.name} {vi_a.width}x{vi_a.height} {vi_a.fps:.0f}fps {vi_a.seconds:.1f}s")
    log(f"Return:  {returned.name} {vi_b.width}x{vi_b.height} {vi_b.fps:.0f}fps {vi_b.seconds:.1f}s")

    tracks = {}
    for role, path, vi in (("a", packing, vi_a), ("b", returned, vi_b)):
        for kind, prompt in (("box", s.container_prompt), ("item", s.item_prompt),
                             ("seal", s.seal_prompt)):
            t = sam.track(path, prompt, vi.n_frames)
            log(f"  SAM [{role}] '{prompt}': {len(t.frames())} frames with detections")
            tracks[role, kind] = t

    pack = analyze_packing(tracks["a", "box"], tracks["a", "item"], tracks["a", "seal"], vi_a.fps, s)
    ret = analyze_return(tracks["b", "box"], tracks["b", "item"], tracks["b", "seal"], vi_b.fps, s)
    integrity.add_integrity_warnings(pack, packing, tracks["a", "box"], s)
    integrity.add_integrity_warnings(ret, returned, tracks["b", "box"], s)

    # Keyframes: packing -> before the item went in; return -> after it was seen in the box.
    ins = pack.events.get("inserted")
    range_a = (0, ins) if ins is not None else None
    first_in = ret.events.get("first_inside")
    range_b = (first_in, vi_b.n_frames) if first_in is not None else None
    kf_a = select_keyframes(packing, tracks["a", "item"], s, range_a)
    kf_b = select_keyframes(returned, tracks["b", "item"], s, range_b)
    qr_a, qr_b = decode_qr(kf_a), decode_qr(kf_b)
    log(f"  QR shipped: {qr_a or 'none'} | returned: {qr_b or 'none'}")

    spark_result = None
    if spark is not None and kf_a and kf_b:
        key = f"{file_sha256(packing)[:8]}_{file_sha256(returned)[:8]}"
        n = s.keyframes_per_video
        spark_result = spark.compare(kf_a[:n], kf_b[:n], key)
        log(f"  Muse Spark: same_unit={spark_result.get('same_unit')}")

    v = decide(pack, ret, qr_a, qr_b, spark_result)
    spend = sam.estimated_spend + (getattr(spark, "estimated_spend", 0.0) or 0.0)
    videos = {role: {"file": p.name, "fps": vi.fps, "n_frames": vi.n_frames,
                     "seconds": round(vi.seconds, 3), "width": vi.width, "height": vi.height,
                     "rotation": vi.rotation}
              for role, p, vi in (("packing", packing, vi_a), ("return", returned, vi_b))}
    rep = evidence.report(v, order_id, pack, ret, qr_a, qr_b, spark_result,
                          integrity.hashes(packing, returned), spend, kf_a, kf_b, videos)
    rep["crops"] = evidence.save_crops(kf_a, kf_b, s.out_dir)
    report_path = evidence.save(rep, s.out_dir)
    card_path = evidence.card(v, order_id, pack, ret, kf_a, kf_b, qr_a, qr_b,
                              s.out_dir / "evidence_card.png")
    log(f"\nVERDICT: {v.status} - {v.headline}")
    for r in v.reasons:
        log(f"  - {r}")
    for n in v.review_notes:
        log(f"  ! {n}")
    log(f"Estimated API cost this run: ${spend:.3f}")
    return Result(rep, report_path, card_path)
