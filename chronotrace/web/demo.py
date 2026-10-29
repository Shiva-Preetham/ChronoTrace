"""Demo cases the web page can show with no API call: report.json, evidence card, crops,
and browser-playable copies of both videos.

Synthetic scenarios are always available (built offline). Real-clip cases are packaged
from an earlier `bench` run and replace the synthetic ones when present.

  python -m chronotrace prepare-demo                       # rebuild the synthetic cases
  python -m chronotrace prepare-demo --real data/clips/manifest.csv out/bench
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

DEMO_DIR = Path(__file__).parent / "demo"


def to_h264(src: Path, dst: Path) -> None:
    """Re-encode for browsers. OpenCV's mp4v output often won't play in a <video> tag.
    The analysed original is not changed; its SHA-256 stays in the report."""
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "26",
                    "-c:a", "aac", "-movflags", "+faststart", str(dst)], check=True)


def package_case(dest: Path, meta: dict, packing: Path, returned: Path, report_dir: Path) -> Path:
    """Copy one analysed case into `dest` in the layout the web app serves."""
    dest.mkdir(parents=True, exist_ok=True)
    rep = json.loads((report_dir / "report.json").read_text())
    for name in ["report.json", "evidence_card.png", *rep.get("crops", {}).values()]:
        if (report_dir / name).exists():
            shutil.copyfile(report_dir / name, dest / name)
    to_h264(packing, dest / "packing.mp4")
    to_h264(returned, dest / "return.mp4")
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    return dest


def build_synthetic(root: Path = DEMO_DIR, log=print) -> list[Path]:
    from ..config import Settings
    from ..pipeline import run
    from ..sam import CachedSam
    from ..synthetic import SCENARIOS, MockSam, build_all

    out = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        truth = build_all(work / "clips")
        for sc in SCENARIOS:
            s = Settings(cache_dir=work / "cache", out_dir=work / "out" / sc.name, use_spark=False)
            prompts = {s.container_prompt: "box", s.item_prompt: "item", s.seal_prompt: "seal"}
            sam = CachedSam(MockSam(truth, prompts), s.cache_dir, refresh=True)
            pack, ret = work / "clips" / f"{sc.name}_pack.mp4", work / "clips" / f"{sc.name}_return.mp4"
            run(pack, ret, sc.order_id, s, sam, None, log=lambda *_: None)
            meta = {"id": sc.name, "title": sc.title, "description": sc.description,
                    "source": "synthetic", "expected": sc.expected}
            out.append(package_case(root / "synthetic" / sc.name, meta, pack, ret, s.out_dir))
            log(f"  built {sc.name}")
    return out


TITLES = {"GENUINE_RETURN": "Genuine return", "SWAP_DETECTED": "Swap fraud",
          "SWAP_LIKELY": "Swap fraud", "PACKING_NOT_PROVEN": "Seller at fault"}


def package_real(manifest: Path, bench_out: Path, root: Path = DEMO_DIR, log=print) -> list[Path]:
    """Package real cases that already have a report from `bench` (no API calls)."""
    from ..bench import read_manifest

    out = []
    for c in read_manifest(manifest):
        report_dir = bench_out / c.case_id
        if not (report_dir / "report.json").exists():
            log(f"  skip {c.case_id}: no report in {report_dir} (run bench first)")
            continue
        meta = {"id": c.case_id, "title": TITLES.get(c.expected, c.case_id),
                "description": c.notes or f"Real clip, order {c.order_id}",
                "source": "real", "expected": c.expected}
        out.append(package_case(root / "real" / c.case_id, meta, c.packing, c.returned, report_dir))
        log(f"  packaged {c.case_id}")
    return out


def list_cases(root: Path = DEMO_DIR) -> list[dict]:
    """Real cases if any were packaged, otherwise the synthetic ones."""
    for source in ("real", "synthetic"):
        metas = [json.loads(p.read_text()) for p in sorted((root / source).glob("*/meta.json"))]
        if metas:
            order = {"GENUINE_RETURN": 0, "SWAP_DETECTED": 1, "SWAP_LIKELY": 1, "PACKING_NOT_PROVEN": 2}
            return sorted(metas, key=lambda m: (order.get(m.get("expected"), 3), m["id"]))
    return []


def case_dir(case_id: str, root: Path = DEMO_DIR) -> Path | None:
    for source in ("real", "synthetic"):
        d = root / source / case_id
        if (d / "meta.json").exists():
            return d
    return None
