"""Web demo backend.

  GET  /api/cases          preloaded scenarios (no API call)
  GET  /api/cases/{id}     report.json plus asset URLs
  POST /api/verify         two uploaded videos -> pipeline -> report (needs MODEL_API_KEY)
  /demo/..., /uploads/...  videos, evidence cards, crops
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import demo

STATIC = Path(__file__).parent / "static"
VIDEO_EXT = {".mp4", ".mov"}


def _assets(base_url: str, d: Path, rep: dict) -> dict:
    crops = rep.get("crops", {})
    return {
        "packing_video": f"{base_url}/packing.mp4",
        "return_video": f"{base_url}/return.mp4",
        "evidence_card": f"{base_url}/evidence_card.png" if (d / "evidence_card.png").exists() else None,
        "report": f"{base_url}/report.json",
        "crops": {k: f"{base_url}/{v}" for k, v in crops.items()},
    }


def _case_payload(d: Path, base_url: str) -> dict:
    rep = json.loads((d / "report.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    return {"meta": meta, "report": rep, "assets": _assets(base_url, d, rep)}


def create_app(demo_root: Path = demo.DEMO_DIR, work_root: Path = Path("out/web"),
               cache_dir: Path = Path("cache")) -> FastAPI:
    app = FastAPI(title="ChronoTrace")
    work_root.mkdir(parents=True, exist_ok=True)
    if not demo.list_cases(demo_root):
        demo.build_synthetic(demo_root)        # fresh checkout without assets: build offline

    @app.get("/api/cases")
    def cases() -> list[dict]:
        return demo.list_cases(demo_root)

    @app.get("/api/cases/{case_id}")
    def case(case_id: str) -> dict:
        d = demo.case_dir(case_id, demo_root) if re.fullmatch(r"[\w-]+", case_id) else None
        if d is None:
            raise HTTPException(404, f"No case called '{case_id}'")
        return _case_payload(d, f"/demo/{d.parent.name}/{d.name}")

    @app.get("/api/status")
    def status() -> dict:
        return {"api_key_set": bool(os.environ.get("MODEL_API_KEY"))}

    @app.post("/api/verify")
    def verify(packing: UploadFile = File(...), returned: UploadFile = File(...),
               order_id: str = Form("UNKNOWN"), item_prompt: str = Form("")) -> dict:
        from .. import api
        from ..config import Settings
        from ..identity import SparkComparator
        from ..pipeline import run
        from ..sam import CachedSam, MetaSam

        if not os.environ.get("MODEL_API_KEY"):
            raise HTTPException(503, "MODEL_API_KEY is not set on this server, so new videos "
                                     "can't be analysed. The preloaded scenarios still work.")
        for up in (packing, returned):
            if Path(up.filename or "").suffix.lower() not in VIDEO_EXT:
                raise HTTPException(400, f"'{up.filename}' is not an MP4 or MOV file.")
        order_id = re.sub(r"[^\w-]", "", order_id)[:40] or "UNKNOWN"
        job = uuid.uuid4().hex[:12]
        job_dir = work_root / job
        inp = job_dir / "input"
        inp.mkdir(parents=True)
        paths = []
        for role, up in (("packing", packing), ("return", returned)):
            p = inp / f"{role}{Path(up.filename).suffix.lower()}"
            with open(p, "wb") as f:
                shutil.copyfileobj(up.file, f)
            paths.append(p)

        s = Settings(out_dir=job_dir / "analysis", cache_dir=cache_dir)
        if item_prompt.strip():
            s.item_prompt = item_prompt.strip()[:60]
        try:
            sam = CachedSam(MetaSam(cache_dir), cache_dir)
            spark = SparkComparator(s, cache_dir) if s.use_spark else None
            run(paths[0], paths[1], order_id, s, sam, spark)
        except (api.ApiError, api.BudgetExceeded) as e:
            raise HTTPException(502, str(e)) from e
        meta = {"id": job, "title": "Your upload", "description": f"Order {order_id}",
                "source": "upload", "expected": None}
        demo.package_case(job_dir, meta, paths[0], paths[1], s.out_dir)
        return _case_payload(job_dir, f"/uploads/{job}")

    app.mount("/demo", StaticFiles(directory=demo_root), name="demo")
    app.mount("/uploads", StaticFiles(directory=work_root), name="uploads")
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
