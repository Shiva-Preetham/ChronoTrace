"""Real-API readiness, tested offline: retries, error handling, caching rules, budget,
Spark model fallback and phone rotation flags."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import openai
import pytest

try:                      # openai 3.x ships on httpx2; 1.x on httpx
    import httpx2 as httpx
except ImportError:
    import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chronotrace import api, video  # noqa: E402
from chronotrace.__main__ import check_report  # noqa: E402
from chronotrace.config import load_dotenv, spark_model  # noqa: E402
from chronotrace.identity import _crop  # noqa: E402
from chronotrace.sam import CachedSam, MetaSam  # noqa: E402
from chronotrace.tracks import Det  # noqa: E402

WORK = ROOT / "tests" / "_work" / "api"


def _err(cls, status: int):
    req = httpx.Request("POST", "https://api.meta.ai/v1/responses")
    return cls(f"status {status}", response=httpx.Response(status, request=req), body=None)


def _clip(name: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / name
    if not p.exists():
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        for _ in range(10):
            w.write(np.zeros((48, 64, 3), np.uint8))
        w.release()
    return p


# --- retries and error messages -------------------------------------------------------

def test_retries_429_with_exponential_backoff():
    calls, waits = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise _err(openai.RateLimitError, 429)
        return "ok"

    assert api.with_retries(fn, "x", sleep=waits.append, log=lambda *_: None) == "ok"
    assert len(calls) == 3 and waits == [2.0, 4.0]


def test_gives_up_after_five_tries():
    calls = []

    def fn():
        calls.append(1)
        raise _err(openai.InternalServerError, 503)

    with pytest.raises(api.ApiError):
        api.with_retries(fn, "x", sleep=lambda _: None, log=lambda *_: None)
    assert len(calls) == 5


def test_network_error_is_retried():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise openai.APIConnectionError(request=httpx.Request("GET", "https://x"))
        return 1

    assert api.with_retries(fn, "x", sleep=lambda _: None, log=lambda *_: None) == 1


@pytest.mark.parametrize("cls,status,words", [
    (openai.NotFoundError, 404, "chronotrace check"),
    (openai.BadRequestError, 400, "noun phrase"),
])
def test_client_errors_not_retried_and_explained(cls, status, words):
    calls = []

    def fn():
        calls.append(1)
        raise _err(cls, status)

    with pytest.raises(api.ApiError, match=words):
        api.with_retries(fn, "SAM", sleep=lambda _: None, log=lambda *_: None)
    assert len(calls) == 1


# --- caching rules ------------------------------------------------------------------

class CountingSam:
    def __init__(self, fail: bool = False):
        self.calls: list[str] = []
        self.fail = fail

    def segment(self, video_path: Path, prompt: str) -> list[str]:
        self.calls.append(prompt)
        if self.fail:
            raise api.ApiError("boom")
        return ["<0f>0<|box;x1=1;y1=1;x2=5;y2=5;w=64;h=48|>"]


def test_failed_call_writes_no_cache(tmp_path):
    clip = _clip("tiny.mp4")
    sam = CachedSam(CountingSam(fail=True), tmp_path, budget=api.Budget())
    with pytest.raises(api.ApiError):
        sam.track(clip, "box", 10)
    assert not list(tmp_path.glob("*.json"))
    assert sam.estimated_spend == 0


def test_refresh_single_prompt(tmp_path):
    clip = _clip("tiny.mp4")
    backend = CountingSam()
    CachedSam(backend, tmp_path, budget=api.Budget()).track(clip, "box", 10)
    CachedSam(backend, tmp_path, budget=api.Budget()).track(clip, "packing tape", 10)
    sam = CachedSam(backend, tmp_path, refresh_prompts={"Packing Tape"}, budget=api.Budget())
    sam.track(clip, "box", 10)
    sam.track(clip, "packing tape", 10)
    assert backend.calls == ["box", "packing tape", "packing tape"]


def test_budget_guard(tmp_path):
    clip = _clip("tiny.mp4")
    backend = CountingSam()
    budget = api.Budget(limit=0.015)        # minimum billing is 50 frames = $0.01 per call
    sam = CachedSam(backend, tmp_path, budget=budget)
    sam.track(clip, "box", 10)
    with pytest.raises(api.BudgetExceeded):
        sam.track(clip, "tape", 10)
    assert backend.calls == ["box"]
    sam.track(clip, "box", 10)              # cached: no spend, no error
    assert backend.calls == ["box"]


class _FakeStream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, *a):
        return False


def test_stream_error_event_is_not_cached(tmp_path):
    clip = _clip("tiny.mp4")
    sam = MetaSam(tmp_path, api_key="test-key", log=lambda *_: None)
    events = [SimpleNamespace(type="response.output_text.delta", delta="<0f>partial"),
              SimpleNamespace(type="response.failed")]
    sam.client = SimpleNamespace(
        files=SimpleNamespace(create=lambda **_: SimpleNamespace(id="file-1")),
        responses=SimpleNamespace(stream=lambda **_: _FakeStream(events)))
    cached = CachedSam(sam, tmp_path, budget=api.Budget())
    with pytest.raises(api.ApiError, match="response.failed"):
        cached.track(clip, "box", 10)
    assert not list(tmp_path.glob("*.json"))


# --- model selection and .env ----------------------------------------------------------

def test_pick_spark_model():
    assert api.pick_spark_model(["muse-spark-1.3", "muse-spark-1.4"], "muse-spark-1.3") == "muse-spark-1.3"
    assert api.pick_spark_model(["muse-spark-1.2", "muse-spark-1.10", "sam-3.1"],
                                "muse-spark-1.3") == "muse-spark-1.10"
    assert api.pick_spark_model(["sam-3.1"], "muse-spark-1.3") is None


def test_check_report_flags_missing():
    lines = "\n".join(check_report(["muse-spark-1.2"], "muse-spark-1.3"))
    assert "sam-3.1" in lines and "MISSING" in lines
    assert "CHRONOTRACE_SPARK_MODEL=muse-spark-1.2" in lines


def test_spark_model_env_and_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("CHRONOTRACE_SPARK_MODEL", raising=False)
    assert spark_model() == "muse-spark-1.3"
    env = tmp_path / ".env"
    env.write_text("# comment\nCHRONOTRACE_SPARK_MODEL='muse-spark-2.0'\n")
    load_dotenv(env)
    assert spark_model() == "muse-spark-2.0"
    monkeypatch.delenv("CHRONOTRACE_SPARK_MODEL")


# --- rotation ---------------------------------------------------------------------------

def _rotated_clip() -> Path:
    """Landscape 640x480 frames stored with a 90 degree display flag (phone portrait).
    A red marker sits at the stored top-left; displayed, it is at the top-right."""
    ffmpeg = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    WORK.mkdir(parents=True, exist_ok=True)
    src, out = WORK / "landscape.mp4", WORK / "portrait_flag.mp4"
    if out.exists():
        return out
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for _ in range(15):
        f = np.full((480, 640, 3), 90, np.uint8)
        cv2.rectangle(f, (0, 0), (100, 50), (0, 0, 255), -1)
        w.write(f)
    w.release()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-display_rotation", "-90", "-i", str(src),
                    "-c", "copy", str(out)], check=True)
    return out


def test_rotated_clip_reads_in_display_orientation():
    clip = _rotated_clip()
    vi = video.info(clip)
    assert (vi.width, vi.height, vi.rotation) == (480, 640, 90)
    _, frame = next(video.frames(clip))
    assert frame.shape[:2] == (640, 480)
    red = np.argwhere(frame[:, :, 2] > 200)
    assert red[:, 1].min() >= 420 and red[:, 0].max() <= 110   # top-right after rotation


def test_crop_uses_rotated_frame():
    clip = _rotated_clip()
    _, frame = next(video.frames(clip))
    # SAM reports boxes in the displayed 480x640 frame.
    d = Det(frame=0, obj_id=0, x1=430, y1=0, x2=480, y2=100, w=480, h=640)
    crop = _crop(frame, d, pad=0)
    assert crop[:, :, 2].mean() > 200


def test_crop_falls_back_when_flag_missing():
    landscape = np.full((480, 640, 3), 90, np.uint8)
    cv2.rectangle(landscape, (0, 0), (100, 50), (0, 0, 255), -1)
    d = Det(frame=0, obj_id=0, x1=430, y1=0, x2=480, y2=100, w=480, h=640)
    assert _crop(landscape, d, pad=0)[:, :, 2].mean() > 200
