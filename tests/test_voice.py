"""Spoken order id: normaliser on sample transcripts, the check, and the transcriber's
caching, all offline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chronotrace import api  # noqa: E402
from chronotrace.events import Check, VideoFindings  # noqa: E402
from chronotrace.verdict import decide  # noqa: E402
from chronotrace.voice import (MetaTranscriber, Transcript, _HttpError,  # noqa: E402
                               find_order_ids, order_check)

WORK = ROOT / "tests" / "_work" / "voice"


@pytest.mark.parametrize("text,want", [
    ("Packing order F K eight eight three one four now.", ["FK88314"]),
    ("order FK-88314", ["FK-88314"]),
    ("This is order fk 88314 going out", ["FK88314"]),
    ("Order F.K. 8 8 3 1 4.", ["FK88314"]),
    ("ef kay double eight three one four", ["FK88314"]),
    ("order F K dash eight eight three one four", ["FK-88314"]),
    ("F K triple seven oh one", ["FK77701"]),
    ("order A M one two three four five", ["AM12345"]),
    ("Okay so this one is going out today", []),
    ("I have 2 boxes and 3 tapes", []),
])
def test_find_order_ids(text, want):
    assert find_order_ids(text) == want


def test_order_check_match_mismatch_and_none():
    turns = [{"turnId": 0, "startMs": 500, "endMs": 2500,
              "transcript": "Order F K eight eight three one four."}]
    ok = order_check(Transcript("", turns), "FK-88314", 30.0)
    assert ok.passed is True and ok.frame == 15
    bad = order_check(Transcript("", turns), "FK-99999", 30.0)
    assert bad.passed is False and "FK-99999" in bad.detail
    assert order_check(Transcript("hello there", []), "FK-1", 30.0).passed is None
    assert order_check(None, "FK-1", 30.0).passed is None


def test_spoken_mismatch_is_review_note_only():
    pack = VideoFindings("packing", 30.0)
    pack.checks = [Check("item_inserted", True, ""), Check("inserted_before_sealed", True, ""),
                   Check("not_removed_after_seal", True, ""),
                   Check("order_id_spoken", False, "Packer said order FK11111, but this claim is for FK-2")]
    ret = VideoFindings("return", 30.0)
    ret.checks = [Check("seal_on_arrival", True, ""), Check("item_from_box", True, "")]
    v = decide(pack, ret, {"A": 1}, {"A": 2}, None)
    assert v.status == "GENUINE_RETURN"
    assert any("FK11111" in n for n in v.review_notes)


def _clip(name: str, audio: bool) -> Path:
    ffmpeg = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / name
    if not out.exists():
        args = [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x48:d=1"]
        if audio:
            args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest", "-c:a", "aac"]
        subprocess.run(args + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def test_transcriber_caches_and_handles_no_audio(tmp_path):
    t = MetaTranscriber(tmp_path, api_key="test", budget=api.Budget(), log=lambda *_: None)
    calls = []
    said = "order F K eight eight three one four"
    reply = {"sessionId": "s", "transcript": said, "audioDurationMs": 1000,
             "turns": [{"turnId": 0, "startMs": 0, "endMs": 900, "transcript": said}]}

    def fake_post(wav):
        calls.append(wav)
        assert wav.read_bytes()[:4] == b"RIFF"
        return reply

    t._post = fake_post
    clip = _clip("with_audio.mp4", audio=True)
    first = t.transcribe(clip, tmp_path / "audio")
    second = t.transcribe(clip, tmp_path / "audio")
    assert len(calls) == 1 and first.text == second.text == said

    assert t.transcribe(_clip("no_audio.mp4", audio=False), tmp_path / "audio") is None
    assert len(calls) == 1


def test_transcriber_failure_not_cached(tmp_path):
    t = MetaTranscriber(tmp_path, api_key="test", budget=api.Budget(), log=lambda *_: None)

    def fail(wav):
        raise _HttpError(400, "HTTP 400: unsupported WAV")

    t._post = fail
    with pytest.raises(api.ApiError, match="400"):
        t.transcribe(_clip("with_audio.mp4", audio=True), tmp_path / "audio")
    assert not list(tmp_path.glob("*_transcript.json"))


def test_transcriber_retries_429(tmp_path, monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    t = MetaTranscriber(tmp_path, api_key="test", budget=api.Budget(), log=lambda *_: None)
    calls = []

    def flaky(wav):
        calls.append(1)
        if len(calls) < 2:
            raise _HttpError(429, "HTTP 429", retryable=True)
        return {"transcript": "", "turns": []}

    t._post = flaky
    assert t.transcribe(_clip("with_audio.mp4", audio=True), tmp_path / "audio").text == ""
    assert len(calls) == 2
