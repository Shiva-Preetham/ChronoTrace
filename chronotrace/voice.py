"""Spoken order id: the packer says the order id aloud; we check it against --order.

Request shape from dev.meta.ai/docs/api-reference/voice/transcribe (read Oct 29, 2026):
  POST /v1/asr/transcribe, multipart/form-data with
    request: JSON {"model", "audioEncoding": "WAV", "mode"}   (application/json part)
    audio:   RIFF/WAVE, mono, 16-bit PCM, 16 or 24 kHz, at most 10 minutes, body <= 32 MB
  JSON response: {"sessionId", "transcript", "audioDurationMs",
                  "turns": [{"turnId", "startMs", "endMs", "transcript"}]}
  Turn-level timestamps only. `turns` is empty in PUSH_TO_TALK mode, so we use ENDPOINTING.

A mismatch is only ever a review note. It never changes the verdict.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import api
from .config import BASE_URL, TRANSCRIBE_MODEL, TRANSCRIBE_PRICE_PER_HOUR
from .events import Check
from .sam import file_sha256

MAX_AUDIO_SECONDS = 600

DIGITS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
          "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
# Spelled-out letter names that aren't also common English words.
LETTER_NAMES = {"ay": "A", "bee": "B", "dee": "D", "ef": "F", "eff": "F", "gee": "G", "aitch": "H",
                "jay": "J", "kay": "K", "el": "L", "em": "M", "en": "N", "pee": "P", "cue": "Q",
                "ar": "R", "ess": "S", "vee": "V", "ex": "X", "zed": "Z", "zee": "Z"}
REPEAT = {"double": 2, "triple": 3}
DASH = {"dash", "hyphen", "-"}
# Two-letter words that are English, not order-id prefixes.
STOP2 = {"am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it", "me", "my",
         "no", "of", "ok", "on", "or", "so", "to", "up", "us", "we"}
ORDER_RE = re.compile(r"[A-Z]{2}-?\d{4,6}")


def _token_chars(tok: str) -> str | None:
    """What a spoken token contributes to an id, or None if it's an ordinary word."""
    low = tok.lower()
    if low in DIGITS:
        return DIGITS[low]
    if low in LETTER_NAMES:
        return LETTER_NAMES[low]
    if low in DASH:
        return "-"
    if re.fullmatch(r"[A-Za-z]", tok):
        return tok.upper()
    if re.fullmatch(r"[A-Za-z]{2}", tok) and low not in STOP2:
        return tok.upper()
    if re.fullmatch(r"[A-Za-z]{2}-?\d+|\d+", tok):
        return tok.upper()
    return None


def normalise(text: str) -> str:
    """Turn spoken forms into compact runs: 'order F K eight eight three one four' ->
    'FK88314'. Ordinary words become spaces, so runs never join across them."""
    tokens = re.findall(r"[A-Za-z]+(?:-?\d+)?|\d+|-", text.replace(".", ""))
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.lower() in REPEAT and i + 1 < len(tokens):
            nxt = _token_chars(tokens[i + 1])
            if nxt and len(nxt) == 1:
                out.append(nxt * REPEAT[tok.lower()])
                i += 2
                continue
        out.append(_token_chars(tok) or " ")
        i += 1
    return re.sub(r" +", " ", "".join(out)).strip()


def canonical(order_id: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", order_id.upper())


def find_order_ids(text: str) -> list[str]:
    return [m.group(0) for m in ORDER_RE.finditer(normalise(text))]


@dataclass
class Transcript:
    text: str
    turns: list[dict]


def extract_wav(video: Path, out: Path) -> Path | None:
    """Mono 16 kHz 16-bit WAV of the first 10 minutes, or None if the video has no audio."""
    import imageio_ffmpeg

    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-map_metadata", "-1",
                        "-t", str(MAX_AUDIO_SECONDS), str(out)], capture_output=True, text=True)
    if r.returncode != 0 or not out.exists() or out.stat().st_size <= 44:
        return None
    return out


class _HttpError(Exception):
    def __init__(self, status_code: int | None, message: str, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def _http():
    try:                      # openai 3.x ships on httpx2; 1.x on httpx
        import httpx2 as httpx
    except ImportError:
        import httpx
    return httpx


class MetaTranscriber:
    """Muse Voice Transcribe over HTTP, cached by audio file hash."""

    def __init__(self, cache_dir: Path, api_key: str | None = None,
                 budget: api.Budget | None = None, log=print):
        import os

        self.key = api_key or os.environ.get("MODEL_API_KEY")
        if not self.key:
            raise api.ApiError("MODEL_API_KEY is not set.")
        self.cache_dir = cache_dir
        self.budget = budget or api.SESSION
        self.estimated_spend = 0.0
        self.log = log

    def _post(self, wav: Path) -> dict:
        httpx = _http()
        req = {"model": TRANSCRIBE_MODEL, "audioEncoding": "WAV", "mode": "ENDPOINTING"}
        try:
            with open(wav, "rb") as f:
                r = httpx.post(f"{BASE_URL}/asr/transcribe",
                               headers={"Authorization": f"Bearer {self.key}", "Accept": "application/json"},
                               files={"request": (None, json.dumps(req), "application/json"),
                                      "audio": (wav.name, f, "audio/wav")},
                               timeout=300)
        except httpx.HTTPError as e:
            raise _HttpError(None, f"network error: {e}", retryable=True) from e
        if r.status_code >= 400:
            raise _HttpError(r.status_code, f"HTTP {r.status_code}: {r.text[:300]}",
                             retryable=r.status_code == 429 or r.status_code >= 500)
        return r.json()

    def transcribe(self, video: Path, work_dir: Path) -> Transcript | None:
        digest = file_sha256(video)[:16]
        path = self.cache_dir / f"{digest}_transcript.json"
        if path.exists():
            data = json.loads(path.read_text())
            return None if data is None else Transcript(data.get("transcript", ""), data.get("turns", []))
        work_dir.mkdir(parents=True, exist_ok=True)
        wav = extract_wav(video, work_dir / f"{digest}.wav")
        if wav is None:
            path.write_text("null")          # no audio track: a fact about the file, safe to cache
            return None
        seconds = (wav.stat().st_size - 44) / (16000 * 2)
        cost = seconds / 3600 * TRANSCRIBE_PRICE_PER_HOUR
        self.budget.reserve(cost, "Voice transcription")
        data = api.with_retries(lambda: self._post(wav), f"Transcribe {video.name}", log=self.log)
        self.budget.add(cost)
        self.estimated_spend += cost
        path.write_text(json.dumps(data, indent=2))
        return Transcript(data.get("transcript", ""), data.get("turns", []))


def order_check(transcript: Transcript | None, expected: str, fps: float) -> Check:
    """Check `order_id_spoken`. passed: True match, False a different id was said,
    None no audio or no id heard."""
    if transcript is None:
        return Check("order_id_spoken", None, "No audio track in the packing video")
    want = canonical(expected)
    heard: list[tuple[str, int | None]] = []
    for t in transcript.turns:
        frame = round(t["startMs"] / 1000 * fps) if t.get("startMs") is not None else None
        heard += [(i, frame) for i in find_order_ids(t.get("transcript", ""))]
    if not heard:
        heard = [(i, None) for i in find_order_ids(transcript.text)]
    if not heard:
        return Check("order_id_spoken", None, "No order id heard in the packing video")
    for oid, frame in heard:
        if canonical(oid) == want:
            return Check("order_id_spoken", True, f"Packer said order {oid}", frame)
    oid, frame = heard[0]
    return Check("order_id_spoken", False, f"Packer said order {oid}, but this claim is for {expected}",
                 frame)
