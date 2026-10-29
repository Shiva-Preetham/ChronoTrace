# Progress

## Needs you

- Set `MODEL_API_KEY` (env var or an untracked `.env` file in the repo root), then run
  `python -m chronotrace check` and paste the output here. No key was available during
  the build, so **nothing has been run against the real API yet**.
- Film real clips per `data/clips/README.md` (at least 2 per scenario) and fill in
  `data/clips/manifest.csv`. Phase B tuning and real benchmark numbers wait on this.

## Log

- **Phase A (real-API readiness), Oct 29.** Done: `check` lists models, flags missing
  ones, and suggests a Spark fallback; Spark model id is configurable via
  `CHRONOTRACE_SPARK_MODEL` and falls back automatically to the newest `muse-spark-*` on a
  404. Phone rotation flags handled (`video.py` applies the flag itself; crops fall back to
  matching SAM's aspect). Retries with exponential backoff (5 tries) for 429/5xx/network,
  plain messages for 400/401/404, stream error events raise, failed calls never write a
  cache entry, $10 per-process budget guard, `--refresh-prompt` for a single prompt.
  `tools/sam_test.py` now shares the pipeline's cache and code. Verified offline: 20 tests
  pass (rotation tested on a real ffmpeg-flagged clip). Unverified: everything against the
  real API (no key); `check` not yet run.
- **Phase B (tuning), Oct 29.** Skipped: no real clips and no API key. Created `data/clips/` with a template `manifest.csv` and filming guide. No thresholds changed.
- **Phase C (bench), Oct 29.** Done: `python -m chronotrace bench [manifest]` writes `results.csv` and `summary.md` (cases run, correct, confusion table, reason per miss, estimated cost) to `out/bench/`; stops cleanly at the budget. `--synthetic` runs the 3 generated scenarios with mock SAM and no Spark into `out/bench_synthetic/` (3 of 3, $0; tests plumbing only, not accuracy). Synthetic generator moved into the package. Verified: 24 tests pass. Unverified: real-clip bench (no clips, no key); README results stay "not yet measured".
- **Phase D (web demo), Oct 29.** Done: `python -m chronotrace serve` (FastAPI + one static page) with scenario buttons, verdict banner, two players with clickable event/warning/failed-check timelines, identity panel (crops, QR, Spark or a plain 'not run' note), checks, review notes, hashes and downloads. Preloaded synthetic cases ship in `chronotrace/web/demo/` (H.264 copies, ~1 MB) and load with no key; `prepare-demo --real` packages real cases from a bench run (no API calls). `POST /api/verify` returns 503 with a clear message when no key is set. Verified: 27 tests pass; checked in a browser (marker click seeks, no console errors) and with Playwright via Edge at 1280 px and 390 px (no horizontal scroll), screenshots in `docs/screenshots/`. Unverified: the upload path end to end (needs the key).
