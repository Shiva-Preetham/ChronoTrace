# Submission draft

Draft answers for a typical hackathon form. Everything here matches the code in this repo.
Before submitting, update the "Results" lines if a real benchmark has been run.

## Project name
ChronoTrace

## Track and theme
- Track: **Indie developer** (solo student builder). Judged on creativity, technical
  ambition, and whether it works.
- Theme: **Build for Your World**: return-swap fraud is a problem Indian marketplace sellers
  face first-hand, and the project brings Meta's models to it.

## One-liner
ChronoTrace watches a seller's packing video and the buyer's return-unboxing video and
reports, with timestamped evidence, whether the item that came back is the item that left.

## Problem
Return-swap fraud costs Indian marketplace sellers money: the buyer returns a different,
older or empty item, and without proof of what was shipped the dispute goes against the
seller. Order-management tools like EasyEcom and Unicommerce already record packing videos
and tag them to order ids, but a person still has to watch the footage to build a claim, so
most of it goes unused.

## Solution
Upload the packing video and the return video (plus the order id). ChronoTrace tracks the
box, item and tape in both, finds when the item went into the box and when it was sealed,
checks the seal was intact on arrival and the item came out of that box, reads the QR code on
the item in both videos, and compares the sharpest crops for visible differences. A fixed
rule set turns that evidence into one of six verdicts (genuine return, returned with damage,
swap detected, swap likely, packing not proven, needs review). The output is a
`report.json` with every check and timestamp, SHA-256 hashes of both files, and an evidence
card built only from real frames. A web page shows both videos with clickable event
timelines.

It checks the seller's evidence first: if the packing video doesn't show the item going in
before the box is sealed, the verdict is "packing not proven", whatever the return shows.

## Which Meta models, and how
- **SAM 3.1** (`sam-3.1`, Responses API, streaming): one request per concept ("cardboard
  box", the item noun phrase, "packing tape") on each video. The per-frame boxes drive all
  event detection and keyframe selection. Because SAM object ids aren't reliable across
  occlusion, insertion is inferred as "last seen inside the box and never seen outside
  again". Responses are cached by file hash and prompt for reproducibility and cost.
- **Muse Spark** (`muse-spark-1.3`, Chat Completions with a strict JSON schema): compares
  the sharpest shipped and returned crops and lists visible differences, condition, and new
  damage. Its output is evidence that the rules read; it never sets the verdict directly and
  is never shown as a confidence score.
- **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`): transcribes the packing video's
  audio so the order id the packer says aloud can be compared with the claimed order. A
  mismatch adds a review note only.

## What's novel
- Turns existing, unwatched packing footage into a structured claim packet automatically.
- Neutral by design: the seller's own video is checked first, and the tool says so when the
  seller can't prove packing.
- Event logic built around a documented SAM limitation (ids lost on occlusion) instead of
  assuming perfect tracking.
- Deterministic verdicts: every result traces to named checks with timestamps; models only
  provide evidence.
- No generative imagery anywhere in the evidence.

## Against the indie judging criteria
- **Creativity:** uses footage sellers already record but never watch, and checks the
  seller's evidence as strictly as the buyer's.
- **Technical ambition:** three Meta models (SAM 3.1 video tracking, Muse Spark
  vision with strict JSON output, Muse Voice Transcribe), event inference that survives
  SAM's id loss on occlusion, phone rotation handling, QR decoding, hashing, and a web demo.
- **Whether it works:** 42 offline tests; the three demo scenarios load from saved reports
  with no network, so they don't depend on the API being up. Live-API and real-footage
  results: see Results.

## How it was built
Built solo during the hackathon (all commits from Oct 29, 2026 onward), with help from an AI
coding assistant (Claude Code); commits it helped with are marked as co-authored.

## Challenges
- Inferring "the item went into the box" when the item disappears from view at exactly that
  moment, and SAM's ids aren't stable across it.
- Phone videos stored landscape with a rotation flag, while SAM reports boxes in the
  displayed orientation. Frames are rotated explicitly so crops line up.
- Keeping the demo reliable: preloaded scenarios load from saved reports with no API call,
  every API response is cached, and a hard spend limit stops a run before it overspends.
- Being honest about edit detection: it only catches hard cuts and only ever asks for review.

## Results
- Synthetic scenarios (generated clips, simulated tracking): 3 of 3 correct. This checks the
  rules and plumbing, not real-world accuracy.
- Real footage: not yet measured.

## What's next
- Benchmark and tune on real packing and return clips.
- Integrate with EasyEcom / Unicommerce to fetch packing videos by order id.
- Use SAM masks for tighter "inside the box" geometry on cluttered tables.
- Tamper-evident storage of hashes at packing time.
