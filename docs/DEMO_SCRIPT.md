# Demo video script (2 minutes)

Screen: the web demo (`python -m chronotrace serve`, http://localhost:8000) at 1280 px.
If real clips have been packaged with `prepare-demo --real`, use those; otherwise use the
built-in synthetic scenarios and say they are generated clips.

## 0:00–0:15 The problem

**Say:** "Indian marketplace sellers lose money to return-swap fraud. The buyer sends back a
different, older or empty item, and without proof of what was shipped, the claim goes against
the seller. EasyEcom and Unicommerce already record packing videos and tag them to order
ids, but someone still has to sit and watch them. ChronoTrace watches them."

**Show:** the page header, then the scenario buttons.

## 0:15–1:30 Three scenarios

### 0:15–0:40 Genuine return
**Click:** Genuine return.
**Say:** "Packing video on the left, return-unboxing video on the right. SAM 3.1 tracks the
box, the item and the tape. The rules find when the item went into the box and when it was
sealed." Click the *Item in box* marker on the left timeline; the video jumps there.
"On return, the seal was intact on arrival, and the QR code on the item matches the one
that shipped. Verdict: genuine return."

### 0:40–1:00 Swap fraud
**Click:** Swap fraud.
**Say:** "Same packing video, but a different unit came back. The QR on the returned item
reads a different code, and you can see the extra mark in the crop. Verdict: swap
detected, with the two codes as the reason." Point at the two crops in the identity panel.

### 1:00–1:30 Seller at fault (the credibility moment)
**Click:** Seller at fault.
**Say:** "This is the case that matters most. Here the seller shows the item, but tapes the
box without putting it in. ChronoTrace checks the seller's own evidence first, so it says
*packing not proven*. It is not a tool that blames buyers. It checks both sides." Click
the red failed-check marker on the packing timeline, then scroll to the checks list
showing *Item put in the box: fail*.

## 1:30–1:50 How the work is split

**Show:** the README architecture diagram, or the checks and evidence sections.
**Say:** "SAM 3.1 gives per-frame tracks, one concept per request, all cached by file hash
so the demo is reproducible. Fixed rules turn tracks into events. OpenCV reads the QR codes,
which is the hard identity proof. Muse Spark compares the sharpest crops for damage and
differences, and Muse Voice Transcribe checks the spoken order id. Models only supply
evidence; the verdict comes from deterministic rules, and every verdict points at a named
check. File hashes and the evidence card come from the real frames."

## 1:50–2:00 Honest results and limits

**Say:** "So far it's been tested on generated clips: all three scenarios come out right. It
hasn't been benchmarked on real footage yet, so I'm not quoting an accuracy number. Edit
detection only flags hard cuts for review, never as fraud."
(If a real `out/bench/summary.md` exists by recording time, replace this with its numbers.)
