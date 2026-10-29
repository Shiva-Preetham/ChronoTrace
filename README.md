# ChronoTrace

**Did the item that came back match the item that left?**

Indian marketplace sellers lose money to return-swap fraud: a buyer returns a different,
older or empty item, and without proof of what was shipped the claim goes the buyer's way.
Tools like EasyEcom and Unicommerce already record packing videos and link them to orders,
but a person still has to watch the footage. ChronoTrace watches it.

Give it the seller's **packing video** and the **return-unboxing video**. It returns a
verdict, the evidence behind it, and file hashes, in a format a seller can attach to a claim.

## How it works

| Step | What does it | Why |
|---|---|---|
| Track box, item and tape in both videos | **SAM 3.1** (Meta Model API), one request per concept | Pixel-accurate boxes per frame, zero-shot |
| Detect events: item inserted, box sealed, seal on arrival, item removed | Rules over the tracks (`events.py`) | Deterministic, every verdict traces to a named check |
| Pick the sharpest, unoccluded item frames | Laplacian variance on fixed-size crops | Motion blur makes labels unreadable |
| Identity: QR sticker on the item | OpenCV `QRCodeDetector` | Certain, no model guessing |
| Condition and visual differences | **Muse Spark 1.3**, structured JSON | Damage, marks, wear the QR can't show |
| Edit check | Histogram cut detection, box-visibility gaps | Flagged as *needs review*, never as fraud |
| Evidence | SHA-256 of both files, report JSON, card from real frames | No generated imagery in evidence |

Insertion is detected as "the item's last observations are inside the box and it's never
seen outside again", not by following SAM object ids, because Meta's docs say ids aren't
reliable across full occlusion, which is exactly what happens when an item goes into a box.

### Verdicts

| Status | Meaning |
|---|---|
| `GENUINE_RETURN` | QR matches, no new damage |
| `RETURNED_WITH_DAMAGE` | Same unit, new damage visible |
| `SWAP_DETECTED` | QR codes differ |
| `SWAP_LIKELY` | No QR to confirm, but visible differences |
| `PACKING_NOT_PROVEN` | The seller's own video doesn't show the item packed before sealing |
| `NEEDS_REVIEW` | Not enough evidence for an automatic call |

`PACKING_NOT_PROVEN` is deliberate: the tool checks the seller's evidence first, so it is
neutral rather than a buyer-blaming machine.

## Run it

```bash
pip install -r requirements.txt
export MODEL_API_KEY=...            # Windows: set MODEL_API_KEY=...

python -m chronotrace check         # API access + model ids
python -m chronotrace verify packing.mp4 return.mp4 --order FK-88314 --item "chocolate bar"
python -m chronotrace verify packing.mp4 return.mp4 --no-spark   # SAM + QR only
```

Output goes to `out/<order>/`: `report.json` and `evidence_card.png`. All SAM and Spark
responses are cached in `cache/`, so re-runs are free and reproducible. Use `--refresh`
to re-query SAM.

`tools/sam_test.py` is the first-contact script: runs SAM on one clip and writes an
overlay video so you can eyeball the tracking.

## Filming guide

- Phone fixed top-down (stand or books), good light, 1080p 30 fps, 10–15 s.
- Packing: hold the item's QR to the camera ~1 s, place it in the box, close, tape.
- Return: show the taped box ~1 s, open, take the item out, hold it up ~1 s.
- Keep the box in frame the whole time.

## Tests

```bash
python -m pytest -q
```

Generates synthetic top-down clips for the three demo scenarios and runs the full pipeline
with a mock SAM that emits the documented wire format (with jitter and dropped frames) and
a fake Spark. Under 35% dropped frames and 12 px jitter, all three scenarios still produce
the right verdict. Synthetic clips are clean; real video will be harder, so measured
accuracy on real clips is reported separately.

## Limits

- Edit detection catches hard cuts, not deepfakes or careful re-recording.
- Identity without a QR is a visual likelihood, not proof.
- The platform still decides the claim; this produces evidence, not a ruling.
