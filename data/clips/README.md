# Real clips

Put real packing and return videos here and list them in `manifest.csv`. Video files are
gitignored; only this guide and the manifest are tracked.

## Manifest columns

| Column | Meaning |
|---|---|
| `case_id` | Short unique name, e.g. `swap_02` |
| `packing_file`, `return_file` | File names in this folder (MP4 or MOV, H.264/H.265) |
| `order_id` | The order id shown or spoken in the video, e.g. `FK-10002` |
| `item_prompt` | Short noun phrase SAM should look for, e.g. `earbuds case`. One concept only. |
| `expected_status` | The true answer: `GENUINE_RETURN`, `SWAP_DETECTED`, `PACKING_NOT_PROVEN`, ... |
| `notes` | Anything useful: lighting, what was swapped |

Run them with `python -m chronotrace bench data/clips/manifest.csv`.

## Filming guide

- Phone fixed top-down (stand or a stack of books), good light, 1080p 30 fps, 10–15 s.
- Packing: hold the item's QR sticker to the camera for about 1 s, place it in the box,
  close the flaps, tape it.
- Return: show the taped box for about 1 s, open it, take the item out, hold it up for about 1 s.
- Keep the box in frame the whole time.
- Optional: say the order id aloud near the start of the packing video ("order F K one
  zero zero zero two").

For each of the three scenarios, film at least two cases:

1. Genuine return: the same unit goes out and comes back.
2. Swap: a second unit with a different QR sticker (and ideally a visible mark, like a
   strip of coloured tape) comes back.
3. Seller at fault: the item is shown, but the box is taped without putting it in.
