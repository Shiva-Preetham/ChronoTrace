"""Synthetic top-down packing/return clips with ground-truth geometry, plus a mock SAM
backend that emits the documented wire format. Lets the whole pipeline run offline."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import qrcode

W, H, FPS = 640, 480, 30
BOX = (220, 190, 430, 410)               # x1, y1, x2, y2
ITEM_W, ITEM_H = 150, 110
TAPE = (220, 290, 430, 312)
BG, BOX_C, TAPE_C, ITEM_C = (120, 120, 120), (60, 110, 170), (110, 190, 220), (40, 40, 150)


def qr_img(code: str, size: int = 90) -> np.ndarray:
    img = qrcode.make(code, box_size=6, border=2).convert("RGB")
    return cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (size, size),
                      interpolation=cv2.INTER_NEAREST)


def draw_item(frame, x, y, code, marked, blur):
    x2, y2 = x + ITEM_W, y + ITEM_H
    cv2.rectangle(frame, (x, y), (x2, y2), ITEM_C, -1)
    q = qr_img(code)
    frame[y + 10:y + 100, x + 10:x + 100] = q
    if marked:  # visible condition difference on the swapped unit
        cv2.rectangle(frame, (x + 110, y + 5), (x + 140, y + 105), (0, 0, 230), -1)
    if blur:
        roi = frame[max(0, y - 10):min(H, y2 + 10), max(0, x - 10):min(W, x2 + 10)]
        frame[max(0, y - 10):min(H, y2 + 10), max(0, x - 10):min(W, x2 + 10)] = cv2.blur(roi, (15, 15))
    return (x, y, x2, y2)


def lerp(a, b, t):
    return int(a + (b - a) * t)


def make_packing(path: Path, code: str, scenario: str, n=180) -> dict:
    """scenario: 'normal' or 'never_packed'."""
    gt = {"box": {}, "item": {}, "seal": {}}
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    start, inside = (40, 60), (BOX[0] + 30, BOX[1] + 55)
    for f in range(n):
        fr = np.full((H, W, 3), BG, np.uint8)
        cv2.rectangle(fr, BOX[:2], BOX[2:], BOX_C, -1)
        gt["box"][f] = [BOX]
        item_box = None
        if scenario == "normal":
            if f < 60:
                item_box = draw_item(fr, *start, code, False, False)
            elif f < 100:
                t = (f - 60) / 40
                item_box = draw_item(fr, lerp(start[0], inside[0], t), lerp(start[1], inside[1], t),
                                     code, False, True)
            elif f < 130:
                item_box = draw_item(fr, *inside, code, False, False)
            elif f < 150:
                cv2.rectangle(fr, BOX[:2], BOX[2:], BOX_C, -1)   # flaps closed: item hidden
        else:  # seller never puts the item in
            item_box = draw_item(fr, *start, code, False, False)
        if item_box:
            gt["item"][f] = [item_box]
        if f >= 150:
            cv2.rectangle(fr, TAPE[:2], TAPE[2:], TAPE_C, -1)
            gt["seal"][f] = [TAPE]
        writer.write(fr)
    writer.release()
    return gt


def make_return(path: Path, code: str, marked: bool, n=180) -> dict:
    gt = {"box": {}, "item": {}, "seal": {}}
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    inside, out = (BOX[0] + 30, BOX[1] + 55), (470, 300)
    for f in range(n):
        fr = np.full((H, W, 3), BG, np.uint8)
        cv2.rectangle(fr, BOX[:2], BOX[2:], BOX_C, -1)
        gt["box"][f] = [BOX]
        item_box = None
        if f < 40:
            cv2.rectangle(fr, TAPE[:2], TAPE[2:], TAPE_C, -1)
            gt["seal"][f] = [TAPE]
        elif f < 60:
            item_box = draw_item(fr, *inside, code, marked, False)
        elif f < 100:
            t = (f - 60) / 40
            item_box = draw_item(fr, lerp(inside[0], out[0] - 0, t), lerp(inside[1], out[1], t),
                                 code, marked, True)
        else:
            item_box = draw_item(fr, min(out[0], W - ITEM_W - 1), out[1], code, marked, False)
        if item_box:
            gt["item"][f] = [item_box]
        writer.write(fr)
    writer.release()
    return gt


class MockSam:
    """Returns SAM-format output_text lines from ground truth, with jitter and dropouts
    so the event logic has to cope with imperfect tracking."""

    billable = False

    def __init__(self, truth: dict[str, dict], prompts: dict[str, str], seed: int = 7,
                 drop: float = 0.05, jitter: int = 3):
        self.truth, self.prompts = truth, prompts   # prompts: prompt text -> "box"/"item"/"seal"
        self.rng = random.Random(seed)
        self.drop, self.jitter = drop, jitter

    def segment(self, video: Path, prompt: str) -> list[str]:
        kind = self.prompts[prompt]
        boxes = self.truth[video.name][kind]
        lines = []
        for f in sorted(boxes, key=int):
            if self.rng.random() < self.drop:
                continue
            recs = []
            for oid, (x1, y1, x2, y2) in enumerate(boxes[f]):
                j = lambda v, lim: max(0, min(lim - 1, v + self.rng.randint(-self.jitter, self.jitter)))
                c = round(self.rng.uniform(0.8, 0.99), 4)
                recs.append(f"{oid}<|box;x1={j(x1, W)};y1={j(y1, H)};x2={j(x2, W)};y2={j(y2, H)};"
                            f"w={W};h={H};c={c}|><|mask;x=0;y=0;c={c};data=2,2,!0~|>")
            lines.append(f"<{f}f>" + ",".join(recs))
        return lines


class FakeSpark:
    estimated_spend = 0.0

    def __init__(self, result: dict):
        self.result = result

    def compare(self, shipped, returned, cache_key):
        return self.result


def build_all(root: Path) -> dict:
    """Three demo scenarios. Returns {video_name: ground_truth}."""
    root.mkdir(parents=True, exist_ok=True)
    truth = {
        "genuine_pack.mp4": make_packing(root / "genuine_pack.mp4", "CT-A7F3K", "normal"),
        "genuine_return.mp4": make_return(root / "genuine_return.mp4", "CT-A7F3K", False),
        "swap_pack.mp4": make_packing(root / "swap_pack.mp4", "CT-A7F3K", "normal"),
        "swap_return.mp4": make_return(root / "swap_return.mp4", "CT-B2M9Q", True),
        "fault_pack.mp4": make_packing(root / "fault_pack.mp4", "CT-C5R1T", "never_packed"),
        "fault_return.mp4": make_return(root / "fault_return.mp4", "CT-C5R1T", False),
    }
    (root / "truth.json").write_text(json.dumps(truth))
    return truth


class Scenario:
    def __init__(self, name: str, title: str, order_id: str, expected: str, description: str):
        self.name, self.title, self.order_id = name, title, order_id
        self.expected, self.description = expected, description


SCENARIOS = [
    Scenario("genuine", "Genuine return", "FK-88314", "GENUINE_RETURN",
             "The same unit (QR CT-A7F3K) is packed, sealed, and comes back."),
    Scenario("swap", "Swap fraud", "FK-88315", "SWAP_DETECTED",
             "A different unit with QR CT-B2M9Q and a red mark comes back."),
    Scenario("fault", "Seller at fault", "FK-88316", "PACKING_NOT_PROVEN",
             "The seller shows the item but tapes the box without putting it in."),
]
