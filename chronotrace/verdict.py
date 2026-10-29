"""Deterministic verdict. Models supply evidence; these rules make the call, so every
verdict can be traced to a named check."""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import VideoFindings

HEADLINES = {
    "GENUINE_RETURN": "Returned item matches the shipped item",
    "RETURNED_WITH_DAMAGE": "Same item came back, but with new damage",
    "SWAP_DETECTED": "Returned item does not match the shipped item",
    "SWAP_LIKELY": "Returned item likely differs from the shipped item",
    "PACKING_NOT_PROVEN": "Packing video does not prove the item was packed",
    "NEEDS_REVIEW": "Not enough evidence for an automatic verdict",
}


@dataclass
class Verdict:
    status: str
    reasons: list[str] = field(default_factory=list)
    review_notes: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        return HEADLINES[self.status]


def decide(pack: VideoFindings, ret: VideoFindings, qr_a: dict[str, int], qr_b: dict[str, int],
           spark: dict | None) -> Verdict:
    notes = pack.warnings + ret.warnings

    # 1. The seller's own evidence must hold up first. This is what makes the tool
    #    neutral: it flags the seller as readily as the buyer.
    for name in ("item_inserted", "inserted_before_sealed", "not_removed_after_seal"):
        c = pack.check(name)
        if c is not None and c.passed is False:
            return Verdict("PACKING_NOT_PROVEN", [f"Packing: {c.detail}"], notes)
    if (c := pack.check("box_sealed")) and c.passed is False:
        notes.append(f"Packing: {c.detail}")
    # A spoken order id that doesn't match is worth a look, but never decides the verdict.
    if (c := pack.check("order_id_spoken")) and c.passed is False:
        notes.append(f"Packing: {c.detail}")

    for name in ("seal_on_arrival", "item_from_box"):
        c = ret.check(name)
        if c is not None and c.passed is False:
            notes.append(f"Return: {c.detail}")

    spark_says = (spark or {}).get("same_unit")
    damage = bool((spark or {}).get("new_damage_on_return"))
    diffs = (spark or {}).get("visible_differences") or []

    # 2. QR codes: deterministic identity when both sides have one.
    if qr_a and qr_b:
        common = set(qr_a) & set(qr_b)
        if not common:
            reasons = [f"QR on shipped item: {', '.join(qr_a)}; on returned item: {', '.join(qr_b)}"]
            if diffs:
                reasons.append("Also visible: " + "; ".join(diffs))
            return Verdict("SWAP_DETECTED", reasons, notes)
        code = sorted(common)[0]
        if damage:
            return Verdict("RETURNED_WITH_DAMAGE",
                           [f"QR {code} matches on both videos",
                            "New on return: " + ("; ".join(diffs) or spark.get("condition_returned", ""))],
                           notes)
        if spark_says == "no":
            notes.append("QR matches, but Muse Spark saw differences: " + "; ".join(diffs))
        return Verdict("GENUINE_RETURN", [f"QR {code} matches on both videos"], notes)

    if qr_a and not qr_b:
        reasons = [f"Shipped item carried QR {', '.join(qr_a)}; no QR found on the returned item"]
        if spark_says == "no":
            return Verdict("SWAP_LIKELY", reasons + ["Visible differences: " + "; ".join(diffs)], notes)
        return Verdict("NEEDS_REVIEW", reasons, notes)

    # 3. No usable QR: fall back to visual comparison, stated as likelihood only.
    if spark_says == "no":
        return Verdict("SWAP_LIKELY", ["Visible differences: " + "; ".join(diffs)], notes)
    if spark_says == "yes":
        if damage:
            return Verdict("RETURNED_WITH_DAMAGE",
                           ["Looks like the same unit (no QR to confirm)", "New on return: " + "; ".join(diffs)],
                           notes)
        return Verdict("NEEDS_REVIEW",
                       ["Looks like the same unit, but there's no QR to confirm identity"], notes)
    return Verdict("NEEDS_REVIEW", ["No QR on either video and no clear visual difference"], notes)
