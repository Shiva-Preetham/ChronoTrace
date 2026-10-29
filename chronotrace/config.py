"""All tunable settings in one place. Change prompts and thresholds here, not in code."""

from dataclasses import dataclass, field
from pathlib import Path

BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
SPARK_MODEL = "muse-spark-1.3"

# Published rates (dev.meta.ai/docs/pricing-rate-limits), used only for cost estimates.
SAM_PRICE_PER_FRAME = 0.20 / 1000
SAM_MIN_BILLED_FRAMES = 50
SPARK_PRICE_IN = 1.25 / 1_000_000
SPARK_PRICE_OUT = 4.25 / 1_000_000


@dataclass
class Settings:
    # SAM concept prompts: one concept per request (comma lists return zero matches).
    container_prompt: str = "cardboard box"
    item_prompt: str = "chocolate bar"
    seal_prompt: str = "packing tape"

    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    out_dir: Path = field(default_factory=lambda: Path("out"))

    # Geometry: an item counts as "in the box" when this share of its box area
    # lies inside the container box.
    inside_fraction: float = 0.6
    # A seal counts as "on the box" when this share of its area lies on the container.
    seal_on_box_fraction: float = 0.5
    # Events must persist this long (seconds) to count; filters one-frame flickers.
    min_event_seconds: float = 0.4
    # Return video: a seal seen within this many seconds of the start counts as
    # "present on arrival".
    arrival_window_seconds: float = 1.5

    # Keyframes: only frames where the item mask is at least this share of its
    # largest observed area (proxy for "not hidden behind a hand").
    keyframe_min_area_ratio: float = 0.7
    keyframes_per_video: int = 3
    keyframe_crop_size: int = 384

    # Edit check: HSV histogram correlation below this between consecutive frames
    # is a candidate hard cut. Reported as "needs review", never as fraud.
    cut_hist_threshold: float = 0.5

    # Muse Spark
    spark_reasoning_effort: str | None = "low"
    use_spark: bool = True
