"""
Real-world degradation transforms used for the robustness evaluation.

This is the core of the project's actual contribution: measuring how
detection accuracy holds up (or doesn't) once an image has been through
the kind of processing it would realistically see in the wild — a saved
JPEG, a resize, a crop, a screenshot.

Each function takes a PIL Image and returns a degraded PIL Image, so they
can be composed and applied at eval time (and optionally as train-time
augmentation, if you want to test whether training on degraded data
improves robustness).
"""

import io
from dataclasses import dataclass
from typing import Callable

from PIL import Image


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode through JPEG at the given quality (1-95)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_roundtrip(img: Image.Image, scale: float) -> Image.Image:
    """Downscale then upscale back to original size (mimics a shared/re-uploaded image)."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def center_crop(img: Image.Image, keep_fraction: float) -> Image.Image:
    """Center-crop to keep_fraction of width/height, then resize back to original size."""
    w, h = img.size
    new_w, new_h = int(w * keep_fraction), int(h * keep_fraction)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


def screenshot_simulate(img: Image.Image) -> Image.Image:
    """
    Rough approximation of a phone screenshot: downscale to a typical
    screen-capture resolution, then JPEG-compress at a moderate quality.
    Not a substitute for real screenshot samples, but a cheap, consistent
    stand-in for the eval sweep.
    """
    w, h = img.size
    target_w = min(w, 1080)
    scale = target_w / w
    resized = img.resize((target_w, max(1, int(h * scale))), Image.BILINEAR)
    return jpeg_compress(resized, quality=75)


@dataclass
class DegradationLevel:
    name: str
    fn: Callable[[Image.Image], Image.Image]


def build_degradation_suite() -> list[DegradationLevel]:
    """
    The fixed set of degradation levels applied in the robustness eval.
    Includes a 'clean' no-op level as the baseline to compare against.
    """
    levels = [DegradationLevel("clean", lambda img: img)]

    for q in (90, 70, 50, 30):
        levels.append(DegradationLevel(f"jpeg_q{q}", lambda img, q=q: jpeg_compress(img, q)))

    for scale in (0.75, 0.5, 0.25):
        levels.append(DegradationLevel(f"resize_{scale}", lambda img, s=scale: resize_roundtrip(img, s)))

    for frac in (0.8, 0.6):
        levels.append(DegradationLevel(f"crop_{frac}", lambda img, f=frac: center_crop(img, f)))

    levels.append(DegradationLevel("screenshot_sim", screenshot_simulate))

    return levels
