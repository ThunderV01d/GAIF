"""
Generates a tiny synthetic dataset in the expected genimage/ layout, purely
to smoke-test the pipeline (indexing -> eval harness -> plot output) before
spending time on the real GenImage download or the pretrained weights.

Not real data — random noise images. Only useful for confirming nothing in
the pipeline is broken; accuracy numbers from this are meaningless.

Usage:
    python -m scripts.make_synthetic_data
"""

from pathlib import Path

import numpy as np
from PIL import Image


def make_synthetic_data(root: Path = Path("data/raw/genimage"), n_per_class: int = 15, seed: int = 42):
    rng = np.random.default_rng(seed)

    for gen in ["gen_a", "gen_b"]:
        for label in ["real", "fake"]:
            d = root / gen / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                # Arbitrary distinction (smoother vs harsher noise) — just
                # needs to be non-degenerate for the pipeline to have
                # something to process, not a meaningful real/fake signal.
                if label == "real":
                    arr = rng.integers(80, 180, (256, 256, 3), dtype=np.uint8)
                else:
                    arr = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{i:03d}.jpg", quality=95)

    print(f"Synthetic smoke-test dataset written to {root}")


if __name__ == "__main__":
    make_synthetic_data()
