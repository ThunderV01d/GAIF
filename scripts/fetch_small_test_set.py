"""
Fetch a small, real labeled test set by STREAMING from the Tiny-GenImage
dataset on HuggingFace, rather than downloading the full ~8GB dataset.

Streaming mode pulls examples on demand, so requesting e.g. 300 images
only transfers those 300 images, not the whole dataset. Needs
huggingface.co access (won't work in a network-restricted sandbox).

Source: https://huggingface.co/datasets/TheKernel01/Tiny-GenImage
Labels: 0 = real, 1 = fake. Generator field distinguishes which
generative model produced each fake image (or 'Real' for real images).

Usage:
    python -m scripts.fetch_small_test_set --n-per-class 40 --out-dir data/raw/genimage
"""

import argparse
from pathlib import Path

from datasets import load_dataset

GENERATOR_LABEL_NAMES = {
    0: "real_shared",  # all real images share one bucket -- see note below
    1: "adm",
    2: "biggan",
    3: "glide",
    4: "midjourney",
    5: "sd14",
    6: "sd15",
    7: "vqdm",
    8: "wukong",
}


def fetch_small_test_set(out_dir: Path, n_per_class: int = 40, split: str = "validation"):
    """
    Pulls n_per_class REAL images (all bucketed as 'real_shared') and
    n_per_class FAKE images from EACH of the 8 generators (so ~8x
    n_per_class fake images total), matching the genimage/<generator>/
    {real,fake}/ layout src/data/dataset.py expects.

    Real images all land under real_shared/real/ rather than one bucket
    per generator, since Tiny-GenImage doesn't pair specific real images
    to specific generators -- this keeps the layout honest rather than
    fabricating a pairing that doesn't exist in the source data.
    """
    print(f"Streaming {split} split from TheKernel01/Tiny-GenImage (no full download)...")
    ds = load_dataset("TheKernel01/Tiny-GenImage", split=split, streaming=True)

    real_count = 0
    fake_counts = {name: 0 for name in GENERATOR_LABEL_NAMES.values() if name != "real_shared"}
    target_fake_per_generator = n_per_class

    for example in ds:
        label = example["label"]
        generator_id = example["generator"]
        generator_name = GENERATOR_LABEL_NAMES.get(generator_id, f"unknown_{generator_id}")

        if label == 0:
            if real_count >= n_per_class:
                continue
            subdir = out_dir / "real_shared" / "real"
            idx = real_count
            real_count += 1
        else:
            if fake_counts.get(generator_name, 0) >= target_fake_per_generator:
                continue
            subdir = out_dir / generator_name / "fake"
            idx = fake_counts[generator_name]
            fake_counts[generator_name] += 1

        subdir.mkdir(parents=True, exist_ok=True)
        example["image"].convert("RGB").save(subdir / f"{idx:04d}.jpg", quality=95)

        # Stop once every bucket is full.
        if real_count >= n_per_class and all(c >= target_fake_per_generator for c in fake_counts.values()):
            break

    print(f"Saved {real_count} real images to {out_dir}/real_shared/real/")
    for name, count in fake_counts.items():
        print(f"Saved {count} fake images to {out_dir}/{name}/fake/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=40,
                        help="Real images fetched, and fake images fetched PER GENERATOR "
                             "(so total fake count is ~8x this number)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/genimage"))
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    fetch_small_test_set(args.out_dir, n_per_class=args.n_per_class, split=args.split)


if __name__ == "__main__":
    main()