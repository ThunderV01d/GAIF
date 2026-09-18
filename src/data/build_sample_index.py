"""
Convenience script: index raw data directories into pickled Sample lists
that the rest of the pipeline (clip_embeddings.py, robustness_eval.py)
consumes. Also does the train/val/test split.

Usage:
    python -m src.data.build_sample_index \
        --genimage-root data/raw/genimage \
        --unseen-root data/raw/unseen \
        --per-generator-cap 5000 \
        --out-dir data/raw
"""

import argparse
import pickle
import random
from pathlib import Path

from src.data.dataset import load_dataset


def split_train_val_test(samples, val_fraction=0.15, test_fraction=0.15, seed=42):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)

    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genimage-root", type=Path, required=True)
    parser.add_argument("--unseen-root", type=Path, default=None)
    parser.add_argument("--per-generator-cap", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    data = load_dataset(
        genimage_root=args.genimage_root,
        unseen_root=args.unseen_root,
        per_generator_cap=args.per_generator_cap,
    )

    train, val, test = split_train_val_test(data["train_val"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test), ("unseen", data["unseen"])]:
        path = args.out_dir / f"{name}_samples.pkl"
        with open(path, "wb") as f:
            pickle.dump(split, f)
        print(f"{name}: {len(split)} samples -> {path}")


if __name__ == "__main__":
    main()
