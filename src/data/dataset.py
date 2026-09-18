"""
Dataset indexing for the image detector.

Expects data laid out under `data/raw/` as:

    data/raw/genimage/<generator_name>/{real,fake}/*.jpg
    data/raw/unseen/<generator_name>/{real,fake}/*.jpg

`genimage/` holds the generators used for train/val (GenImage's own split).
`unseen/` holds the hand-collected, held-out generators (e.g. Flux,
Midjourney v6/v7) used ONLY at eval time to measure generalization to
generators the model never saw during training. Do not train on this set.

Labels: 0 = real, 1 = AI-generated.
"""

import random
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Sample:
    path: Path
    label: int          # 0 = real, 1 = fake
    generator: str       # e.g. "sdv1.5", "midjourney", "real" for real images


def _list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def index_split(root: Path, generators: list[str] | None = None) -> list[Sample]:
    """
    Index a directory shaped like root/<generator>/{real,fake}/*.

    If `generators` is None, indexes every generator subdirectory found.
    """
    samples: list[Sample] = []
    gen_dirs = (
        [root / g for g in generators] if generators else sorted(d for d in root.iterdir() if d.is_dir())
    )

    for gen_dir in gen_dirs:
        if not gen_dir.is_dir():
            continue
        generator_name = gen_dir.name

        for label, subdir_name in [(0, "real"), (1, "fake")]:
            for img_path in _list_images(gen_dir / subdir_name):
                samples.append(Sample(path=img_path, label=label, generator=generator_name))

    return samples


def stratified_subsample(samples: list[Sample], per_generator_cap: int, seed: int = 42) -> list[Sample]:
    """
    Cap the number of real + fake samples per generator, so a handful of
    generators with huge counts don't dominate training. Keeps the dataset
    at a sane size for local/Colab-scale training.
    """
    rng = random.Random(seed)
    by_key: dict[tuple[str, int], list[Sample]] = {}
    for s in samples:
        by_key.setdefault((s.generator, s.label), []).append(s)

    out: list[Sample] = []
    for key, group in by_key.items():
        rng.shuffle(group)
        out.extend(group[:per_generator_cap])
    return out


def load_dataset(
    genimage_root: Path,
    unseen_root: Path | None = None,
    per_generator_cap: int = 5000,
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """
    Returns a dict with 'train_val' (subsampled GenImage samples, to be
    split into train/val by the caller) and 'unseen' (full, uncapped —
    this set is small by construction and used only for eval).
    """
    train_val = index_split(genimage_root)
    train_val = stratified_subsample(train_val, per_generator_cap=per_generator_cap, seed=seed)

    unseen = index_split(unseen_root) if unseen_root else []

    return {"train_val": train_val, "unseen": unseen}
