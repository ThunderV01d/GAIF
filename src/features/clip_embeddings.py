"""
Extract and cache CLIP embeddings for a list of Samples.

Run once per dataset split. Caches to a .npz file so the (frozen) CLIP
backbone never needs to run again during linear-probe training/eval —
this is what makes training feasible on a weak/low-VRAM GPU: after this
step, training the classifier head is just matrix math on ~512-dim
vectors, not backprop through a ViT.

Usage:
    python -m src.features.clip_embeddings \
        --samples-pickle data/raw/train_val_samples.pkl \
        --out data/embeddings/train_val.npz
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from src.data.dataset import Sample


def load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval().to(device)
    return model, preprocess


@torch.no_grad()
def extract_embeddings(
    samples: list[Sample],
    model,
    preprocess,
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (embeddings [N, D], labels [N], generators [N]) as numpy arrays.
    Images that fail to load are skipped and logged, not silently dropped.
    """
    embeddings, labels, generators = [], [], []
    batch_imgs, batch_labels, batch_gens = [], [], []

    def flush():
        if not batch_imgs:
            return
        batch = torch.stack(batch_imgs).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize, standard for CLIP features
        embeddings.append(feats.cpu().numpy())
        labels.extend(batch_labels)
        generators.extend(batch_gens)
        batch_imgs.clear()
        batch_labels.clear()
        batch_gens.clear()

    failed = 0
    for sample in tqdm(samples, desc="Extracting CLIP embeddings"):
        try:
            img = Image.open(sample.path).convert("RGB")
        except Exception as e:
            failed += 1
            continue

        batch_imgs.append(preprocess(img))
        batch_labels.append(sample.label)
        batch_gens.append(sample.generator)

        if len(batch_imgs) >= batch_size:
            flush()

    flush()

    if failed:
        print(f"Warning: {failed} images failed to load and were skipped.")

    return np.concatenate(embeddings, axis=0), np.array(labels), generators


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-pickle", type=Path, required=True,
                        help="Pickled list[Sample] (from src.data.dataset)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(args.samples_pickle, "rb") as f:
        samples = pickle.load(f)

    model, preprocess = load_clip(device=args.device)
    embeddings, labels, generators = extract_embeddings(
        samples, model, preprocess, device=args.device, batch_size=args.batch_size
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, embeddings=embeddings, labels=labels, generators=generators)
    print(f"Saved {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim) to {args.out}")


if __name__ == "__main__":
    main()
