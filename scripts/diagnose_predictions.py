"""
Diagnostic: prints raw logits (not just accuracy) for a handful of known
real and fake images, and confirms which CLIP config actually loaded.
Run this when results look suspicious (e.g. flat/constant predictions)
to tell apart "model is genuinely bad" from "something's misconfigured".

Usage:
    python -m scripts.diagnose_predictions --n 5
"""

import argparse
import pickle
import warnings
from pathlib import Path

import torch

from src.features.clip_embeddings import load_clip
from src.models.pretrained import CLIP_MODEL_NAME, load_pretrained_univfd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-pickle", type=Path, default=Path("data/raw/test_samples.pkl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/univfd/fc_weights.pth"))
    parser.add_argument("--n", type=int, default=5, help="How many real + fake samples to inspect")
    args = parser.parse_args()

    print(f"CLIP_MODEL_NAME in use: {CLIP_MODEL_NAME}")
    print("Loading CLIP (watch for a QuickGELU mismatch warning below -- there should be NONE):")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clip_model, preprocess = load_clip(model_name=CLIP_MODEL_NAME, device="cpu")
        gelu_warnings = [str(w.message) for w in caught if "quick" in str(w.message).lower()]
    if gelu_warnings:
        print("  !! QuickGELU warning STILL present:", gelu_warnings[0])
    else:
        print("  OK: no QuickGELU warning.")

    head = load_pretrained_univfd(args.checkpoint, device="cpu")

    with open(args.samples_pickle, "rb") as f:
        samples = pickle.load(f)

    reals = [s for s in samples if s.label == 0][: args.n]
    fakes = [s for s in samples if s.label == 1][: args.n]

    print(f"\nRaw logits (positive = model thinks FAKE, negative = model thinks REAL):")
    from PIL import Image

    with torch.no_grad():
        for label_name, group in [("REAL", reals), ("FAKE", fakes)]:
            print(f"\n  -- {label_name} images --")
            for s in group:
                img = Image.open(s.path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0)
                feat = clip_model.encode_image(tensor)
                feat_norm = feat / feat.norm(dim=-1, keepdim=True)
                logit = head(feat_norm).item()
                print(f"    {s.path.name:20s} (gen={s.generator:12s}) logit={logit:+.4f}  "
                      f"feat_norm_before_normalize={feat.norm().item():.4f}")


if __name__ == "__main__":
    main()