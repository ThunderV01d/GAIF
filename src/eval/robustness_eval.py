"""
Robustness evaluation harness — the centerpiece of this project.

Takes a trained detector (CLIP embedding + linear probe head) and a set
of test images, applies each degradation level from
src/augment/degradation.py, and reports accuracy per degradation level
per generator. Produces the accuracy-vs-degradation-severity chart that
is the main result of the image phase.

Unlike training, this MUST run on raw images (not cached embeddings),
because the degradation has to be applied to the actual pixels before
re-encoding with CLIP each time.

Usage:
    python -m src.eval.robustness_eval \
        --samples-pickle data/raw/test_samples.pkl \
        --checkpoint results/linear_probe.pt \
        --out results/robustness_report.csv \
        --plot results/robustness_plot.png
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.augment.degradation import build_degradation_suite
from src.data.dataset import Sample
from src.features.clip_embeddings import load_clip
from src.train.train_linear_probe import LinearProbeHead


@torch.no_grad()
def evaluate(
    samples: list[Sample],
    checkpoint_path: Path,
    device: str = "cpu",
) -> pd.DataFrame:
    clip_model, preprocess = load_clip(device=device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    head = LinearProbeHead(embedding_dim=ckpt["embedding_dim"]).to(device)
    head.load_state_dict(ckpt["model_state"])
    head.eval()

    degradation_levels = build_degradation_suite()
    rows = []

    for sample in tqdm(samples, desc="Evaluating robustness"):
        try:
            img = Image.open(sample.path).convert("RGB")
        except Exception:
            continue

        for level in degradation_levels:
            degraded = level.fn(img)
            tensor = preprocess(degraded).unsqueeze(0).to(device)
            feat = clip_model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logit = head(feat)
            pred = int(torch.sigmoid(logit).item() > 0.5)

            rows.append(
                {
                    "generator": sample.generator,
                    "label": sample.label,
                    "degradation": level.name,
                    "pred": pred,
                    "correct": int(pred == sample.label),
                }
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (degradation level, generator), plus an overall row per level."""
    per_gen = df.groupby(["degradation", "generator"])["correct"].mean().reset_index()
    per_gen = per_gen.rename(columns={"correct": "accuracy"})

    overall = df.groupby("degradation")["correct"].mean().reset_index()
    overall["generator"] = "ALL"
    overall = overall.rename(columns={"correct": "accuracy"})

    return pd.concat([per_gen, overall], ignore_index=True)


def plot_accuracy_vs_degradation(summary: pd.DataFrame, out_path: Path):
    """The headline chart: accuracy vs. degradation, overall + per generator."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Order degradation levels roughly clean -> most degraded for a readable x-axis.
    level_order = summary["degradation"].unique().tolist()

    overall = summary[summary["generator"] == "ALL"].set_index("degradation").reindex(level_order)
    ax.plot(level_order, overall["accuracy"], marker="o", linewidth=2, color="black", label="Overall")

    for gen in summary["generator"].unique():
        if gen == "ALL":
            continue
        gen_data = summary[summary["generator"] == gen].set_index("degradation").reindex(level_order)
        ax.plot(level_order, gen_data["accuracy"], marker=".", alpha=0.4, linewidth=1, label=gen)

    ax.set_xlabel("Degradation level")
    ax.set_ylabel("Accuracy")
    ax.set_title("Detection accuracy vs. real-world degradation")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-pickle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="CSV of per-sample results summary")
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.samples_pickle, "rb") as f:
        samples = pickle.load(f)

    raw_results = evaluate(samples, args.checkpoint, device=args.device)
    summary = summarize(raw_results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"Saved summary to {args.out}")

    plot_accuracy_vs_degradation(summary, args.plot)


if __name__ == "__main__":
    main()
