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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.augment.degradation import build_degradation_suite
from src.data.dataset import Sample
from src.features.clip_embeddings import load_clip
from src.models.heads import LinearProbeHead
from src.models.pretrained import CLIP_MODEL_NAME as UNIVFD_CLIP_MODEL_NAME
from src.models.pretrained import load_pretrained_univfd

# The CLIP backbone MUST match what the head was trained on, or you get
# either a shape-mismatch crash or (worse) silently wrong features if
# dims ever happened to coincide. 'custom' uses whatever backbone you
# passed to clip_embeddings.py at training time -- ViT-B-32 here matches
# that script's own default; change both together if you train with a
# different backbone.
CLIP_BACKBONE_BY_HEAD_TYPE = {
    "univfd": UNIVFD_CLIP_MODEL_NAME,  # ViT-L-14
    "custom": "ViT-B-32",
}


def load_head(head_type: str, checkpoint_path: Path, device: str = "cpu"):
    """
    head_type: 'custom' (our own trained LinearProbeHead, from
    train_linear_probe.py) or 'univfd' (published pretrained weights,
    downloaded via scripts/download_univfd_weights.sh).
    """
    if head_type == "univfd":
        return load_pretrained_univfd(checkpoint_path, device=device)

    if head_type == "custom":
        ckpt = torch.load(checkpoint_path, map_location=device)
        head = LinearProbeHead(embedding_dim=ckpt["embedding_dim"]).to(device)
        head.load_state_dict(ckpt["model_state"])
        head.eval()
        return head

    raise ValueError(f"Unknown head_type: {head_type!r} (expected 'custom' or 'univfd')")


@torch.no_grad()
def evaluate(
    samples: list[Sample],
    checkpoint_path: Path,
    head_type: str = "custom",
    device: str = "cpu",
) -> pd.DataFrame:
    clip_model, preprocess = load_clip(model_name=CLIP_BACKBONE_BY_HEAD_TYPE[head_type], device=device)
    head = load_head(head_type, checkpoint_path, device=device)

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
            logit_val = logit.item()
            pred = int(torch.sigmoid(logit).item() > 0.5)

            rows.append(
                {
                    "generator": sample.generator,
                    "label": sample.label,
                    "degradation": level.name,
                    "logit": logit_val,
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


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """roc_auc_score needs both classes present -- returns NaN otherwise
    rather than crashing (e.g. a generator bucket with no real pairs)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def summarize_auroc(df: pd.DataFrame) -> pd.DataFrame:
    """
    AUROC per (degradation level, generator), plus an overall row per level.
    Threshold-independent, unlike raw accuracy -- this is what actually
    tells you whether the model has discriminative signal at all, even if
    its decision threshold is miscalibrated for this domain (which is
    common when using an off-the-shelf pretrained head zero-shot on a
    different generator distribution than it was trained on).

    Per-generator AUROC pools that generator's own rows with every real
    (label=0) row available at the same degradation level, since real
    images aren't necessarily tied 1:1 to a specific generator in this
    dataset's layout (see scripts/fetch_small_test_set.py).
    """
    rows = []
    for degradation in df["degradation"].unique():
        level_df = df[df["degradation"] == degradation]

        rows.append({
            "degradation": degradation,
            "generator": "ALL",
            "auroc": _safe_auroc(level_df["label"].values, level_df["logit"].values),
        })

        for gen in level_df["generator"].unique():
            gen_rows = level_df[(level_df["generator"] == gen) | (level_df["label"] == 0)]
            rows.append({
                "degradation": degradation,
                "generator": gen,
                "auroc": _safe_auroc(gen_rows["label"].values, gen_rows["logit"].values),
            })

    return pd.DataFrame(rows)


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


def plot_auroc_vs_degradation(summary: pd.DataFrame, out_path: Path):
    """
    AUROC vs. degradation -- the metric to actually trust when the raw
    decision threshold may not be calibrated for this test distribution
    (e.g. a ProGAN-trained checkpoint evaluated on diffusion-model fakes).
    0.5 = no better than chance; 1.0 = perfect ranking.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    level_order = summary["degradation"].unique().tolist()

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")

    overall = summary[summary["generator"] == "ALL"].set_index("degradation").reindex(level_order)
    ax.plot(level_order, overall["auroc"], marker="o", linewidth=2, color="black", label="Overall")

    for gen in summary["generator"].unique():
        if gen == "ALL":
            continue
        gen_data = summary[summary["generator"] == gen].set_index("degradation").reindex(level_order)
        if gen_data["auroc"].isna().all():
            continue  # e.g. a real-only bucket with no fakes paired to it -- nothing to plot
        ax.plot(level_order, gen_data["auroc"], marker=".", alpha=0.4, linewidth=1, label=gen)

    ax.set_xlabel("Degradation level")
    ax.set_ylabel("AUROC")
    ax.set_title("Detection AUROC vs. real-world degradation")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-pickle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to either your trained checkpoint (--head-type custom) "
                             "or weights/univfd/fc_weights.pth (--head-type univfd)")
    parser.add_argument("--head-type", choices=["custom", "univfd"], default="custom")
    parser.add_argument("--out", type=Path, required=True, help="CSV of per-sample results summary (accuracy)")
    parser.add_argument("--plot", type=Path, required=True, help="Accuracy-vs-degradation plot")
    parser.add_argument("--auroc-out", type=Path, default=None,
                        help="CSV of AUROC summary (default: alongside --out, suffixed _auroc)")
    parser.add_argument("--auroc-plot", type=Path, default=None,
                        help="AUROC-vs-degradation plot (default: alongside --plot, suffixed _auroc)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    auroc_out = args.auroc_out or args.out.with_name(args.out.stem + "_auroc" + args.out.suffix)
    auroc_plot = args.auroc_plot or args.plot.with_name(args.plot.stem + "_auroc" + args.plot.suffix)

    with open(args.samples_pickle, "rb") as f:
        samples = pickle.load(f)

    raw_results = evaluate(samples, args.checkpoint, head_type=args.head_type, device=args.device)

    summary = summarize(raw_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"Saved summary to {args.out}")
    plot_accuracy_vs_degradation(summary, args.plot)

    auroc_summary = summarize_auroc(raw_results)
    auroc_out.parent.mkdir(parents=True, exist_ok=True)
    auroc_summary.to_csv(auroc_out, index=False)
    print(f"Saved AUROC summary to {auroc_out}")
    plot_auroc_vs_degradation(auroc_summary, auroc_plot)


if __name__ == "__main__":
    main()