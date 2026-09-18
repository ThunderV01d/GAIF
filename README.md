# GAIF (Gen-AI Flagger)

A synthetic media detector focused not just on clean-lab accuracy, but on
**measuring how detection performance degrades under realistic, real-world
conditions** (JPEG recompression, resizing, cropping, screenshotting); a gap that current commercial detectors are known to struggle with.

## Scope & roadmap

This is being built in phases, each shipped to a rigorous standard before
moving to the next, rather than three shallow modalities at once:

- [x] **Phase 1 — Images** (in progress)
- [ ] **Phase 2 — Audio** (planned: ASVspoof-based synthetic speech detection)
- [ ] **Phase 3 — Video** (planned: frame-sampled baseline + temporal
      consistency heuristic; explicitly scoped as a simpler baseline, not a
      from-scratch video deepfake model)

## Phase 1: Image detector

### Approach

- **Backbone:** CLIP ViT-B/32 (frozen) + linear probe / small MLP head
- **Baseline comparison:** ResNet50, fine-tuned
- **Dataset:** [GenImage](https://github.com/GenImage-Dataset/GenImage)
  (real images from ImageNet vs. generated images from 8 generators:
  Midjourney, SDv1.4/1.5, ADM, GLIDE, Wukong, VQDM, BigGAN), supplemented
  with a small hand-collected set from generators *not* in GenImage
  (e.g. Flux, Midjourney v6/v7) to measure generalization to unseen
  generators — this is the more realistic test of "does this actually
  work in the wild."
- **The core contribution:** a robustness evaluation harness that measures
  accuracy vs. degradation severity (JPEG quality, resize, crop, screenshot
  simulation), reported per-architecture and per-generator. See
  `src/augment/degradation.py` and `src/eval/robustness_eval.py`.

### Repo structure

```
src/
  data/       dataset loading, GenImage indexing, unseen-generator set
  features/   CLIP embedding extraction + caching
  augment/    degradation transforms (JPEG, resize, crop, screenshot-sim)
  train/      linear probe + ResNet50 fine-tuning scripts
  eval/       robustness evaluation harness, plotting
data/
  raw/        raw image data (not committed — see data/raw/README.md)
  embeddings/ cached CLIP embeddings (not committed)
results/      trained checkpoints, eval plots, metrics tables
```

### Compute notes

Developed on a local GPU with <8GB VRAM. CLIP embeddings are precomputed
and cached once, so linear-probe training runs comfortably on modest
hardware. The heavier ResNet50 fine-tune and full robustness sweep are
run on Colab (free T4 tier) — see `notebooks/`.

### Status

Scaffold in progress — see individual module docstrings for what's
implemented vs. stubbed.
