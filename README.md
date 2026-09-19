# GAIF (Gen-AI Flagger)

A synthetic media detector focused not just on clean-lab accuracy, but on
**measuring how detection performance degrades under realistic, real-world
conditions** (JPEG recompression, resizing, cropping, screenshotting) — a
gap that current commercial detectors are known to struggle with.

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

The centerpiece of this project isn't "train a detector from scratch" —
it's the **robustness evaluation harness**: measuring how detection
accuracy holds up under real-world degradation (JPEG recompression,
resizing, cropping, screenshotting), which published detectors are
known to struggle with. Model training is secondary to that, so the
plan is built around getting a credible detector running fast:

1. **Pretrained baseline first (no training, no big dataset needed):**
   [UnivFD](https://arxiv.org/abs/2302.10174) (Ojha et al., CVPR 2023) —
   frozen CLIP ViT-B/32 + a single linear layer, trained on ProGAN data.
   Published weights via [SIDBench](https://huggingface.co/dkarageo/sidbench).
   See `src/models/pretrained.py` and `scripts/download_univfd_weights.sh`.
   Running this against GenImage's diffusion-model generators is itself
   a cross-generator generalization test, since UnivFD was trained on a
   GAN, not a diffusion model — worth reporting as a result in its own
   right, not just a convenience baseline.
2. **Robustness eval against the pretrained baseline** — this alone is a
   complete, reportable result, and doesn't require any dataset download
   beyond a small labeled test set.
3. **Own trained linear probe (optional, stretch):** same architecture
   family, trained on [GenImage](https://github.com/GenImage-Dataset/GenImage)
   (or its smaller `tiny-genimage` variant) — see `src/train/train_linear_probe.py`.
   Compared against the UnivFD baseline under the identical robustness
   sweep. Supplement with a small hand-collected set from generators
   *not* in GenImage (e.g. Flux, Midjourney v6/v7) to test generalization
   to unseen generators.
4. **Optional further comparison:** ResNet50, fine-tuned, as a second
   architecture point of comparison.

### Repo structure

```
src/
  data/       dataset loading, GenImage indexing, unseen-generator set
  features/   CLIP embedding extraction + caching
  models/     shared head architectures (custom MLP probe, UnivFD's
              linear head) + pretrained-weight loading
  augment/    degradation transforms (JPEG, resize, crop, screenshot-sim)
  train/      linear probe + (optional) ResNet50 fine-tuning scripts
  eval/       robustness evaluation harness, plotting — works against
              either the pretrained UnivFD head or your own trained one
scripts/      one-off setup scripts (e.g. downloading pretrained weights)
data/
  raw/        raw image data (not committed — see data/raw/README.md)
  embeddings/ cached CLIP embeddings (not committed)
weights/      downloaded pretrained checkpoints (not committed)
results/      trained checkpoints, eval plots, metrics tables
```

### Compute notes

Developed on a local GPU with <8GB VRAM. CLIP embeddings are precomputed
and cached once, so linear-probe training runs comfortably on modest
hardware. The heavier ResNet50 fine-tune and full robustness sweep are
run on Colab (free T4 tier) — see `notebooks/`.

### Getting started

```bash
pip install -r requirements.txt

# 1. Get pretrained UnivFD weights (run locally — needs huggingface.co access)
bash scripts/download_univfd_weights.sh

# 2. Build a small labeled test set (see data/raw/README.md) and index it
python -m src.data.build_sample_index \
    --genimage-root data/raw/genimage --out-dir data/raw

# 3. Run the robustness eval against the pretrained baseline — no
#    training required for this step
python -m src.eval.robustness_eval \
    --samples-pickle data/raw/test_samples.pkl \
    --checkpoint weights/univfd/fc_weights.pth \
    --head-type univfd \
    --out results/robustness_report.csv \
    --plot results/robustness_plot.png
```

Training your own probe (optional) follows the same eval command with
`--head-type custom` and your own checkpoint from
`src/train/train_linear_probe.py`.

### Status

Scaffold in progress — see individual module docstrings for what's
implemented vs. stubbed.
