# Data layout (not committed to git)

## Do you need this at all?

Not necessarily. If you're just running the robustness eval against the
**pretrained UnivFD baseline** (see main README, step 3), you only need
a small labeled test set — a few hundred real + generated images is
enough to produce a meaningful robustness-vs-degradation chart. You do
NOT need the full GenImage download for that.

The full `genimage/` layout below is only needed if you're training your
**own** probe (`src/train/train_linear_probe.py`), which is an optional
stretch goal, not a requirement to get results.

## `genimage/` — main train/val/test source (only needed for training your own probe)

Full dataset: https://github.com/GenImage-Dataset/GenImage. See that
repo's README for download links (Google Drive / Kaggle mirrors tend to
be more reliable than the original Baidu host from outside Asia).

**Smaller alternative:** search for `tiny-genimage` on Kaggle/HuggingFace
— a reduced-size subset of GenImage, much more practical to download and
plenty for training a linear probe (which only needs a few thousand
examples per generator, not the full million-image set).

Arrange either as:

```
data/raw/genimage/
  sdv1.4/
    real/*.jpg
    fake/*.jpg
  sdv1.5/
    real/*.jpg
    fake/*.jpg
  midjourney/
    ...
  (etc. for ADM, GLIDE, Wukong, VQDM, BigGAN)
```

## `unseen/` — held-out generalization test set (hand-collected)

Small, hand-collected set (~200-500 images per generator is plenty) from
generators NOT in GenImage, e.g. Flux, Midjourney v6/v7, recent
DALL-E/Nano Banana. Real images can be a fresh sample of your own photos
or another photo set not overlapping with GenImage's real-image source.

```
data/raw/unseen/
  flux/
    real/*.jpg
    fake/*.jpg
  midjourney_v6/
    ...
```

**Important:** this set is never used for training — only for the final
generalization eval in `src/eval/robustness_eval.py`. Keep it that way,
or the "does this generalize to unseen generators" result is meaningless.

## Building the sample index

Once the above is populated:

```bash
python -m src.data.build_sample_index \
    --genimage-root data/raw/genimage \
    --unseen-root data/raw/unseen \
    --per-generator-cap 5000 \
    --out-dir data/raw
```

This produces `train_samples.pkl`, `val_samples.pkl`, `test_samples.pkl`,
and `unseen_samples.pkl` in `data/raw/`.
