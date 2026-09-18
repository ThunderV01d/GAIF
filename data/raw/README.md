# Data layout (not committed to git)

## `genimage/` — main train/val/test source

Download from https://github.com/GenImage-Dataset/GenImage (request access
per their instructions). Arrange as:

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
