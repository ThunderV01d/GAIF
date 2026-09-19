"""
Load the pretrained UnivFD linear-probe weights.

These are NOT downloaded automatically by this script (this environment's
network access doesn't include huggingface.co) — download them yourself
first. See scripts/download_univfd_weights.sh for the exact command, or
just run:

    pip install -U "huggingface_hub[cli]"
    hf download dkarageo/sidbench --local-dir weights --include "univfd/*"

That gets you `weights/univfd/fc_weights.pth`. Point this loader at it.

Source: Ojha et al., "Towards Universal Fake Image Detectors that
Generalize Across Generative Models" (CVPR 2023). Weights bundled via
the SIDBench project (https://huggingface.co/dkarageo/sidbench), trained
on ProGAN data with a frozen CLIP ViT-L/14 backbone (768-dim embeddings
-- NOT ViT-B/32's 512-dim; using the wrong backbone here will either
crash on a shape mismatch or, worse, silently run with the wrong
features if the dims ever happened to coincide). Use CLIP_MODEL_NAME
below wherever you extract embeddings to feed this head, e.g.:

    from src.models.pretrained import CLIP_MODEL_NAME
    load_clip(model_name=CLIP_MODEL_NAME)

NOTE: because this was trained on ProGAN (a GAN, not a diffusion model),
running it against GenImage's diffusion-model generators (SD, Midjourney,
etc.) is itself a cross-generator generalization test — worth reporting
as its own result, not just a convenience baseline.
"""

from pathlib import Path

import torch

from src.models.heads import UnivFDHead

CLIP_MODEL_NAME = "ViT-L-14"
CLIP_EMBEDDING_DIM = 768


def load_pretrained_univfd(weights_path: Path, device: str = "cpu") -> UnivFDHead:
    """
    Loads the published UnivFD linear head. Raises a clear error (rather
    than a confusing shape-mismatch traceback) if the file is missing or
    doesn't look like the expected checkpoint.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No UnivFD weights found at {weights_path}. "
            "Download them first — see this module's docstring or "
            "scripts/download_univfd_weights.sh."
        )

    state_dict = torch.load(weights_path, map_location=device)

    # SIDBench's checkpoint may be saved as a raw state_dict or wrapped
    # in a dict with a 'model'/'state_dict' key depending on version —
    # handle both rather than failing silently on the wrong one.
    if isinstance(state_dict, dict) and "fc.weight" not in state_dict and "weight" not in state_dict:
        for key in ("model", "state_dict", "model_state"):
            if key in state_dict:
                state_dict = state_dict[key]
                break

    # The actual checkpoint stores the linear layer's params as bare
    # "weight"/"bias" (it was saved directly from an nn.Linear, not from
    # a wrapper module) — remap to match our UnivFDHead's "fc.weight"/
    # "fc.bias" naming.
    if "weight" in state_dict and "fc.weight" not in state_dict:
        state_dict = {
            "fc.weight": state_dict["weight"],
            "fc.bias": state_dict["bias"],
        }

    head = UnivFDHead(embedding_dim=CLIP_EMBEDDING_DIM)
    try:
        head.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint at {weights_path} didn't match the expected UnivFD "
            f"architecture (single linear layer, {CLIP_EMBEDDING_DIM} -> 1). "
            f"Original error: {e}"
        )

    head.eval().to(device)
    return head
