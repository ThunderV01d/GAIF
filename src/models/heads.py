"""
Classifier heads that sit on top of frozen CLIP embeddings.

Two heads are defined here because they're not interchangeable:

- `LinearProbeHead`: the small MLP head this project trains from scratch
  (see src/train/train_linear_probe.py). Has a hidden layer + dropout.

- `UnivFDHead`: a plain single linear layer (logistic regression on CLIP
  features), matching the architecture used in Ojha et al., "Towards
  Universal Fake Image Detectors that Generalize Across Generative
  Models" (CVPR 2023). Pretrained weights for this exact head are
  published (see src/models/pretrained.py) — it MUST stay architecturally
  identical to the original (no hidden layer) or the published weights
  won't load correctly.

Both take L2-normalized CLIP image embeddings as input and output a
single logit (sigmoid -> probability the image is AI-generated).
"""

import torch.nn as nn


class LinearProbeHead(nn.Module):
    """Our own trained head: embedding_dim -> hidden -> 1."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class UnivFDHead(nn.Module):
    """
    UnivFD's original head: a single linear layer, no hidden layer.
    Matches the architecture the published `fc_weights.pth` checkpoint
    was trained with (CLIP ViT-B/32 -> 512-dim embeddings -> 1 logit).
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)
