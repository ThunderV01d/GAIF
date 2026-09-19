#!/usr/bin/env bash
# Downloads the pretrained UnivFD linear-probe weights via SIDBench's
# HuggingFace repo. Run this locally (not in a network-restricted
# environment) — huggingface.co needs to be reachable.
#
# Usage: bash scripts/download_univfd_weights.sh

set -e

pip install -U "huggingface_hub[cli]"

hf download dkarageo/sidbench --local-dir weights --include "univfd/*"

echo ""
echo "Done. Weights at: weights/univfd/fc_weights.pth"
echo "Point src/models/pretrained.py's load_pretrained_univfd() at that path."
