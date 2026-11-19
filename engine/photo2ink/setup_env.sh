#!/usr/bin/env bash
# Separate Python environment for InkSight + docTR (TensorFlow), so TensorFlow
# never shares a process with PyTorch. Ends by loading InkSight and decoding one
# test word; a non-zero exit means the setup failed.
#
#   bash engine/photo2ink/setup_env.sh [ENV_DIR]          # versions from the official InkSight notebook
#   INK_PINS=card bash engine/photo2ink/setup_env.sh      # versions from the Hugging Face model card
set -euo pipefail
ENV=${1:-/content/ink_env}
PINS=${INK_PINS:-repo}

pip -q install uv
uv venv -q --python 3.11 "$ENV"
if [ "$PINS" = card ]; then
  TF=("tensorflow[and-cuda]==2.17.1" "tensorflow-text==2.17.0" "tf-keras==2.17.0")
else
  TF=("tensorflow[and-cuda]==2.20.0" "tensorflow-text==2.20.1" "tf-keras==2.20.1")
fi
uv pip install --python "$ENV/bin/python" "${TF[@]}" "huggingface_hub<1.0" "python-doctr[tf]==0.10.0" \
  pillow matplotlib numpy
"$ENV/bin/python" -m engine.photo2ink.inksight --selftest
