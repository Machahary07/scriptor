"""Google InkSight Small-p: a word image in, the pen strokes that wrote it out.

InkSight [Mitrevski et al., 2024] reads a 224 x 224 word image and returns text
followed by ink tokens. Loading, padding and token decoding follow the official
inference notebook (github.com/google-research/inksight, Apache-2.0).

TensorFlow is imported inside functions only. This module runs in the separate
environment built by setup_env.sh, never in the PyTorch training process.

    python -m engine.photo2ink.inksight --selftest   # load the model and decode one image
"""

from __future__ import annotations

import os
import re
import sys
import time

import numpy as np
from PIL import Image

MODEL_ID = "Derendering/InkSight-Small-p"
PROMPT = "Recognize and derender."
FALLBACK_PROMPT = "Derender the ink."  # used when the first answer has no strokes
CANVAS = 224
_TOKEN = re.compile(r"<ink_token_(\d+)>")
_SPECIAL = re.compile(r"<[^<>\s]+>")  # ink tokens and sentinels such as <extra_id_0>


def configure_tf() -> None:
    """Environment the released checkpoint expects; must run before TensorFlow is imported."""
    flags = ("--xla_gpu_autotune_level=0 "
             "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter,custom_kernel-fusion-autotuner")
    if flags not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {flags}".strip()
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # from_pretrained_keras needs Keras 2 (tf_keras)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def scale_and_pad(crop: Image.Image) -> tuple[Image.Image, float, int, int]:
    """Fit the crop into a black 224 x 224 square. Returns image, ratio, dx, dy."""
    ratio = min(CANVAS / crop.width, CANVAS / crop.height)
    small = crop.convert("RGB").resize((max(1, int(crop.width * ratio)), max(1, int(crop.height * ratio))))
    out = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    dx, dy = (CANVAS - small.width) // 2, (CANVAS - small.height) // 2
    out.paste(small, (dx, dy))
    return out, ratio, dx, dy


def text_to_tokens(text: str) -> list[int]:
    return [int(t) for t in _TOKEN.findall(text)]


def recognised_text(text: str) -> str:
    """The words InkSight read, i.e. its output without the ink tokens."""
    return " ".join(_SPECIAL.sub(" ", text).split())


def detokenize(tokens: list[int]) -> list[np.ndarray]:
    """Ink tokens to strokes, each an (N, 2) array in 224-canvas pixels.

    x is the token value, y is the token minus 225, and 450 starts a stroke.
    """
    per_dim = CANVAS + 1
    start = per_dim * 2
    if any(t < 0 or t > start for t in tokens):
        raise ValueError(f"ink token indices should be between 0 and {start}")
    strokes, cur, i = [], [], 0
    while i < len(tokens):
        if tokens[i] == start:
            if cur:
                strokes.append(cur)
            cur = []
            i += 1
        elif i + 1 < len(tokens) and tokens[i + 1] != start:
            x, y = tokens[i], tokens[i + 1] - per_dim
            if 0 <= x <= CANVAS and 0 <= y <= CANVAS:
                cur.append((x, y))
            i += 2
        else:  # an x without its y
            i += 1
    if cur:
        strokes.append(cur)
    return [np.asarray(s, dtype=np.float64) for s in strokes]


def load_model():
    """Returns (serving function, how it was loaded)."""
    configure_tf()
    import tensorflow as tf
    import tensorflow_text  # noqa: F401  registers the text ops the model uses

    try:
        from huggingface_hub import from_pretrained_keras

        model = from_pretrained_keras(MODEL_ID)
        how = "huggingface_hub.from_pretrained_keras"
    except Exception as e:  # same weights, loaded as a plain SavedModel
        from huggingface_hub import snapshot_download

        print(f"from_pretrained_keras failed ({type(e).__name__}: {e}); loading the SavedModel directly",
              file=sys.stderr)
        model = tf.saved_model.load(snapshot_download(MODEL_ID))
        how = f"tf.saved_model.load (from_pretrained_keras failed: {type(e).__name__})"
    return model.signatures["serving_default"], how


class InkSight:
    def __init__(self):
        self.cf, self.load_method = load_model()

    def _call(self, images: list[Image.Image], prompt: str) -> list[str]:
        import tensorflow as tf

        enc = tf.stack([tf.reshape(tf.io.encode_jpeg(np.asarray(im)[:, :, :3]), (1,)) for im in images])
        out = self.cf(**{"input_text": tf.constant([prompt] * len(images), dtype=tf.string), "image/encoded": enc})
        return [row[0].decode() for row in out["output_0"].numpy()]

    def run(self, images: list[Image.Image], batch_size: int = 32, log=print) -> list[dict]:
        """For each 224 x 224 image: strokes (canvas px), recognised text and the prompt used."""
        res = []
        t0 = time.time()
        for b in range(0, len(images), batch_size):
            batch = images[b : b + batch_size]
            for im, text in zip(batch, self._call(batch, PROMPT)):
                strokes, prompt = detokenize(text_to_tokens(text)), PROMPT
                if not strokes:
                    retry = self._call([im], FALLBACK_PROMPT)[0]
                    strokes, prompt = detokenize(text_to_tokens(retry)), FALLBACK_PROMPT
                res.append({"strokes": strokes, "text": recognised_text(text), "prompt": prompt})
            log(f"  InkSight {len(res)}/{len(images)} words ({time.time() - t0:.0f} s)")
        return res


def _selftest() -> int:
    from PIL import ImageDraw

    t0 = time.time()
    model = InkSight()
    print(f"loaded {MODEL_ID} via {model.load_method} in {time.time() - t0:.0f} s")
    import tensorflow as tf

    print("tensorflow", tf.__version__, "GPUs:", [d.name for d in tf.config.list_physical_devices("GPU")])
    im = Image.new("RGB", (160, 60), "white")
    ImageDraw.Draw(im).text((10, 15), "hello", fill="black", font_size=30)
    t1 = time.time()
    r = model.run([scale_and_pad(im)[0]], log=lambda *_: None)[0]
    print(f"decoded test word in {time.time() - t1:.1f} s: text={r['text']!r}, "
          f"{len(r['strokes'])} strokes, {sum(len(s) for s in r['strokes'])} points")
    return 0 if r["strokes"] else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
