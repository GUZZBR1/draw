"""Inference adapter for the temporal pairwise ink ONNX model.

This model is separate from :mod:`draw.neural`: it receives aligned grayscale
``before``/``after`` images and predicts newly drawn ink, not a full ink mask
or a page quadrilateral. The app uses it only as optional additional evidence
alongside its existing full-frame heuristic.
"""

import os
from pathlib import Path

import numpy as np

_INPUT_NAME = "pair"
_OUTPUT_NAME = "logits"
_CLASS_COUNT = 3
_NEW_INK_CLASS = 2
_SPATIAL_MULTIPLE = 8
_DEFAULT_ORT_THREADS = 4


def _ort_intra_threads():
    raw = os.environ.get("DRAW_ORT_THREADS", "")
    try:
        value = int(raw)
    except ValueError:
        value = 0
    return value if value > 0 else _DEFAULT_ORT_THREADS


def build_pair_input(after, before):
    """Build the training-compatible NCHW tensor from grayscale uint8 frames."""
    after = np.asarray(after)
    before = np.asarray(before)
    if after.ndim != 2 or before.ndim != 2:
        raise ValueError("before e after devem ser imagens em tons de cinza")
    if after.shape != before.shape:
        raise ValueError(
            "before e after devem ter a mesma resolução e estar alinhados"
        )
    if min(after.shape) == 0:
        raise ValueError("before e after não podem estar vazios")
    if after.dtype != np.uint8 or before.dtype != np.uint8:
        raise ValueError("before e after devem usar pixels uint8")

    a = after.astype(np.float32) / 255.0
    b = before.astype(np.float32) / 255.0
    delta = np.clip((b - a) * 255.0 / 48.0, -1.0, 1.0) * 0.5 + 0.5
    return np.stack([a, b, delta], axis=0)[None].astype(np.float32)


def _validate_model_contract(session):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_names = [item.name for item in inputs]
    output_names = [item.name for item in outputs]
    if input_names != [_INPUT_NAME]:
        raise ValueError(
            f"modelo de tinta pareada: esperava entrada '{_INPUT_NAME}', "
            f"recebi {input_names}"
        )
    if output_names != [_OUTPUT_NAME]:
        raise ValueError(
            f"modelo de tinta pareada: esperava saída '{_OUTPUT_NAME}', "
            f"recebi {output_names}"
        )

    input_shape = inputs[0].shape
    output_shape = outputs[0].shape
    if len(input_shape) != 4 or input_shape[1] != 3:
        raise ValueError(
            "entrada 'pair' precisa ter forma [N, 3, H, W]"
        )
    if len(output_shape) != 4 or output_shape[1] != _CLASS_COUNT:
        raise ValueError(
            "saída 'logits' precisa ter forma [N, 3, H, W] "
            "(resto, conteúdo antigo, tinta nova)"
        )


def _load_session(model_path):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime nao esta instalado; o modelo de tinta pareada "
            "precisa dele. Instale com: pip install onnxruntime"
        ) from error

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"modelo de tinta ONNX nao encontrado: {path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = _ort_intra_threads()
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    _validate_model_contract(session)
    return session


class PairInkDetector:
    """Return a mask of new ink from two aligned grayscale frames."""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self._session = None

    def load(self):
        """Load and validate the ONNX session now, so failures happen early."""
        if self._session is None:
            self._session = _load_session(self.model_path)
        return self

    def predict(self, after, before):
        """Return a uint8 mask (255 = newly drawn ink) for one frame pair."""
        height, width = after.shape
        pad_bottom = (-height) % _SPATIAL_MULTIPLE
        pad_right = (-width) % _SPATIAL_MULTIPLE
        if pad_bottom or pad_right:
            after = np.pad(
                after, ((0, pad_bottom), (0, pad_right)), mode="edge"
            )
            before = np.pad(
                before, ((0, pad_bottom), (0, pad_right)), mode="edge"
            )
        tensor = build_pair_input(after, before)
        if self._session is None:
            self.load()
        logits = self._session.run([_OUTPUT_NAME], {_INPUT_NAME: tensor})[0]
        logits = np.asarray(logits)
        expected_prefix = (1, _CLASS_COUNT)
        if logits.ndim != 4 or logits.shape[:2] != expected_prefix:
            raise ValueError(
                "logits precisa ter forma [1, 3, H, W]; "
                f"recebi {logits.shape}"
            )
        if logits.shape[2:] != after.shape:
            raise ValueError(
                "a resolução de logits precisa corresponder às imagens; "
                f"recebi {logits.shape[2:]} para {after.shape}"
            )
        if not np.isfinite(logits).all():
            raise ValueError("logits do modelo de tinta contêm valores não finitos")
        new_ink = np.argmax(logits[0], axis=0) == _NEW_INK_CLASS
        return (new_ink[:height, :width].astype(np.uint8) * 255)
