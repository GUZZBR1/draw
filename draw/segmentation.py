"""Opt-in adapter for the four-class single-frame SegFormer ONNX model.

This model segments visible pixels (background, paper, hand, ink). It does
not detect newly drawn pixels across time. The existing paper detector and
temporal accumulator remain separate consumers.
"""

import os

import cv2
import numpy as np

from . import config

BACKGROUND, PAPER, HAND, INK = range(4)
CLASS_NAMES = ("background", "paper", "hand", "ink")
INPUT_NAME = "image"
OUTPUT_NAME = "logits"
INPUT_SIZE = 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MODEL_ENV = "DRAW_SEGMENTATION_MODEL"


def _preprocess(frame_bgr):
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("frame precisa ser uma imagem BGR não vazia HxWx3")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32) / 255.0
    normalized = (resized - _MEAN) / _STD
    return normalized.transpose(2, 0, 1)[None].astype(np.float32)


def ink_observation(class_mask):
    """Map class IDs to visible ink evidence and known-visible pixels.

    Hand and background pixels are unknown to the temporal accumulator, so
    neither can add ink or vote to erase existing ink.
    """
    classes = np.asarray(class_mask)
    if classes.ndim != 2 or not np.isin(classes, np.arange(4)).all():
        raise ValueError("class_mask deve ser 2D e conter somente IDs 0..3")
    observed_ink = (classes == INK).astype(np.uint8) * 255
    visible = (classes == PAPER) | (classes == INK)
    return observed_ink, visible


def warp_class_mask(class_mask, corners, destination_size=None):
    """Perspective-warp categorical classes using nearest-neighbor only."""
    classes = np.asarray(class_mask)
    if classes.ndim != 2 or not np.isin(classes, np.arange(4)).all():
        raise ValueError("class_mask deve ser 2D e conter somente IDs 0..3")
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if destination_size is None:
        destination_size = (config.PAPER_W, config.PAPER_H)
    width, height = map(int, destination_size)
    if width <= 0 or height <= 0:
        raise ValueError("destination_size deve ser positivo")
    dst = np.asarray(
        [(0, 0), (width, 0), (width, height), (0, height)],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        classes.astype(np.uint8),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BACKGROUND,
    )


class InkSegmentationModel:
    """Lazy ONNX Runtime wrapper for four-class visible-pixel masks."""

    def __init__(self, model_path=None, intra_op_threads=4):
        self.model_path = model_path or os.environ.get(_MODEL_ENV)
        if not self.model_path:
            raise ValueError(
                f"informe o modelo ou defina {_MODEL_ENV}"
            )
        self.intra_op_threads = int(intra_op_threads)
        if self.intra_op_threads <= 0:
            raise ValueError("intra_op_threads deve ser positivo")
        self._session = None

    def load(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime não instalado") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.intra_op_threads
        try:
            session = ort.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"não foi possível carregar modelo ONNX {self.model_path}: {exc}"
            ) from exc
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != INPUT_NAME:
            raise ValueError(
                f"ONNX deve ter somente o input {INPUT_NAME!r}; "
                f"recebidos {[item.name for item in inputs]}"
            )
        if inputs[0].type != "tensor(float)":
            raise ValueError(
                f"input ONNX deve ser float32; veio {inputs[0].type}"
            )
        if len(outputs) != 1 or outputs[0].name != OUTPUT_NAME:
            raise ValueError(
                f"ONNX deve ter somente o output {OUTPUT_NAME!r}; "
                f"recebidos {[item.name for item in outputs]}"
            )
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1:] != [3, INPUT_SIZE, INPUT_SIZE]:
            raise ValueError(
                "input ONNX deve ser NCHW com 3 canais e 512x512; "
                f"recebido {shape}"
            )
        if outputs[0].type != "tensor(float)":
            raise ValueError(f"logits ONNX deve ser float32; veio {outputs[0].type}")
        output_shape = outputs[0].shape
        if len(output_shape) != 4 or output_shape[1] != 4:
            raise ValueError(
                "logits ONNX deve ter 4 canais; "
                f"recebido {output_shape}"
            )
        self._session = session
        return session

    def predict_classes(self, frame_bgr):
        """Return a HxW uint8 class map aligned to the input BGR frame."""
        session = self.load()
        frame = np.asarray(frame_bgr)
        tensor = _preprocess(frame)
        try:
            logits = session.run([OUTPUT_NAME], {INPUT_NAME: tensor})[0]
        except Exception as exc:
            raise RuntimeError(f"falha na inferência ONNX: {exc}") from exc
        if logits.ndim != 4 or logits.shape[:2] != (1, 4):
            raise ValueError(
                f"logits precisam ter shape [1,4,H,W]; vieram {logits.shape}"
            )
        if not logits.shape[2] or not logits.shape[3] or not np.isfinite(logits).all():
            raise ValueError("logits ONNX vazios, NaN ou infinitos")
        # Match train/eval: bilinear interpolation of logits before argmax.
        logits_512 = np.stack(
            [
                cv2.resize(
                    logits[0, class_id],
                    (INPUT_SIZE, INPUT_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
                for class_id in range(4)
            ],
            axis=0,
        )
        classes_512 = np.argmax(logits_512, axis=0).astype(np.uint8)
        frame_h, frame_w = frame.shape[:2]
        if (frame_h, frame_w) != (INPUT_SIZE, INPUT_SIZE):
            classes_512 = cv2.resize(
                classes_512,
                (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return classes_512

    def predict_masks(self, frame_bgr):
        """Return mutually-exclusive boolean masks keyed by class name."""
        classes = self.predict_classes(frame_bgr)
        return {name: classes == class_id for class_id, name in enumerate(CLASS_NAMES)}
