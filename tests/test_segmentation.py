import sys
import types

import numpy as np
import pytest

from draw.app import _ink_observation_for_frame, _resolve_segmentation_model
from draw.ink import InkAccumulator
from draw.segmentation import (
    BACKGROUND,
    HAND,
    INK,
    PAPER,
    InkSegmentationModel,
    _preprocess,
    ink_observation,
    warp_class_mask,
)


class _Meta:
    def __init__(self, name, shape, type_="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_


class _FakeSession:
    def __init__(self, logits=None, output_name="logits"):
        self.input = _Meta("image", [None, 3, 512, 512])
        self.output = _Meta(output_name, [None, 4, 2, 2])
        self.logits = logits
        self.last_input = None

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [self.output]

    def run(self, names, inputs):
        assert names == ["logits"]
        self.last_input = inputs["image"]
        return [self.logits]


def _model(session):
    model = InkSegmentationModel("synthetic-test.onnx")
    model._session = session
    return model


def test_app_segmentation_path_is_opt_in_and_has_separate_environment_key():
    assert _resolve_segmentation_model(None, {}) is None
    assert _resolve_segmentation_model(
        None, {"DRAW_NEURAL_MODEL": "legacy.onnx"}
    ) is None
    assert _resolve_segmentation_model(None, {"DRAW_SEGMENTATION_MODEL": "seg.onnx"}) == "seg.onnx"
    assert _resolve_segmentation_model("explicit.onnx", {}) == "explicit.onnx"


def test_app_keeps_classic_path_by_default_and_uses_it_on_fallback(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    calls = []

    def classic(image):
        calls.append(image)
        return np.full((2, 2), 255, dtype=np.uint8)

    monkeypatch.setattr("draw.app.ink_module.extract_ink", classic)
    observed, visible = _ink_observation_for_frame(frame)
    assert np.all(observed == 255)
    assert visible is None

    # `None` is the app's result when optional ONNX inference fails.
    fallback_observed, fallback_visible = _ink_observation_for_frame(
        frame, class_mask=None
    )
    assert np.array_equal(fallback_observed, observed)
    assert fallback_visible is None
    assert len(calls) == 2 and all(image is frame for image in calls)


def test_app_sends_four_class_observation_to_accumulator_boundary():
    classes = np.array([[BACKGROUND, PAPER], [HAND, INK]], dtype=np.uint8)
    observed, visible = _ink_observation_for_frame(
        np.zeros((2, 2, 3), dtype=np.uint8), class_mask=classes
    )
    assert observed.tolist() == [[0, 0], [0, 255]]
    assert visible.tolist() == [[False, True], [False, True]]


def test_preprocess_matches_training_rgb_imagenet_contract():
    # BGR red -> RGB red: normalized red channel receives the high value.
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    frame[0, 0] = (0, 0, 255)
    tensor = _preprocess(frame)
    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == np.float32
    assert np.allclose(tensor[0, 0, 0, 0], (1.0 - 0.485) / 0.229)
    assert np.allclose(tensor[0, 1, 0, 0], (0.0 - 0.456) / 0.224)
    assert np.allclose(tensor[0, 2, 0, 0], (0.0 - 0.406) / 0.225)


def test_predict_resizes_logits_before_argmax_and_returns_named_masks():
    # Four low-resolution cells choose one class apiece.
    logits = np.zeros((1, 4, 2, 2), dtype=np.float32)
    for y in range(2):
        for x in range(2):
            logits[0, y * 2 + x, y, x] = 3.0
    fake = _FakeSession(logits)
    model = _model(fake)
    frame = np.zeros((9, 13, 3), dtype=np.uint8)
    masks = model.predict_masks(frame)
    assert list(masks) == ["background", "paper", "hand", "ink"]
    assert all(mask.shape == (9, 13) and mask.dtype == bool for mask in masks.values())
    assert np.all(np.sum(np.stack(list(masks.values())), axis=0) == 1)
    assert set(np.unique(model.predict_classes(frame))) == {0, 1, 2, 3}
    assert fake.last_input.shape == (1, 3, 512, 512)
    assert fake.last_input.dtype == np.float32


def test_invalid_output_shape_and_nonfinite_logits_fail_closed():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    wrong_channels = _FakeSession(np.zeros((1, 3, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=r"shape \[1,4,H,W\]"):
        _model(wrong_channels).predict_classes(frame)

    bad_logits = np.zeros((1, 4, 2, 2), dtype=np.float32)
    bad_logits[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN ou infinitos"):
        _model(_FakeSession(bad_logits)).predict_classes(frame)


def test_onnx_runtime_execution_error_is_wrapped_for_app_fallback():
    class RuntimeProviderError(Exception):
        pass

    fake = _FakeSession(np.zeros((1, 4, 2, 2), dtype=np.float32))

    def fail_run(*args, **kwargs):
        raise RuntimeProviderError("provider execution failed")

    fake.run = fail_run
    with pytest.raises(RuntimeError, match="provider execution failed"):
        _model(fake).predict_classes(np.zeros((8, 8, 3), dtype=np.uint8))


def test_load_validates_names_and_static_input_contract(monkeypatch):
    class FakeOrtSession(_FakeSession):
        def __init__(self, input_name="image", input_shape=None):
            super().__init__(np.zeros((1, 4, 2, 2), dtype=np.float32))
            self.input.name = input_name
            if input_shape is not None:
                self.input.shape = input_shape

    ort_session = FakeOrtSession()
    fake_ort = types.SimpleNamespace(
        SessionOptions=lambda: types.SimpleNamespace(intra_op_num_threads=0),
        InferenceSession=lambda *args, **kwargs: ort_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    assert InkSegmentationModel("unused.onnx").load() is ort_session

    wrong_input = FakeOrtSession(input_name="pixels")
    fake_ort.InferenceSession = lambda *args, **kwargs: wrong_input
    with pytest.raises(ValueError, match="input 'image'"):
        InkSegmentationModel("unused.onnx").load()

    wrong_shape = FakeOrtSession(input_shape=[None, 3, 384, 384])
    fake_ort.InferenceSession = lambda *args, **kwargs: wrong_shape
    with pytest.raises(ValueError, match="512x512"):
        InkSegmentationModel("unused.onnx").load()

    class OrtLoadError(Exception):
        pass

    def fail_load(*args, **kwargs):
        raise OrtLoadError("invalid model file")

    fake_ort.InferenceSession = fail_load
    with pytest.raises(RuntimeError, match="invalid model file"):
        InkSegmentationModel("bad.onnx").load()


def test_warp_class_mask_preserves_ids_with_nearest_neighbor():
    classes = np.array(
        [[BACKGROUND, PAPER, HAND, INK], [INK, HAND, PAPER, BACKGROUND],
         [PAPER, INK, BACKGROUND, HAND]],
        dtype=np.uint8,
    )
    corners = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=np.float32)
    warped = warp_class_mask(classes, corners, destination_size=(4, 3))
    assert np.array_equal(warped, classes)
    assert set(np.unique(warped)) == {BACKGROUND, PAPER, HAND, INK}


def test_hand_and_background_are_unknown_to_temporal_accumulator():
    classes = np.array([[INK, HAND, BACKGROUND, PAPER]], dtype=np.uint8)
    observed, visible = ink_observation(classes)
    assert observed.tolist() == [[255, 0, 0, 0]]
    assert visible.tolist() == [[True, False, False, True]]

    accumulator = InkAccumulator(
        shape=(1, 4), window=4, enter=0.45, exit=0.2,
        tolerance=0, min_visible=1,
    )
    accumulator.update(observed, visible)
    assert accumulator.state[0, 0]
    # Repeated occlusion or background predictions do not cast clear votes.
    for unknown in (HAND, BACKGROUND, HAND, BACKGROUND):
        unknown_observed, unknown_visible = ink_observation(
            np.array([[unknown]], dtype=np.uint8)
        )
        accumulator.update(unknown_observed, unknown_visible)
    assert accumulator.state[0, 0]
    # Visible paper may cast a clear vote under the existing hysteresis rule.
    for _ in range(4):
        paper_observed, paper_visible = ink_observation(
            np.array([[PAPER]], dtype=np.uint8)
        )
        accumulator.update(paper_observed, paper_visible)
    assert not accumulator.state[0, 0]
