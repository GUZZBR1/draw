"""Contract checks for the experimental pairwise ink model adapter."""

import sys
import types

import numpy as np
import pytest

from draw.pair_ink import PairInkDetector, _load_session, build_pair_input


def test_build_pair_input_matches_training_channel_order_and_delta():
    after = np.array([[0, 255, 128]], dtype=np.uint8)
    before = np.array([[255, 0, 128]], dtype=np.uint8)

    tensor = build_pair_input(after, before)

    assert tensor.shape == (1, 3, 1, 3)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, 0, 0], [0.0, 1.0, 128 / 255])
    np.testing.assert_allclose(tensor[0, 1, 0], [1.0, 0.0, 128 / 255])
    np.testing.assert_allclose(tensor[0, 2, 0], [1.0, 0.0, 0.5])


@pytest.mark.parametrize(
    ("after", "before", "message"),
    [
        (np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.uint8), "tons de cinza"),
        (np.zeros((2, 2), np.uint8), np.zeros((3, 2), np.uint8), "mesma resolução"),
        (np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32), "uint8"),
    ],
)
def test_build_pair_input_rejects_invalid_frames(after, before, message):
    with pytest.raises(ValueError, match=message):
        build_pair_input(after, before)


class _FakeSession:
    def __init__(self, input_name="pair", output_name="logits",
                 input_shape=(1, 3, "h", "w"),
                 output_shape=(1, 3, "h", "w"), result=None):
        self._inputs = [types.SimpleNamespace(name=input_name, shape=input_shape)]
        self._outputs = [types.SimpleNamespace(name=output_name, shape=output_shape)]
        self.result = result
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, names, feed):
        assert names == ["logits"]
        self.feed = feed
        return [self.result]


def test_pair_ink_detector_returns_only_class_two(monkeypatch):
    logits = np.zeros((1, 3, 8, 8), dtype=np.float32)
    logits[0, 2, 0, 1] = 2.0
    logits[0, 1, 1, 2] = 2.0
    session = _FakeSession(result=logits)
    detector = PairInkDetector("unused.onnx")
    detector._session = session

    mask = detector.predict(
        np.zeros((2, 3), dtype=np.uint8), np.full((2, 3), 255, dtype=np.uint8)
    )

    expected = np.zeros((2, 3), dtype=np.uint8)
    expected[0, 1] = 255
    np.testing.assert_array_equal(mask, expected)
    assert session.feed["pair"].shape == (1, 3, 8, 8)


@pytest.mark.parametrize(
    ("input_name", "output_name", "input_shape", "output_shape", "message"),
    [
        ("image", "logits", (1, 3, 4, 4), (1, 3, 4, 4), "entrada 'pair'"),
        ("pair", "mask", (1, 3, 4, 4), (1, 3, 4, 4), "saída 'logits'"),
        ("pair", "logits", (1, 1, 4, 4), (1, 3, 4, 4), "[N, 3, H, W]"),
        ("pair", "logits", (1, 3, 4, 4), (1, 4, 4, 4), "[N, 3, H, W]"),
    ],
)
def test_load_session_rejects_incompatible_contract(
    tmp_path, monkeypatch, input_name, output_name, input_shape,
    output_shape, message,
):
    class FakeRuntime(types.ModuleType):
        class SessionOptions:
            pass

        @staticmethod
        def InferenceSession(*_args, **_kwargs):
            return _FakeSession(input_name, output_name, input_shape, output_shape)

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeRuntime("onnxruntime"))

    with pytest.raises(ValueError, match=message):
        _load_session(model_path)


def test_prediction_rejects_nonfinite_logits():
    logits = np.zeros((1, 3, 8, 8), dtype=np.float32)
    logits[0, 2, 0, 0] = np.nan
    detector = PairInkDetector("unused.onnx")
    detector._session = _FakeSession(result=logits)

    with pytest.raises(ValueError, match="não finitos"):
        detector.predict(
            np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)
        )


def test_prediction_rejects_spatially_mismatched_logits():
    logits = np.zeros((1, 3, 1, 2), dtype=np.float32)
    detector = PairInkDetector("unused.onnx")
    detector._session = _FakeSession(result=logits)

    with pytest.raises(ValueError, match="resolução de logits"):
        detector.predict(
            np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)
        )
