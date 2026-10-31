"""Model path selection and page/ink model separation."""

import sys
import types

import pytest

from draw.app import (
    resolve_ink_model,
    resolve_neural_model,
    resolve_pair_ink_model,
)
from draw.neural import _load_session


def test_neural_model_is_disabled_without_cli_or_environment():
    assert resolve_neural_model(None, None) is None


def test_environment_model_activates_without_cli_flag():
    assert resolve_neural_model(None, "page.onnx") == "page.onnx"


def test_explicit_model_path_overrides_environment():
    assert resolve_neural_model("explicit.onnx", "environment.onnx") == (
        "explicit.onnx"
    )


def test_empty_neural_flag_requires_environment_path():
    with pytest.raises(ValueError, match="DRAW_NEURAL_MODEL"):
        resolve_neural_model("", None)


def test_neural_model_cannot_be_used_with_markers_mode():
    with pytest.raises(ValueError, match="--markers"):
        resolve_neural_model("page.onnx", None, markers_mode=True)


def test_markers_mode_ignores_environment_model():
    assert resolve_neural_model(None, "page.onnx", markers_mode=True) is None


def test_ink_model_is_disabled_without_cli_or_environment():
    assert resolve_ink_model(None, None) is None


def test_ink_environment_model_activates_without_cli_flag():
    assert resolve_ink_model(None, "pair.onnx") == "pair.onnx"


def test_explicit_ink_model_overrides_environment():
    assert resolve_ink_model("explicit.onnx", "environment.onnx") == (
        "explicit.onnx"
    )


def test_empty_ink_model_flag_requires_environment_path():
    with pytest.raises(ValueError, match="DRAW_INK_MODEL"):
        resolve_ink_model("", None)


def test_markers_mode_ignores_ink_environment_model():
    assert resolve_ink_model(None, "pair.onnx", markers_mode=True) is None


def test_ink_model_flag_cannot_be_used_with_markers_mode():
    with pytest.raises(ValueError, match="--markers"):
        resolve_ink_model("pair.onnx", None, markers_mode=True)


def test_pair_ink_model_is_disabled_without_cli_or_environment():
    assert resolve_pair_ink_model(None, None) is None


def test_pair_ink_environment_model_activates_without_cli_flag():
    assert resolve_pair_ink_model(None, "pair.onnx") == "pair.onnx"


def test_explicit_pair_ink_path_overrides_environment():
    assert resolve_pair_ink_model("explicit.onnx", "environment.onnx") == (
        "explicit.onnx"
    )


def test_empty_pair_ink_flag_requires_environment_path():
    with pytest.raises(ValueError, match="DRAW_PAIR_INK_MODEL"):
        resolve_pair_ink_model("", None)


def test_pair_ink_model_flag_cannot_be_used_with_markers_mode():
    with pytest.raises(ValueError, match="--markers"):
        resolve_pair_ink_model("pair.onnx", None, markers_mode=True)


def test_pairwise_ink_model_is_rejected_as_page_detector(tmp_path, monkeypatch):
    class FakeSession:
        def get_inputs(self):
            return [types.SimpleNamespace(name="pair")]

    class FakeRuntime(types.ModuleType):
        class SessionOptions:
            pass

        @staticmethod
        def InferenceSession(*_args, **_kwargs):
            return FakeSession()

    model_path = tmp_path / "pair-model.onnx"
    model_path.write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeRuntime("onnxruntime"))

    with pytest.raises(ValueError, match="temporais de tinta"):
        _load_session(model_path)
