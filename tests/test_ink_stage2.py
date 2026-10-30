"""Testes do segundo estágio de tinta: só numpy, sem modelo, sem rede."""

import numpy as np

from draw.ink_stage2 import BACKGROUND, HAND, INK, PAPER, InkStage2


def _mask():
    # 3x3: fundo, papel, mão, tinta, papel.
    return np.array(
        [
            [BACKGROUND, PAPER, HAND],
            [PAPER, INK, PAPER],
            [HAND, PAPER, INK],
        ]
    )


def test_disabled_returns_copy_unchanged():
    stage = InkStage2(enabled=False)
    seg = _mask()
    out = stage.refine(seg, specialist_mask=np.ones((3, 3), dtype=bool))
    assert np.array_equal(out, seg)
    assert out is not seg


def test_specialist_adds_ink_only_on_paper_or_ink():
    stage = InkStage2(enabled=True)
    seg = _mask()
    # Especialista vê tinta em tudo, inclusive fundo e mão: só papel/tinta vira tinta.
    out = stage.refine(seg, specialist_mask=np.ones((3, 3), dtype=bool))
    assert out[0, 0] == BACKGROUND  # fundo não vira tinta
    assert out[0, 2] == HAND  # mão nunca é tocada
    assert out[1, 0] == INK  # papel com tinta do especialista vira tinta


def test_specialist_removes_ink_back_to_paper():
    stage = InkStage2(enabled=True)
    seg = _mask()
    out = stage.refine(seg, specialist_mask=np.zeros((3, 3), dtype=bool))
    assert out[1, 1] == PAPER  # tinta sem confirmação volta a papel
    assert out[2, 2] == PAPER
    assert out[0, 2] == HAND


def test_no_specialist_input_returns_unchanged():
    stage = InkStage2(enabled=True, model_path="/nonexistent/draw_ink_v2.onnx")
    seg = _mask()
    assert np.array_equal(stage.refine(seg), seg)


def test_from_env_off_by_default(monkeypatch):
    monkeypatch.delenv("DRAW_INK_V2_ENABLED", raising=False)
    assert InkStage2.from_env().enabled is False
