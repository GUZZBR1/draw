"""Etapa 3 do segundo estágio de tinta: as três garantias, juntas.

Testa a implementação da branch (draw/ink_stage2.py, InkStage2),
sem modelo e sem rede: o especialista é uma máscara booleana injetada.
Classes: 0=background, 1=paper, 2=hand, 3=ink.

1. Desligado, o resultado é idêntico ao SegFormer.
2. Ligado, só a classe ink muda: fundo e mão não mudam, e todo pixel
   que muda é uma transição paper<->ink (paper vira tinta onde o
   especialista confirma; tinta sem confirmação volta a paper).
3. Entrada sem tinta do especialista não vira tinta: com o especialista
   em False em tudo, nenhuma tinta nova aparece (a tinta só encolhe,
   de volta para paper), e uma base sem tinta continua sem tinta.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw.ink_stage2 import BACKGROUND, HAND, INK, PAPER, InkStage2  # noqa: E402


def _base():
    return np.array(
        [
            [BACKGROUND, PAPER, HAND, INK],
            [PAPER, PAPER, HAND, BACKGROUND],
            [HAND, BACKGROUND, INK, PAPER],
            [BACKGROUND, HAND, PAPER, INK],
        ]
    )


def test_off_is_identical_to_segformer():
    seg = _base()
    out = InkStage2(enabled=False).refine(
        seg, specialist_mask=np.ones_like(seg, dtype=bool)
    )
    assert np.array_equal(out, seg)
    assert out is not seg


def test_on_only_ink_changes():
    seg = _base()
    specialist = np.array(
        [
            [True, True, True, False],
            [False, True, True, True],
            [True, False, False, True],
            [True, True, False, False],
        ],
        dtype=bool,
    )
    out = InkStage2(enabled=True).refine(seg, specialist_mask=specialist)
    # Fundo e mão: máscaras idênticas, pixel a pixel.
    assert np.array_equal(out == BACKGROUND, seg == BACKGROUND)
    assert np.array_equal(out == HAND, seg == HAND)
    # Todo pixel que mudou é paper<->ink, e tinta nova só onde o
    # especialista confirmou dentro de paper/ink.
    changed = out != seg
    assert np.all(np.isin(out[changed], [PAPER, INK]))
    assert np.all(np.isin(seg[changed], [PAPER, INK]))
    new_ink = (out == INK) & (seg != INK)
    assert np.all(new_ink <= specialist)
    assert np.all(new_ink <= (seg == PAPER))


def test_no_specialist_ink_creates_no_ink():
    seg = _base()
    out = InkStage2(enabled=True).refine(
        seg, specialist_mask=np.zeros_like(seg, dtype=bool)
    )
    # Nenhuma tinta nova: o conjunto de tinta só encolhe (volta a paper).
    assert np.all((out == INK) <= (seg == INK))
    # Base sem tinta continua sem tinta, mesmo com o estágio ligado.
    no_ink = np.array(
        [
            [BACKGROUND, PAPER, HAND],
            [PAPER, BACKGROUND, HAND],
        ]
    )
    out2 = InkStage2(enabled=True).refine(
        no_ink, specialist_mask=np.zeros_like(no_ink, dtype=bool)
    )
    assert np.array_equal(out2, no_ink)
    assert not np.any(out2 == INK)
