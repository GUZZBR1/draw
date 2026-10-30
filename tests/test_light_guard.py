"""Testes da guarda combinada de luz (draw/light_guard.py).

Regras travadas aqui, todas medidas nas cenas sintéticas do
selftest / measure_paper, sem mudar nenhum limiar de paper/ink/light:
- Com a flag ligada, se "off" já detectar, fica com "off" (base).
- Se "off" não detectar, tenta "background" (sombra, luz baixa).
- Resultado de "background" é descartado quando o interior do quad,
  medido no frame ORIGINAL, é majoritariamente saturado (kraft:
  fração S < 60 ~ 0,01, bem abaixo de 0,5).
- Resultado de "background" é descartado quando o quad é de quadro
  inteiro (área > 90% do frame e 4 cantos perto das bordas).
- A margem de borda é relativa: 2,5% do menor lado do frame.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw import config  # noqa: E402
from draw import light as light_module  # noqa: E402
from draw import light_guard as guard_module  # noqa: E402
from draw import paper as paper_module  # noqa: E402
import selftest as selftest_module  # noqa: E402
from tools.measure_paper import (  # noqa: E402
    BASE_DST,
    draw_on_sheet,
    place_sheet,
)


def _apply_shadow(scene):
    out = scene.astype(np.float32)
    height, width = out.shape[:2]
    gradient = np.linspace(0.45, 1.0, width)[None, :, None]
    out = out * gradient
    yy, xx = np.mgrid[0:height, 0:width]
    blob = np.exp(
        -(((xx - 700) ** 2) / (2 * 300 ** 2)
          + ((yy - 500) ** 2) / (2 * 250 ** 2))
    )
    out = out * (1 - 0.5 * blob)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def test_off_wins_when_it_already_detects():
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
        scene, "both"
    )
    assert verdict == "paper" and mode == "off"
    assert float(np.linalg.norm(corners - corners_gt, axis=1).max()) <= 12.0


def test_background_rescues_shadow_and_low_light():
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    for variant in (
        _apply_shadow(scene),
        np.clip(scene.astype(np.float32) * 0.35, 0, 255).astype(np.uint8),
    ):
        corners_off, _ = paper_module.detect_sheet_with_confidence(variant)
        assert corners_off is None
        corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
            variant, "both"
        )
        assert verdict == "paper" and mode == "background"
        assert float(np.linalg.norm(corners - corners_gt, axis=1).max()) <= 12.0


def test_off_flag_is_unchanged():
    scene, _, _ = selftest_module.build_markerless_scene()
    variant = _apply_shadow(scene)
    corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
        variant, "off"
    )
    assert (corners, verdict, mode) == (None, None, "off")


def test_kraft_detected_by_background_is_discarded():
    kraft = draw_on_sheet(
        np.full((config.PAPER_H, config.PAPER_W, 3), (140, 180, 210),
                dtype=np.uint8)
    )
    scene = place_sheet(kraft, BASE_DST)
    # Sem a guarda, background detectaria kraft como papel: a recusa
    # abaixo é da guarda, não de ausência de detecção.
    bg_corners, bg_verdict = paper_module.detect_sheet_with_confidence(
        light_module.normalize_light(scene, "background")
    )
    assert bg_verdict == "paper" and bg_corners is not None
    assert guard_module.unsaturated_fraction(scene, BASE_DST) < 0.5
    assert not guard_module.guard_passes(scene, bg_corners)
    corners, verdict, mode = guard_module.detect_sheet_with_light_guard(
        scene, "both"
    )
    assert (corners, verdict, mode) == (None, None, "off")


def test_edge_margin_is_relative_to_frame_size():
    assert guard_module.edge_margin_px((1200, 1600, 3)) == 30.0
    assert guard_module.edge_margin_px((600, 800, 3)) == 15.0
    assert guard_module.edge_margin_px((2400, 3200, 3)) == 60.0


def test_full_frame_quad_detection():
    shape = (1200, 1600, 3)
    full = np.array(
        [(0, 0), (1599, 0), (1599, 1199), (0, 1199)], dtype=np.float32
    )
    assert guard_module.is_full_frame_quad(shape, full)
    # Área ainda acima de 90%, mas um canto longe de todas as bordas:
    # não é quadro inteiro.
    one_corner_in = np.array(
        [(100, 100), (1599, 0), (1599, 1199), (0, 1199)],
        dtype=np.float32,
    )
    assert not guard_module.is_full_frame_quad(shape, one_corner_in)
    # Folha normal da cena base: área bem abaixo de 90%.
    scene, _, corners_gt = selftest_module.build_markerless_scene()
    assert not guard_module.is_full_frame_quad(shape, corners_gt)
    assert guard_module.guard_passes(scene, corners_gt)
    assert not guard_module.guard_passes(scene, full)
