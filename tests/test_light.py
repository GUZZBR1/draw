"""Testes da normalização de luz, só sintéticos, sem hardware.

Variantes geradas por código a partir da cena sem marcadores do
selftest, no mesmo padrão dos builders de lá: sombra (gradiente +
mancha escura), gradiente, luz baixa (0,35x) e brilho (mancha clara).
O pipeline testado é o da fiação real do app: detecção no frame
normalizado, retificação do frame original, normalização da folha
antes da tinta.

Regras travadas aqui:
- Com a flag desligada, sombra e luz baixa NÃO são detectadas (é o
  gap medido, não um bug do teste).
- Com "both", sombra e luz baixa são detectadas, com erro de canto
  <= 12 px e recall >= 90%.
- Com "both", base e brilho não pioram mais que 2 pontos de recall
  em relação a desligada.
- Negativos continuam sem veredito de papel com "both": tapete e caixa
  podem subir de None para blank_maybe (o portão achata regiões
  uniformes), mas blank_maybe não é papel e detect_sheet continua
  devolvendo None. Nenhum falso positivo novo.
- Com "off", a normalização é identidade, então os números atuais
  não mudam.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw import app as app_module  # noqa: E402
from draw import ink as ink_module  # noqa: E402
from draw import light as light_module  # noqa: E402
from draw import paper as paper_module  # noqa: E402
import selftest as selftest_module  # noqa: E402

_REGION = (slice(200, 1100), slice(200, 650))


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


def _apply_gradient(scene):
    height, width = scene.shape[:2]
    gradient = np.linspace(0.55, 1.0, width)[None, :, None]
    return np.clip(scene.astype(np.float32) * gradient, 0, 255).astype(np.uint8)


def _apply_low_light(scene):
    return np.clip(scene.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)


def _apply_glare(scene):
    out = scene.astype(np.float32)
    height, width = out.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    blob = np.exp(
        -(((xx - 900) ** 2) / (2 * 180 ** 2)
          + ((yy - 450) ** 2) / (2 * 180 ** 2))
    )
    out = out + (blob * 180)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _pipeline(scene, ground_truth, corners_gt, mode):
    """Pipeline do app: detecção normalizada, warp do original, tinta
    da folha normalizada. Retorna (veredito, erro, recall)."""
    corners, verdict = paper_module.detect_sheet_with_confidence(
        app_module._detection_frame(scene, mode)
    )
    if corners is None:
        return verdict, None, None
    warped = paper_module.warp_sheet(scene, corners)
    recovered = ink_module.extract_ink(
        app_module._ink_input(warped, mode)
    )
    error = float(np.linalg.norm(corners - corners_gt, axis=1).max())
    hits = int(
        ((recovered[_REGION] > 0) & (ground_truth[_REGION] > 0)).sum()
    )
    truth_px = int((ground_truth[_REGION] > 0).sum())
    return verdict, error, hits / max(truth_px, 1)


def test_light_modes_and_off_is_identity():
    assert light_module.LIGHT_MODES == ("off", "background", "clahe", "both")
    image = np.random.default_rng(7).integers(0, 255, (50, 60, 3), dtype=np.uint8)
    assert np.array_equal(light_module.normalize_light(image, "off"), image)
    for mode in ("background", "clahe", "both"):
        out = light_module.normalize_light(image, mode)
        assert out.shape == image.shape and out.dtype == np.uint8
    with pytest.raises(ValueError):
        light_module.normalize_light(image, "retinex")


def test_background_division_flattens_gradient():
    height, width = 100, 200
    gradient = np.linspace(60, 240, width, dtype=np.uint8)[None, :, None]
    image = np.repeat(gradient, height, axis=0)
    image = np.repeat(image, 3, axis=2)
    flat = light_module.background_division(image)
    # Antes: desvio alto por causa do gradiente. Depois: quase plano.
    assert float(image[:, :, 0].std()) > 40.0
    assert float(flat[:, :, 0].std()) < 20.0


def test_shadow_and_low_light_rescued_by_both():
    scene, truth, corners_gt = selftest_module.build_markerless_scene()
    for name, variant in (
        ("sombra", _apply_shadow(scene)),
        ("luz baixa", _apply_low_light(scene)),
    ):
        # Sem flag, o gap medido é ausência de detecção.
        verdict_off, error_off, _ = _pipeline(
            variant, truth, corners_gt, "off"
        )
        assert verdict_off != "paper" and error_off is None, (
            f"{name} detectada sem flag: {verdict_off}"
        )
        verdict, error, recall = _pipeline(variant, truth, corners_gt, "both")
        assert verdict == "paper", f"{name} com both: {verdict}"
        assert error <= 12.0, f"{name} erro de canto: {error}"
        assert recall >= 0.90, f"{name} recall com both: {recall:.2%}"


def test_base_and_glare_do_not_regress_with_both():
    scene, truth, corners_gt = selftest_module.build_markerless_scene()
    for name, variant in (
        ("base", scene),
        ("gradiente", _apply_gradient(scene)),
        ("brilho", _apply_glare(scene)),
    ):
        _, _, recall_off = _pipeline(variant, truth, corners_gt, "off")
        verdict, error, recall = _pipeline(variant, truth, corners_gt, "both")
        assert verdict == "paper", f"{name} com both: {verdict}"
        assert error <= 12.0, f"{name} erro de canto: {error}"
        assert recall >= 0.90, f"{name} recall com both: {recall:.2%}"
        assert recall >= recall_off - 0.02, (
            f"{name} piorou mais que 2 pontos: {recall_off:.2%} -> {recall:.2%}"
        )


def test_negatives_stay_not_paper_with_both():
    scenes = selftest_module.build_negative_scenes()
    scenes["folha lisa"] = selftest_module.build_blank_sheet_scene()
    for name, scene in scenes.items():
        # Veredito diferente de "paper" já basta: detect_sheet só
        # devolve cantos quando o veredito é "paper".
        _, verdict = paper_module.detect_sheet_with_confidence(
            app_module._detection_frame(scene, "both")
        )
        assert verdict != "paper", f"falso positivo com both: {name}"


def test_app_run_accepts_light_normalize():
    import inspect
    params = inspect.signature(app_module.run).parameters
    assert params["light_normalize"].default == "off"
    assert app_module._detection_frame(
        np.zeros((10, 10, 3), dtype=np.uint8), "off"
    ).shape == (10, 10, 3)
