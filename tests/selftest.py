"""Teste do pipeline sem webcam.

Monta uma cena sintética: pega a folha-modelo, desenha nela um círculo,
uma linha e um "X" por código, cola essa folha inclinada sobre um fundo
escuro (simulando a mesa vista pela webcam) e roda o pipeline real —
detecção, homografia e extração de tinta.

Passa se a folha for encontrada, a imagem retificada tiver o tamanho
esperado e uma fração razoável da tinta desenhada for recuperada.

Uso: python tests/selftest.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config, ink, paper  # noqa: E402
from template.generate_template import build_template  # noqa: E402


def build_scene():
    sheet = build_template()
    # Desenho "a mão" dentro da área útil.
    cv2.circle(sheet, (425, 500), 110, (0, 0, 0), 6)
    cv2.line(sheet, (280, 800), (570, 800), (0, 0, 0), 6)
    cv2.line(sheet, (320, 950), (530, 1080), (0, 0, 0), 6)
    cv2.line(sheet, (530, 950), (320, 1080), (0, 0, 0), 6)
    ground_truth_ink = ink.extract_ink(sheet)

    # Cola a folha inclinada sobre uma "mesa" escura.
    scene = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    src = np.array(
        [(0, 0), (config.PAPER_W, 0), (config.PAPER_W, config.PAPER_H),
         (0, config.PAPER_H)],
        dtype=np.float32,
    )
    dst = np.array(
        [(430, 130), (1150, 210), (1080, 1120), (330, 1040)], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    placed = cv2.warpPerspective(sheet, matrix, (1600, 1200))
    mask = cv2.warpPerspective(
        np.full(sheet.shape[:2], 255, dtype=np.uint8), matrix, (1600, 1200)
    )
    scene[mask > 0] = placed[mask > 0]
    return scene, ground_truth_ink


def main():
    scene, ground_truth_ink = build_scene()

    warped, markers = paper.warp_paper(scene)
    assert warped is not None, f"folha nao detectada (marcadores: {markers})"
    assert warped.shape[:2] == (config.PAPER_H, config.PAPER_W), warped.shape

    recovered = ink.extract_ink(warped)
    # Compara só a área útil central (longe das bordas e dos marcadores).
    region = (slice(200, 1100), slice(200, 650))
    truth_px = int((ground_truth_ink[region] > 0).sum())
    hits = int(
        ((recovered[region] > 0) & (ground_truth_ink[region] > 0)).sum()
    )
    recall = hits / max(truth_px, 1)
    print(f"tinta esperada: {truth_px}px | recuperada (acerto): {hits}px "
          f"| recall: {recall:.2%}")
    assert truth_px > 1000, "desenho sintetico quase sem tinta"
    assert recall > 0.60, f"recall baixo demais: {recall:.2%}"

    print("SELFTEST OK — deteccao, homografia e extracao de tinta funcionando.")


if __name__ == "__main__":
    main()
