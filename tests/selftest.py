"""Teste do pipeline sem webcam.

Monta uma cena sintética: pega a folha-modelo, desenha nela um círculo,
uma linha e um "X" por código, cola essa folha inclinada sobre um fundo
escuro (simulando a mesa vista pela webcam) e roda o pipeline real,
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


def build_markerless_scene():
    sheet = np.full(
        (config.PAPER_H, config.PAPER_W, 3), 248, dtype=np.uint8
    )
    cv2.circle(sheet, (425, 500), 110, (0, 0, 0), 6)
    cv2.line(sheet, (280, 800), (570, 800), (0, 0, 0), 6)
    cv2.line(sheet, (320, 950), (530, 1080), (0, 0, 0), 6)
    cv2.line(sheet, (530, 950), (320, 1080), (0, 0, 0), 6)
    ground_truth_ink = ink.extract_ink(sheet)

    scene = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    src = np.array(
        [(0, 0), (config.PAPER_W, 0), (config.PAPER_W, config.PAPER_H),
         (0, config.PAPER_H)],
        dtype=np.float32,
    )
    dst = np.array(
        [(430, 130), (1150, 210), (1080, 1120), (330, 1040)],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    placed = cv2.warpPerspective(sheet, matrix, (1600, 1200))
    mask = cv2.warpPerspective(
        np.full(sheet.shape[:2], 255, dtype=np.uint8), matrix, (1600, 1200)
    )
    scene[mask > 0] = placed[mask > 0]
    return scene, ground_truth_ink, dst


def build_nested_scene():
    sheet = np.full(
        (config.PAPER_H, config.PAPER_W, 3), 248, dtype=np.uint8
    )
    cv2.circle(sheet, (425, 500), 110, (0, 0, 0), 8)
    cv2.line(sheet, (280, 800), (570, 800), (0, 0, 0), 8)
    cv2.line(sheet, (320, 950), (530, 1080), (0, 0, 0), 8)
    cv2.line(sheet, (530, 950), (320, 1080), (0, 0, 0), 8)
    ground_truth_ink = ink.extract_ink(sheet)

    # Mesa escura com um tapete claro cobrindo quase tudo. O tapete e uma
    # elipse com um corte retangular fundo no topo. O contorno dele e o
    # maior da cena, nao e convexo e nunca fecha como quadrilatero. A
    # folha fica aninhada dentro dele, como a pagina do caderno aberto
    # sobre a mesa na cena real que motivou este teste.
    scene = np.full((1200, 1600, 3), 60, dtype=np.uint8)
    cv2.ellipse(scene, (800, 600), (770, 545), 0, 0, 360, (150, 150, 150), -1)
    cv2.rectangle(scene, (1100, 0), (1560, 640), (60, 60, 60), -1)

    src = np.array(
        [(0, 0), (config.PAPER_W, 0), (config.PAPER_W, config.PAPER_H),
         (0, config.PAPER_H)],
        dtype=np.float32,
    )
    dst = np.array(
        [(460, 180), (1120, 250), (1050, 1080), (380, 1000)],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    placed = cv2.warpPerspective(sheet, matrix, (1600, 1200))

    # Mascara com cantos levemente arredondados, como papel de verdade.
    radius = 6
    w, h = config.PAPER_W, config.PAPER_H
    mask_src = np.zeros((config.PAPER_H, config.PAPER_W), dtype=np.uint8)
    cv2.rectangle(mask_src, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask_src, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in ((radius, radius), (w - radius, radius),
                   (w - radius, h - radius), (radius, h - radius)):
        cv2.circle(mask_src, (cx, cy), radius, 255, -1)
    mask = cv2.warpPerspective(mask_src, matrix, (1600, 1200))
    scene[mask > 0] = placed[mask > 0]
    return scene, ground_truth_ink, dst


def nested_main():
    scene, ground_truth_ink, ground_truth_corners = build_nested_scene()

    corners = paper.detect_sheet(scene)
    assert corners is not None, "folha aninhada nao detectada"
    corner_errors = np.linalg.norm(corners - ground_truth_corners, axis=1)
    max_corner_error = float(corner_errors.max())
    print(f"cantos da folha aninhada: erro maximo: {max_corner_error:.2f}px")
    assert max_corner_error <= 12.0, corner_errors

    warped = paper.warp_sheet(scene, corners)
    assert warped.shape[:2] == (config.PAPER_H, config.PAPER_W), warped.shape

    recovered = ink.extract_ink(warped)
    region = (slice(200, 1100), slice(200, 650))
    truth_px = int((ground_truth_ink[region] > 0).sum())
    hits = int(
        ((recovered[region] > 0) & (ground_truth_ink[region] > 0)).sum()
    )
    recall = hits / max(truth_px, 1)
    print(f"tinta esperada: {truth_px}px | recuperada (acerto): {hits}px "
          f"| recall: {recall:.2%}")
    assert truth_px > 1000, "desenho sintetico quase sem tinta"
    assert recall >= 0.90, f"recall baixo demais: {recall:.2%}"

    print("SELFTEST FOLHA ANINHADA OK. Contorno interno encontrado.")


def markerless_main():
    scene, ground_truth_ink, ground_truth_corners = build_markerless_scene()

    corners = paper.detect_sheet(scene)
    assert corners is not None, "folha sem marcadores nao detectada"
    corner_errors = np.linalg.norm(corners - ground_truth_corners, axis=1)
    max_corner_error = float(corner_errors.max())
    print(f"cantos sem marcadores: erro maximo: {max_corner_error:.2f}px")
    assert max_corner_error <= 12.0, corner_errors

    warped = paper.warp_sheet(scene, corners)
    assert warped.shape[:2] == (config.PAPER_H, config.PAPER_W), warped.shape

    recovered = ink.extract_ink(warped)
    region = (slice(200, 1100), slice(200, 650))
    truth_px = int((ground_truth_ink[region] > 0).sum())
    hits = int(
        ((recovered[region] > 0) & (ground_truth_ink[region] > 0)).sum()
    )
    recall = hits / max(truth_px, 1)
    print(f"tinta esperada: {truth_px}px | recuperada (acerto): {hits}px "
          f"| recall: {recall:.2%}")
    assert truth_px > 1000, "desenho sintetico quase sem tinta"
    assert recall >= 0.90, f"recall baixo demais: {recall:.2%}"

    print("SELFTEST SEM MARCADORES OK. Deteccao por contorno funcionando.")


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

    print("SELFTEST OK. Deteccao, homografia e extracao de tinta funcionando.")


if __name__ == "__main__":
    markerless_main()
    nested_main()
    main()
