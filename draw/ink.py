"""Extração da tinta: separar o que foi desenhado do papel branco.

Usa limiar adaptativo sobre a folha retificada — robusto a variações de
iluminação ao longo da página. As áreas dos marcadores são apagadas da
máscara para não virarem "tinta" no canvas.

Como a tinta é lida da folha a cada frame (e não acumulada), apagar com
a borracha funciona naturalmente: sumiu do papel, some do canvas.
"""

import cv2
import numpy as np

from . import config


def _marker_mask(shape):
    """Máscara (255 = ignorar) cobrindo as áreas dos marcadores."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    radius = config.MARKER_SIZE_PX // 2 + 15
    for center in config.MARKER_CENTERS.values():
        cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask


def extract_ink(warped):
    """Retorna a máscara de tinta (255 = tinta) da folha retificada."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=12,
    )
    ink = cv2.bitwise_not(binary)  # tinta escura vira branco (255)
    ink[_marker_mask(warped.shape) > 0] = 0
    # Remove ruído pequeno (poeira, textura do papel).
    kernel = np.ones((2, 2), dtype=np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    return ink


def render_canvas(ink_mask):
    """Renderiza o canvas digital: fundo branco, tinta preta."""
    canvas = np.full(
        (config.PAPER_H, config.PAPER_W, 3), 255, dtype=np.uint8
    )
    canvas[ink_mask > 0] = (0, 0, 0)
    return canvas
