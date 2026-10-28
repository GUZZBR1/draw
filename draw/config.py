"""Configuração central do Draw.

A folha-modelo é um A4 com quatro marcadores ArUco, um em cada canto.
Todas as coordenadas aqui estão no espaço da folha retificada
(a "vista de cima" que o pipeline produz).
"""

import cv2

# Tamanho da folha retificada em pixels (proporção A4: 210 x 297 mm).
PAPER_W = 850
PAPER_H = 1200

# Marcadores: id -> nome do canto.
MARKER_IDS = (0, 1, 2, 3)  # 0=sup. esq., 1=sup. dir., 2=inf. dir., 3=inf. esq.
MARKER_SIZE_PX = 120

# Centros dos marcadores no espaço da folha retificada.
MARKER_CENTERS = {
    0: (90, 90),
    1: (PAPER_W - 90, 90),
    2: (PAPER_W - 90, PAPER_H - 90),
    3: (90, PAPER_H - 90),
}

ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50


def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
