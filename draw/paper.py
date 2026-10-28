"""Detecção da folha e correção de perspectiva (homografia).

A webcam vê a folha inclinada. Com os centros dos quatro marcadores
ArUco, calculamos a homografia que leva a folha do frame da câmera para
o espaço retificado definido em config.py — uma vista de cima perfeita.
"""

import cv2
import numpy as np

from . import config


def _make_detector():
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(config.get_dictionary(), params)


_DETECTOR = _make_detector()


def detect_markers(frame):
    """Retorna {id: centro (x, y)} dos marcadores encontrados no frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    found = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id in config.MARKER_IDS:
                center = marker_corners[0].mean(axis=0)
                found[marker_id] = (float(center[0]), float(center[1]))
    return found


def warp_paper(frame):
    """Retifica a folha do frame.

    Retorna (imagem_retificada, marcadores) ou (None, marcadores) se os
    quatro cantos não estiverem visíveis.
    """
    markers = detect_markers(frame)
    if len(markers) < len(config.MARKER_IDS):
        return None, markers

    src = np.array([markers[i] for i in config.MARKER_IDS], dtype=np.float32)
    dst = np.array(
        [config.MARKER_CENTERS[i] for i in config.MARKER_IDS], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, (config.PAPER_W, config.PAPER_H))
    return warped, markers
