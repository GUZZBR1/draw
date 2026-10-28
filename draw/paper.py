"""Detecção da folha e correção de perspectiva (homografia).

A webcam vê a folha inclinada. O modo padrão encontra o contorno externo
da folha e usa os quatro cantos dela para montar a vista de cima. O modo
com marcadores usa os centros dos quatro ArUco da folha-modelo. Os dois
caminhos entregam a folha retificada no mesmo espaço definido em
config.py.
"""

import cv2
import numpy as np

from . import config


def _make_detector():
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(config.get_dictionary(), params)


_DETECTOR = _make_detector()


def _order_corners(points):
    """Ordena quatro pontos como TL, TR, BR e BL."""
    ordered = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = ordered.mean(axis=0)
    angles = np.arctan2(
        ordered[:, 1] - center[1], ordered[:, 0] - center[0]
    )
    ordered = ordered[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def _aspect_ratio(corners):
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    width = (top + bottom) / 2.0
    height = (left + right) / 2.0
    return width / max(height, 1.0)


def _sheet_candidate(contour, scale, small_frame_area):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    if cv2.contourArea(approx) < small_frame_area * 0.12:
        return None

    corners = _order_corners(approx.reshape(4, 2))
    if not 0.4 <= _aspect_ratio(corners) <= 2.5:
        return None
    return (corners / scale).astype(np.float32)


def _refine_corners(frame, corners):
    frame_h, frame_w = frame.shape[:2]
    if (
        np.any(corners < 11)
        or np.any(corners[:, 0] > frame_w - 12)
        or np.any(corners[:, 1] > frame_h - 12)
    ):
        return corners

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    refined = corners.reshape(-1, 1, 2).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.01,
    )
    cv2.cornerSubPix(gray, refined, (10, 10), (-1, -1), criteria)
    return refined.reshape(4, 2).astype(np.float32)


def detect_sheet(frame):
    """Encontra os cantos de uma folha sem marcadores.

    Retorna um array float32 com os cantos TL, TR, BR e BL. Retorna None
    quando nenhum quadrilátero grande e convexo aparece no frame.
    """
    if frame is None or frame.size == 0:
        return None

    frame_h, frame_w = frame.shape[:2]
    scale = min(1.0, 960.0 / max(frame_w, frame_h))
    if scale < 1.0:
        small = cv2.resize(
            frame,
            (int(frame_w * scale), int(frame_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = frame

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    small_frame_area = float(small.shape[0] * small.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        corners = _sheet_candidate(contour, scale, small_frame_area)
        if corners is not None:
            return _refine_corners(frame, corners)
    return None


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


def warp_sheet(frame, corners, destination=None):
    """Retifica a folha a partir de quatro cantos explícitos."""
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if destination is None:
        destination = [
            (0, 0),
            (config.PAPER_W, 0),
            (config.PAPER_W, config.PAPER_H),
            (0, config.PAPER_H),
        ]
    dst = np.asarray(destination, dtype=np.float32).reshape(4, 2)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        frame, matrix, (config.PAPER_W, config.PAPER_H)
    )


def warp_paper(frame):
    """Retifica a folha-modelo a partir dos marcadores.

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
    return warp_sheet(frame, src, dst), markers
