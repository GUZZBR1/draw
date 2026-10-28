"""Detecção da folha e correção de perspectiva (homografia).

A webcam vê a folha inclinada. O modo padrão procura o contorno da folha
na árvore de contornos da cena e usa os quatro cantos dela para montar a
vista de cima. Um portão de textura examina o interior do candidato
antes de aceitar a folha: formato sozinho não basta. O modo com
marcadores usa os centros dos quatro ArUco da folha-modelo. Os dois
caminhos entregam a folha retificada no mesmo
espaço definido em config.py.
"""

import cv2
import numpy as np

from . import config


# A folha nem sempre fecha como quadrilátero num epsilon só. Cantos
# arredondados e a lombada de um caderno pedem aproximações mais grossas.
_SHEET_EPSILON_LADDER = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
_SHEET_MAX_CANDIDATES = 8
_SHEET_MIN_AREA_RATIO = 0.12

# Portão de textura: a geometria sozinha aceitava qualquer quadrilátero
# grande. Na auditoria de textura, 60% das cenas sem papel ganharam uma
# folha. O portão mede o interior do candidato antes de aceitar: fração
# de branco, densidade de bordas internas e variância do Laplaciano.
#
# A regra original da auditoria foi calibrada em papel pautado, onde a
# densidade de bordas internas passa de 0,03 com folga. A folha com
# tinta esparsa, caso principal do produto e das cenas deste
# repositório, mede densidade de bordas perto de 0,006, abaixo desse
# limiar, mas variância do Laplaciano entre 523 e 783. Por isso o ramo
# extra: Laplaciano a partir de 200 também afirma papel. O pior falso
# positivo medido na auditoria ficou em 27,0, longe dessa faixa.
#
# Os limiares abaixo são calibrações de laboratório, feitas sobre cenas
# sintéticas e um frame real. Eles precisam ser refeitos com as
# capturas reais da mesa do usuário.
_TEXTURE_MIN_WHITE_FRACTION = 0.5
_TEXTURE_MIN_EDGE_DENSITY = 0.03
_TEXTURE_MIN_LAPLACIAN_VARIANCE = 30.0
_TEXTURE_HIGH_LAPLACIAN_VARIANCE = 200.0
_TEXTURE_INTERIOR_SHRINK = 0.86


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
    for epsilon in _SHEET_EPSILON_LADDER:
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if cv2.contourArea(approx) < small_frame_area * _SHEET_MIN_AREA_RATIO:
            continue

        corners = _order_corners(approx.reshape(4, 2))
        if not 0.4 <= _aspect_ratio(corners) <= 2.5:
            continue
        return (corners / scale).astype(np.float32)
    return None


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


def _texture_features(frame, corners):
    """Mede o interior de um quadrilátero candidato.

    Retorna (fração de branco, densidade de bordas, variância do
    Laplaciano). O interior é encolhido para 86% em torno do centroide
    e erodido com núcleo 9x9, para a borda do candidato não contaminar
    as medidas. Branco é saturação abaixo de 60 e valor acima de 180
    no HSV. A variância é a do Laplaciano CV_32F sobre o cinza. As
    bordas internas são Canny 30/100 sobre o cinza, contadas só dentro
    do interior encolhido.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    center = corners.mean(axis=0)
    shrunk = center + (corners - center) * _TEXTURE_INTERIOR_SHRINK
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [shrunk.astype(np.int32)], 255)
    eroded = cv2.erode(mask, np.ones((9, 9), np.uint8))
    selected = eroded > 0

    saturation = hsv[:, :, 1][selected]
    value = hsv[:, :, 2][selected]
    white_fraction = float(np.mean((saturation < 60) & (value > 180)))
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_variance = float(np.var(laplacian[selected]))
    edges = cv2.Canny(gray, 30, 100)
    edge_density = float(np.mean(edges[selected] > 0))
    return white_fraction, edge_density, laplacian_variance


def _texture_verdict(frame, corners):
    """Veredito do portão de textura para um candidato.

    "paper" quando o interior parece papel. "blank_maybe" quando o
    candidato é branco o bastante, mas liso demais para afirmar papel.
    None quando nem branco o bastante ele é.
    """
    white_fraction, edge_density, laplacian_variance = _texture_features(
        frame, corners
    )
    if white_fraction < _TEXTURE_MIN_WHITE_FRACTION:
        return None
    textured = laplacian_variance >= _TEXTURE_MIN_LAPLACIAN_VARIANCE and (
        edge_density >= _TEXTURE_MIN_EDGE_DENSITY
        or laplacian_variance >= _TEXTURE_HIGH_LAPLACIAN_VARIANCE
    )
    return "paper" if textured else "blank_maybe"


def detect_sheet(frame):
    """Encontra os cantos de uma folha sem marcadores.

    Retorna um array float32 com os cantos TL, TR, BR e BL. Retorna None
    quando nenhum quadrilátero grande e convexo passa no portão de
    textura como papel. Candidato branco e liso demais não conta como
    folha aqui: ele aparece em detect_sheet_with_confidence com o
    veredito blank_maybe.
    """
    corners, verdict = detect_sheet_with_confidence(frame)
    return corners if verdict == "paper" else None


def detect_sheet_with_confidence(frame):
    """Encontra os cantos de uma folha sem marcadores, com veredito.

    Retorna (cantos, veredito). O veredito é "paper" quando o interior
    do candidato passa no portão de textura. É "blank_maybe" quando um
    candidato branco o bastante aparece, mas liso demais para afirmar
    papel. É None, com cantos None, quando não há candidato.
    """
    if frame is None or frame.size == 0:
        return None, None

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
    # RETR_TREE: o contorno da folha pode ser interno na hierarquia, como
    # a volta de um caderno dentro de um borrão maior fundido na cena.
    contours, _ = cv2.findContours(
        closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    small_frame_area = float(small.shape[0] * small.shape[1])
    min_area = small_frame_area * _SHEET_MIN_AREA_RATIO
    candidates = [
        contour
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)
        if cv2.contourArea(contour) >= min_area
    ][:_SHEET_MAX_CANDIDATES]
    # Um candidato que falha no portão não encerra a busca: um contorno
    # seguinte ainda pode ser a folha.
    blank_corners = None
    for contour in candidates:
        corners = _sheet_candidate(contour, scale, small_frame_area)
        if corners is None:
            continue
        verdict = _texture_verdict(frame, corners)
        if verdict == "paper":
            return _refine_corners(frame, corners), "paper"
        if verdict == "blank_maybe" and blank_corners is None:
            blank_corners = corners
    if blank_corners is not None:
        return _refine_corners(frame, blank_corners), "blank_maybe"
    return None, None


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
