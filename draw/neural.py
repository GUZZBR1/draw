"""Detector neural da folha: segmentação por ONNX atrás do contrato
de draw.paper.detect_sheet_with_confidence.

O modelo da fase 2 (SegFormer-B0 exportado para ONNX) recebe o frame
esmagado para 512x512, normalizado com média e desvio do ImageNet, e
devolve logits de máscara em dois canais e um logit de presença. A
presença decide se há papel. A máscara decide onde: o quadrilátero da
folha sai do maior contorno da máscara limiarizada, com a mesma escada
de epsilon do detector clássico, um reajuste de cada aresta como reta
sobre os pontos do contorno e refino subpixel nos cantos dentro do
quadro.

O reajuste de arestas existe para o caso comum de uso: a página
cortada pela borda do quadro. Ali o vértice da aproximação mora sobre
a borda e não é o canto da página. A reta ajustada sobre os pontos
visíveis da aresta projeta o canto para fora do quadro, que é onde a
página continua. Pontos do contorno colados na borda do frame ficam
fora do ajuste: ali quem cortou a máscara foi o quadro, não a página.
Cada canto reajustado só é aceito se ficar a no máximo
_FIT_MAX_DISPLACE_PX do vértice da aproximação; sem essa trava, uma
aresta curta ou contaminada projeta cantos para longe demais.

A sessão ONNX é preguiçosa e o onnxruntime só é importado no primeiro
uso. Importar este módulo nunca quebra em ambiente sem a dependência;
usar o detector sem ela levanta um erro explícito.

O arquivo do modelo fica fora do repositório (binário de 15 MB). O
caminho chega por argumento ou pela variável de ambiente
DRAW_NEURAL_MODEL.
"""

import os
from pathlib import Path

import cv2
import numpy as np

from . import paper as _paper

_INPUT_SIZE = 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MASK_THRESHOLD = 0.5
_MIN_CONTOUR_AREA_RATIO = 0.01
_FIT_MAX_DISPLACE_PX = 100.0
_ENV_MODEL_VAR = "DRAW_NEURAL_MODEL"


def _load_session(model_path):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime nao esta instalado; o detector neural precisa "
            "dele. Instale com: pip install onnxruntime"
        ) from error
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"modelo ONNX nao encontrado: {path}. Passe o caminho do "
            f"arquivo .onnx ou defina a variavel {_ENV_MODEL_VAR}."
        )
    return ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )


def _order_corners(points):
    return _paper._order_corners(points)


def _largest_contour(mask_bin):
    contours, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _approx_quad(contour):
    perimeter = cv2.arcLength(contour, True)
    for epsilon in _paper._SHEET_EPSILON_LADDER:
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_corners(approx.reshape(4, 2))
    return None


def _fit_line(points):
    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
    ).ravel()
    direction = np.array([vx, vy], dtype=np.float64)
    point = np.array([x0, y0], dtype=np.float64)
    return direction, point


def _intersect(line_a, line_b):
    dir_a, point_a = line_a
    dir_b, point_b = line_b
    matrix = np.column_stack([dir_a, -dir_b])
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    t = np.linalg.solve(matrix, point_b - point_a)[0]
    return point_a + t * dir_a


def _segment_distances(points, start, end):
    segment = end - start
    t = np.clip(
        ((points - start) @ segment) / max(float(segment @ segment), 1e-9),
        0.0,
        1.0,
    )
    return np.linalg.norm(points - (start + np.outer(t, segment)), axis=1)


def _refit_corners(contour, quad, frame_shape):
    """Reajusta os cantos como interseções das retas das arestas.

    Cada ponto do contorno é atribuído à aresta mais próxima. Pontos
    colados nos vértices da aproximação ficam fora (são a virada de
    uma aresta para outra). Pontos sobre a borda do frame também
    ficam fora quando sobra ponto bastante sem eles: a máscara ali
    foi cortada pelo quadro. O ajuste é Huber com uma poda por
    resíduo. Um canto reajustado só substitui o vértice se ficar a
    no máximo _FIT_MAX_DISPLACE_PX dele, e o quadrilátero final
    precisa continuar convexo.
    """
    frame_h, frame_w = frame_shape
    points = contour.reshape(-1, 2).astype(np.float64)
    distances = np.stack(
        [
            _segment_distances(points, quad[i], quad[(i + 1) % 4])
            for i in range(4)
        ],
        axis=1,
    )
    assignment = np.argmin(distances, axis=1)
    lines = []
    for i in range(4):
        start = quad[i].astype(np.float64)
        end = quad[(i + 1) % 4].astype(np.float64)
        side = points[assignment == i]
        if len(side) < 8:
            return quad
        near_vertex = (np.linalg.norm(side - start, axis=1) < 25) | (
            np.linalg.norm(side - end, axis=1) < 25
        )
        side = side[~near_vertex]
        on_border = (
            (side[:, 0] <= 0.5)
            | (side[:, 0] >= frame_w - 1.5)
            | (side[:, 1] <= 0.5)
            | (side[:, 1] >= frame_h - 1.5)
        )
        off_border = side[~on_border]
        if len(off_border) >= 8:
            side = off_border
        if len(side) < 8:
            return quad
        line = _fit_line(side)
        direction, point = line
        normal = np.array([-direction[1], direction[0]])
        residuals = np.abs((side - point) @ normal)
        keep = residuals <= max(3.0, float(np.percentile(residuals, 70)))
        if keep.sum() >= 8:
            line = _fit_line(side[keep])
        lines.append(line)
    refined = quad.copy()
    for i in range(4):
        corner = _intersect(lines[i - 1], lines[i])
        if corner is None:
            continue
        if np.linalg.norm(corner - quad[i]) <= _FIT_MAX_DISPLACE_PX:
            refined[i] = corner
    refined = _order_corners(refined)
    if not cv2.isContourConvex(refined.reshape(4, 1, 2).astype(np.float32)):
        return quad
    return refined


def _refine_inside(frame, corners):
    """Refino subpixel só nos cantos que estão dentro do quadro.

    Canto projetado para fora do quadro (página cortada) não tem
    vizinhança de imagem para refinar; ele fica como a reta o pôs.
    Cantos dentro usam o mesmo acolchoamento do detector clássico.
    """
    frame_h, frame_w = frame.shape[:2]
    inside = [
        i
        for i, (x, y) in enumerate(corners)
        if 0.0 <= x <= frame_w - 1 and 0.0 <= y <= frame_h - 1
    ]
    if not inside:
        return corners
    pad = 12
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(
        gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE
    )
    selected = (corners[inside] + pad).reshape(-1, 1, 2).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.01,
    )
    cv2.cornerSubPix(padded, selected, (10, 10), (-1, -1), criteria)
    refined = corners.copy()
    refined[inside] = selected.reshape(-1, 2) - pad
    return refined.astype(np.float32)


def fit_quad(mask_bin, frame=None):
    """Ajusta o quadrilátero da folha sobre a máscara binária.

    Retorna float32[4,2] TL/TR/BR/BL ou None quando a máscara não
    sustenta um quadrilátero. Com o frame, os cantos dentro do quadro
    ganham refino subpixel sobre a imagem.
    """
    contour = _largest_contour(mask_bin)
    if contour is None:
        return None
    frame_area = float(mask_bin.shape[0] * mask_bin.shape[1])
    if cv2.contourArea(contour) < frame_area * _MIN_CONTOUR_AREA_RATIO:
        return None
    quad = _approx_quad(contour)
    if quad is None:
        return None
    quad = _refit_corners(contour, quad, mask_bin.shape[:2])
    if frame is not None:
        quad = _refine_inside(frame, quad)
    return quad.astype(np.float32)


class NeuralSheetDetector:
    """Detector neural com sessão própria.

    __call__(frame) devolve (cantos, veredito, máscara): o mesmo trio
    da interface dos detectores do harness de avaliação. O veredito é
    "paper" quando o logit de presença é positivo e o quadrilátero
    fecha; fora isso, (None, None, máscara). A máscara é uint8
    0/255 no tamanho do frame, limiarizada em 0,5 como na avaliação
    da fase 2.
    """

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self._session = None

    def _get_session(self):
        if self._session is None:
            self._session = _load_session(self.model_path)
        return self._session

    def load(self):
        """Carrega a sessão agora, para erro de modelo falhar cedo."""
        self._get_session()

    def predict(self, frame):
        """Roda o modelo. Retorna (logit de presença, máscara 0/255)."""
        session = self._get_session()
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (_INPUT_SIZE, _INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        tensor = ((resized / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)
        tensor = tensor[None].astype(np.float32)
        logits, presence = session.run(
            ["mask_logits", "presence_logit"], {"image": tensor}
        )
        channel_bg, channel_paper = logits[0, 0], logits[0, 1]
        peak = np.maximum(channel_bg, channel_paper)
        exp_bg = np.exp(channel_bg - peak)
        exp_paper = np.exp(channel_paper - peak)
        probability = exp_paper / (exp_bg + exp_paper)
        mask_small = (probability > _MASK_THRESHOLD).astype(np.uint8)
        mask = cv2.resize(
            mask_small, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
        )
        return float(presence[0]), mask * 255

    def __call__(self, frame):
        if frame is None or frame.size == 0:
            return None, None, None
        presence, mask = self.predict(frame)
        if presence <= 0:
            return None, None, mask
        corners = fit_quad(mask, frame)
        if corners is None:
            return None, None, mask
        return corners, "paper", mask

    def detect_sheet_with_confidence(self, frame):
        """O par (cantos, veredito) do contrato de draw.paper."""
        corners, verdict, _ = self(frame)
        return corners, verdict


_DEFAULT_DETECTOR = None


def _default_detector(model_path=None):
    global _DEFAULT_DETECTOR
    path = model_path or os.environ.get(_ENV_MODEL_VAR)
    if not path:
        raise RuntimeError(
            "caminho do modelo nao informado: passe model_path ou "
            f"defina a variavel {_ENV_MODEL_VAR} com o caminho do .onnx."
        )
    if _DEFAULT_DETECTOR is None or _DEFAULT_DETECTOR.model_path != str(path):
        _DEFAULT_DETECTOR = NeuralSheetDetector(path)
    return _DEFAULT_DETECTOR


def detect_sheet_with_confidence(frame, model_path=None):
    """Encontra os cantos da folha com o modelo neural.

    Mesmo contrato de draw.paper.detect_sheet_with_confidence:
    (cantos, "paper") quando o modelo afirma papel e o quadrilátero
    fecha; (None, None) quando não. O caminho do modelo vem do
    argumento ou da variável DRAW_NEURAL_MODEL.
    """
    return _default_detector(model_path).detect_sheet_with_confidence(frame)
