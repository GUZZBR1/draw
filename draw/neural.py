"""Detector neural da folha: segmentação por ONNX atrás do contrato
de draw.paper.detect_sheet_with_confidence.

O modelo da fase 2 (SegFormer-B0 exportado para ONNX) recebe o frame
esmagado para a resolução de entrada, normalizado com média e desvio
do ImageNet, e devolve logits de máscara em dois canais e um logit
de presença. A resolução segue o arquivo: um ONNX de formato fixo
impõe o tamanho dele (512x512 no export original); um ONNX dinâmico
(eixos H/W simbólicos) roda a 384x384, o ponto medido no Ryzen 7
7700X do Lucas em 2026-10-29 — session.run de 122 para 58 ms, IoU de
máscara de 0,984 contra o 512, nenhum veredito de presença trocado
nos frames dele fora as três frames da caixa branca, que a 384
deixam de ser falso positivo. A sessão roda com 4 threads intra-op,
também medido: o padrão do ONNX Runtime marcava 248 ms no mesmo
tensor, 4 threads marcam 122 ms e 16 threads pioram para 210 ms
(draw-lab/perf-ryzen-2026-10-29/REPORT.md). A
presença decide se há papel. A máscara decide onde: o quadrilátero da
folha sai do maior contorno da máscara limiarizada, com a mesma escada
de epsilon do detector clássico, um reajuste de cada aresta como reta
sobre os pontos do contorno e refino subpixel nos cantos dentro do
quadro.

Quando a escada de epsilon não fecha sobre o contorno, o
quadrilátero é o retângulo de área mínima do contorno, sem reajuste
nem refino: medido nos frames reais em que a escada falha, os dois
passos degradam essa reserva (os números ficam em _min_area_quad). A
reserva só existe onde antes o retorno era nenhum quadrilátero;
onde a escada fecha, o caminho é o mesmo de sempre.

O reajuste de arestas existe para o caso comum de uso: a página
cortada pela borda do quadro. Ali o vértice da aproximação mora sobre
a borda e não é o canto da página. A reta ajustada sobre os pontos
visíveis da aresta projeta o canto para fora do quadro, que é onde a
página continua. Pontos do contorno colados na borda do frame ficam
fora do ajuste: ali quem cortou a máscara foi o quadro, não a página.
Cada canto reajustado só é aceito se ficar a no máximo
_FIT_MAX_DISPLACE_PX do vértice da aproximação; sem essa trava, uma
aresta curta ou contaminada projeta cantos para longe demais.

Aresta cuja cobertura de pontos é curta (a mão cobre a maior parte
dela) não ganha reta ajustada: dentro de um quadro só, os pontos
visíveis dela são a borda do braço, não a aresta, e a reta do
fragmento projeta os cantos a centenas de pixels (o teto acima veta
a correção e o canto congela no vértice da aproximação, que nesses
quadros mora sobre a borda do quadro). Quem resolve é a memória de
arestas do detector: a aresta coberta herda a última reta confiável
dela mesma, e só sem herança vale a reta travada na direção do
segmento. Os cantos vizinhos de uma aresta travada ou herdada não
passam pelo teto: a projeção já nasce limitada.

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

_INPUT_SIZE = 384
_ORT_INTRA_THREADS = 4
_ENV_THREADS_VAR = "DRAW_ORT_THREADS"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MASK_THRESHOLD = 0.5
_MIN_CONTOUR_AREA_RATIO = 0.01
_FIT_MAX_DISPLACE_PX = 100.0
_FIT_MIN_EDGE_COVERAGE = 0.5
_MEMORY_MIN_MASK_IOU = 0.7
_MEMORY_MASK_STRIDE = 4
_ENV_MODEL_VAR = "DRAW_NEURAL_MODEL"

# Portão do caminho clássico (medido em 2026-10-30 no laboratório,
# draw-lab/fp-gate-2026-10-30/): com um modelo neural carregado, um
# logit de presença até este valor é um "não há papel" forte o bastante
# para o clássico não poder declarar folha por cima dele (o incidente
# da caixa branca: a v2 media logit ≈ −10,9 na caixa e o banner vinha
# do clássico). Nos frames reais medidos (44 positivos por modelo: os
# 43 do frames-dataset e o frame_real), o menor logit de positivo foi
# +7,38 (v3 dinâmico); nos 6 negativos do dataset, o maior logit foi
# −6,98 (v3). O limiar fica 10,4 abaixo do positivo mais fraco e 4,0
# acima do negativo mais forte: veta o clássico em 18/18 corridas
# (3 modelos × 6 negativos) sobre os negativos do dataset e não custa
# nenhum positivo medido. Ele NÃO cobre a caixa branca na v3 a 512
# (logit +0,41 a +0,98: a própria neural afirma papel, fraco, e o
# quadrilátero fecha sobre ~3,5% de máscara) nem os logits −0,006 a
# −0,382 da v3 dinâmica na caixa: nenhum dos dois é "fortemente
# negativo", e baixar o limiar para perto de zero desligaria a reserva
# clássica justamente no cenário de generalização, a folha real que a
# neural subdispara num setup novo.
PRESENCE_VETO_LOGIT = -3.0

# Guarda de plausibilidade do ajuste (medida em 2026-10-30 no
# laboratório, draw-lab/noite-2026-10-30/block6/): o ajuste neural cru
# é biestável sob oclusão. Nas 1.132 detecções gravadas dos dois
# vídeos de replay, uma família de ajustes colapsa uma aresta (a
# direita com 213–269 px contra a esquerda com 980–1.025 px: razão
# entre arestas opostas de 3,81 a 4,67), enquanto tudo o que o app
# exibiu em trechos quietos fica em 2,42 ou menos e o máximo do outro
# vídeo é 3,00. A guarda recusa o quadrilátero cuja maior razão entre
# arestas opostas passa de _FIT_OPPOSITE_RATIO_MAX. Recusa também o
# que extrapola um canto para além de _FIT_EXCURSION_DIAG_FRAC da
# diagonal do quadro: nas gravações a folha sai do recorte de tela de
# verdade até 9,9% da diagonal, mas o modo fora do quadro medido no
# frame_real.png (1289x710) chegou a 21,4%. Ajuste recusado vale
# "sem quadrilátero" e não alimenta a memória de arestas: quem chamou
# decide (no app, o rastreador segura o quadrilátero anterior). Nas
# gravações, a guarda recusa 26 de 1.132 ajustes (25 da família
# colapsada de um vídeo, 1 do outro). Ressalva: as gravações são de
# tela, com overlay queimado; a guarda precisa de reconfirmação em
# captura crua.
_FIT_OPPOSITE_RATIO_MAX = 3.0
_FIT_EXCURSION_DIAG_FRAC = 0.15


def _fit_plausible(corners, frame_shape):
    """True se o quadrilátero passa na guarda de plausibilidade."""
    quad = np.asarray(corners, dtype=np.float64)
    if quad.shape != (4, 2):
        return False
    edges = np.array(
        [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    )
    if edges.min() <= 0:
        return False
    for a, b in ((0, 2), (3, 1)):
        ratio = max(edges[a], edges[b]) / min(edges[a], edges[b])
        if ratio > _FIT_OPPOSITE_RATIO_MAX:
            return False
    height, width = frame_shape[:2]
    margin = _FIT_EXCURSION_DIAG_FRAC * float(np.hypot(width, height))
    return not (
        quad[:, 0].min() < -margin
        or quad[:, 0].max() > width + margin
        or quad[:, 1].min() < -margin
        or quad[:, 1].max() > height + margin
    )


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
    options = ort.SessionOptions()
    options.intra_op_num_threads = _ort_intra_threads()
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _ort_intra_threads():
    """Threads intra-op da sessão: 4, medido no Ryzen do Lucas.

    A variável DRAW_ORT_THREADS troca o valor sem tocar no código,
    para medir em outra máquina; valor inválido ou não positivo
    volta ao medido.
    """
    raw = os.environ.get(_ENV_THREADS_VAR)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return _ORT_INTRA_THREADS


def _session_input_size(session):
    """Resolução de entrada: a do arquivo, se fixa; senão _INPUT_SIZE."""
    shape = session.get_inputs()[0].shape
    if len(shape) > 2 and isinstance(shape[2], int) and shape[2] > 0:
        return shape[2]
    return _INPUT_SIZE


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


def _min_area_quad(contour):
    """Reserva do ajuste: o retângulo de área mínima do contorno.

    Só é chamado quando a escada de epsilon não fecha. Os cantos
    saem na mesma convenção TL/TR/BR/BL do caminho principal, e o
    retângulo volta cru, sem o reajuste de arestas e sem o refino
    subpixel. Medido: nos frames 24 e 26 das capturas reais, os
    únicos em que a escada falha com o modelo v2, o IoU do
    quadrilátero preenchido do frame 24 contra a máscara anotada
    dele é 0,895 com o retângulo cru, 0,850 com o reajuste, 0,843
    com reajuste e refino e 0,887 só com o refino. O reajuste foi
    desenhado para os vértices da aproximação, que moram perto dos
    cantos da folha; os vértices do retângulo mínimo moram fora
    dela, e as retas ajustadas passam a seguir a borda visível da
    máscara, que nesses frames é oclusão, não a aresta da página.
    """
    return _order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))


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


def _edge_line(side, start, end, centroid, fallback=None):
    """A reta de uma aresta: ajustada, herdada ou travada na direção.

    Com cobertura boa (os pontos ocupam pelo menos
    _FIT_MIN_EDGE_COVERAGE do comprimento do segmento), vale o
    ajuste de Huber com poda por resíduo de sempre, e a reta é
    confiável. Com cobertura curta, os pontos são um fragmento da
    aresta (o resto está coberto, tipicamente pela mão) e nem a
    direção ajustada nem o fragmento localizam a aresta: medido nos
    quadros de escape do vídeo de teste, os pontos do fragmento são
    a borda do braço sobre a folha, a dezenas de pixels da aresta
    verdadeira, e a reta ajustada projeta os cantos a centenas de
    pixels; o teto de deslocamento veta a correção e o canto fica
    congelado no vértice da aproximação, que nesses quadros mora
    sobre a borda do quadro porque a máscara corre por ela.

    Sem cobertura, a ordem de preferência é: a reta herdada da
    chamada anterior (a folha é rígida e a câmera é fixa; a última
    reta confiável da mesma aresta é a melhor estimativa dela
    coberta); na falta dela, a reta travada na direção do segmento
    com o deslocamento mediano dos pontos, cuja projeção nasce
    limitada. Retorna (reta, travada, confiavel).
    """
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length < 1e-6:
        return _fit_line(side), False, True
    direction = segment / length
    along = (side - start) @ direction
    coverage = float(along.max() - along.min()) / length
    if coverage >= _FIT_MIN_EDGE_COVERAGE:
        line = _fit_line(side)
        dir_fit, point_fit = line
        normal = np.array([-dir_fit[1], dir_fit[0]])
        residuals = np.abs((side - point_fit) @ normal)
        keep = residuals <= max(3.0, float(np.percentile(residuals, 70)))
        if keep.sum() >= 8:
            line = _fit_line(side[keep])
        return line, False, True
    if fallback is not None:
        return fallback, True, False
    normal = np.array([-direction[1], direction[0]])
    if float((centroid - start) @ normal) < 0:
        normal = -normal
    offset = float(np.median((side - start) @ normal))
    return (direction, start + offset * normal), True, False


def _refit_corners(contour, quad, frame_shape, fallback_lines=None):
    """Reajusta os cantos como interseções das retas das arestas.

    Cada ponto do contorno é atribuído à aresta mais próxima. Pontos
    colados nos vértices da aproximação ficam fora (são a virada de
    uma aresta para outra). Pontos sobre a borda do frame também
    ficam fora quando sobra ponto bastante sem eles: a máscara ali
    foi cortada pelo quadro. A reta de cada aresta sai de _edge_line
    (ajustada; sem cobertura, herdada de fallback_lines ou travada
    na direção do segmento). Um canto reajustado só substitui o
    vértice se ficar a no máximo _FIT_MAX_DISPLACE_PX dele, exceto
    quando uma das arestas do canto é travada: ali a projeção já
    nasce limitada pela trava e o vértice de referência é justamente
    o contaminado. O quadrilátero final precisa continuar convexo.

    Retorna (quadrilatero, retas usadas, confiavel por aresta); as
    retas voltam None quando alguma aresta tem ponto de menos e o
    reajuste inteiro é abandonado.
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
    centroid = quad.astype(np.float64).mean(axis=0)
    lines = []
    locked = []
    trusted = []
    for i in range(4):
        start = quad[i].astype(np.float64)
        end = quad[(i + 1) % 4].astype(np.float64)
        side = points[assignment == i]
        if len(side) < 8:
            return quad, None, None
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
            return quad, None, None
        fallback = None
        if fallback_lines is not None:
            fallback = fallback_lines[i]
        line, is_locked, is_trusted = _edge_line(
            side, start, end, centroid, fallback
        )
        lines.append(line)
        locked.append(is_locked)
        trusted.append(is_trusted)
    refined = quad.copy()
    for i in range(4):
        corner = _intersect(lines[i - 1], lines[i])
        if corner is None:
            continue
        if locked[i - 1] or locked[i]:
            refined[i] = corner
        elif np.linalg.norm(corner - quad[i]) <= _FIT_MAX_DISPLACE_PX:
            refined[i] = corner
    refined = _order_corners(refined)
    if not cv2.isContourConvex(refined.reshape(4, 1, 2).astype(np.float32)):
        return quad, lines, trusted
    return refined, lines, trusted


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


def _fit_quad_impl(mask_bin, frame, fallback_lines):
    """O ajuste completo sobre a máscara, com herança opcional.

    Retorna (quadrilatero ou None, retas usadas ou None, confiavel
    por aresta ou None). fallback_lines são as retas confiáveis da
    chamada anterior, oferecidas às arestas sem cobertura.
    """
    contour = _largest_contour(mask_bin)
    if contour is None:
        return None, None, None
    frame_area = float(mask_bin.shape[0] * mask_bin.shape[1])
    if cv2.contourArea(contour) < frame_area * _MIN_CONTOUR_AREA_RATIO:
        return None, None, None
    quad = _approx_quad(contour)
    if quad is None:
        return _min_area_quad(contour).astype(np.float32), None, None
    quad, lines, trusted = _refit_corners(
        contour, quad, mask_bin.shape[:2], fallback_lines
    )
    if frame is not None:
        quad = _refine_inside(frame, quad)
    return quad.astype(np.float32), lines, trusted


def fit_quad(mask_bin, frame=None):
    """Ajusta o quadrilátero da folha sobre a máscara binária.

    Retorna float32[4,2] TL/TR/BR/BL ou None quando a máscara não
    tem contorno ou é pequena demais para sustentar um quadrilátero.
    Se a escada de epsilon não fecha, vale a reserva de
    _min_area_quad, sem reajuste nem refino. Com o frame, os cantos
    do caminho principal que estão dentro do quadro ganham refino
    subpixel sobre a imagem. Esta função não tem estado: a memória
    de arestas entre chamadas mora no NeuralSheetDetector.
    """
    quad, _lines, _trusted = _fit_quad_impl(mask_bin, frame, None)
    return quad


class NeuralSheetDetector:
    """Detector neural com sessão própria.

    __call__(frame) devolve (cantos, veredito, máscara): o mesmo trio
    da interface dos detectores do harness de avaliação. O veredito é
    "paper" quando o logit de presença é positivo e o quadrilátero
    fecha; fora isso, (None, None, máscara). A máscara é uint8
    0/255 no tamanho do frame, limiarizada em 0,5 como na avaliação
    da fase 2.

    O detector guarda as retas confiáveis das arestas entre chamadas
    (a memória de arestas): quando uma aresta perde cobertura numa
    chamada (a mão sobre ela), a reta herdada estima essa aresta até
    ela voltar a ser visível. A herança só vale em cena contínua: a
    máscara da chamada precisa ter IoU de pelo menos
    _MEMORY_MIN_MASK_IOU com a da chamada anterior, medido nas
    máscaras reduzidas por _MEMORY_MASK_STRIDE. Cena nova recomeça
    sem herança.

    O atributo last_presence_logit guarda o logit de presença da
    última chamada a predict (None antes da primeira e após um frame
    vazio): é ele que o portão PRESENCE_VETO_LOGIT lê no app para
    decidir se a reserva clássica pode falar.
    """

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self._session = None
        self._resolved_input_size = None
        self._edge_lines = None
        self._memory_mask = None
        self.last_presence_logit = None

    def _get_session(self):
        if self._session is None:
            self._session = _load_session(self.model_path)
        return self._session

    def load(self):
        """Carrega a sessão agora, para erro de modelo falhar cedo."""
        self._get_session()

    def _fallback_lines(self, mask):
        """As retas herdáveis para esta máscara, ou None sem continuidade."""
        if self._edge_lines is None or self._memory_mask is None:
            return None
        small = mask[::_MEMORY_MASK_STRIDE, ::_MEMORY_MASK_STRIDE] > 0
        if small.shape != self._memory_mask.shape:
            return None
        inter = int((small & self._memory_mask).sum())
        union = int((small | self._memory_mask).sum())
        if union == 0 or inter / union < _MEMORY_MIN_MASK_IOU:
            return None
        return self._edge_lines

    def _fit(self, mask, frame):
        """fit_quad com a memória de arestas do detector.

        A guarda de plausibilidade (_fit_plausible) fecha o caminho:
        quadrilátero recusado devolve None e não alimenta a memória
        de arestas, para a geometria selvagem não virar herança.
        """
        fallback = self._fallback_lines(mask)
        corners, lines, trusted = _fit_quad_impl(mask, frame, fallback)
        if corners is not None and not _fit_plausible(
            corners, frame.shape
        ):
            return None
        if lines is not None:
            stored = (
                list(self._edge_lines)
                if self._edge_lines is not None
                else [None, None, None, None]
            )
            for i in range(4):
                if trusted[i]:
                    stored[i] = lines[i]
            self._edge_lines = stored
        self._memory_mask = (
            mask[::_MEMORY_MASK_STRIDE, ::_MEMORY_MASK_STRIDE] > 0
        )
        return corners

    def predict(self, frame):
        """Roda o modelo. Retorna (logit de presença, máscara 0/255)."""
        session = self._get_session()
        if self._resolved_input_size is None:
            self._resolved_input_size = _session_input_size(session)
        size = self._resolved_input_size
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (size, size),
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
        self.last_presence_logit = float(presence[0])
        return self.last_presence_logit, mask * 255

    def __call__(self, frame):
        if frame is None or frame.size == 0:
            self.last_presence_logit = None
            return None, None, None
        presence, mask = self.predict(frame)
        if presence <= 0:
            return None, None, mask
        corners = self._fit(mask, frame)
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
