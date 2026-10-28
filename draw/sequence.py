"""Rastreamento temporal de traços (etapa 5 do Senado).

Definição honesta, antes de qualquer código
-------------------------------------------
A "ordem" registrada aqui é a **ordem de primeiro aparecimento
estável** de cada traço durante uma sessão ao vivo, não a ordem
em que a caneta baixou e levantou. Caneta levantada não é
observável em tinta persistente: a máscara de tinta só diz o que
está no papel agora, não quando cada gesto começou ou terminou.
Da mesma forma, nenhuma ordem pode ser inferida de uma foto
pronta — a máscara final é a mesma qualquer que seja a ordem em
que os traços foram desenhados. Quem precisa de ordem tem que
observar o desenho acontecendo, e é isso que este módulo faz.

Regras do rastreador:

- Só rastreia com a folha travada. Se os cantos saltarem além de
  ``SHEET_JUMP_LIMIT_PX`` (o mesmo limite de 40 px já usado pelo
  loop do app), a sessão marca uma quebra e nenhum traço é casado
  através do salto: o que voltar depois do salto é traço novo.
  Se a folha não estiver travada (``sheet_locked=False``), a
  chamada é ignorada — não cria traço, não casa traço.
- Um traço novo só entra na ordem definitiva depois de persistir
  ``stable_after`` amostras (padrão 12). Flicker de 1 amostra
  nunca vira traço, e traço que some antes de estabilizar é
  descartado em silêncio (nunca chegou a existir na ordem).
- Traço casado mantém ``id`` e posição na ordem; sua geometria é
  atualizada para a versão mais recente do traço (mais limpa do
  que ir apendando pixels ruidosos).
- Traço estável que some é marcado ``erased`` e mantém sua
  posição na ordem. Se a tinta reaparecer depois, é um novo ato
  de desenhar na prática e ganha nova posição na ordem — o ``id``
  antigo nunca é ressuscitado em silêncio.
- Cruzamentos: cada braço de um X continua sendo um segmento
  próprio, como na extração estática. A ordem registrada é a
  ordem de aparecimento de cada braço, não "o X foi um gesto só".
  Adivinhar o gesto seria inventar dado que não existe.
- Taxa: o rastreador não decide a taxa de amostragem; quem chama
  (o app) alimenta uma amostra a cada N quadros retificados.
  ``frame_index`` é o índice da amostra, na contagem de quem
  chama, e ``timestamp_ms`` é opcional.

Este módulo é puro estado e geometria: não abre câmera, não
abre janela, não toca na detecção, na tinta nem na extração
estática. O snapshot atual (tecla ``v``, ordem por centroide)
continua intacto e separado; a sequência só existe em modo
sessão explícito.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import export as export_module
from . import strokes as strokes_module

#: Salto de cantos acima disso quebra a sessão (mesmo limite do app).
SHEET_JUMP_LIMIT_PX = 40.0

#: Amostras seguidas para um traço virar definitivo.
DEFAULT_STABLE_AFTER = 12

#: Distância máxima de centroide para casar dois traços, em px
#: do espaço da folha retificada.
MATCH_CENTROID_PX = 60.0

#: Distância máxima entre pontas para reforçar um casamento
#: quando o centroide se moveu muito (traço que cresce).
MATCH_ENDPOINT_PX = 80.0


def _centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _min_endpoint_distance(points_a, points_b):
    ends_a = (points_a[0], points_a[-1])
    ends_b = (points_b[0], points_b[-1])
    return min(
        math.hypot(ax - bx, ay - by)
        for ax, ay in ends_a
        for bx, by in ends_b
    )


@dataclass
class TrackedStroke:
    """Um traço acompanhado ao longo da sessão."""

    id: int
    points: list
    closed: bool = False
    first_seen_frame: int = 0
    first_seen_ms: object = None
    last_seen_frame: int = 0
    last_seen_ms: object = None
    seen_count: int = 0
    order: object = None
    status: str = "pending"  # pending | stable | erased

    @property
    def centroid(self):
        return _centroid(self.points) if self.points else (0.0, 0.0)


class SequenceTracker:
    """Acompanha traços frame a frame e registra a ordem."""

    def __init__(self, stable_after=DEFAULT_STABLE_AFTER,
                 match_centroid_px=MATCH_CENTROID_PX,
                 match_endpoint_px=MATCH_ENDPOINT_PX):
        if stable_after < 1:
            raise ValueError("stable_after deve ser >= 1")
        self.stable_after = stable_after
        self.match_centroid_px = match_centroid_px
        self.match_endpoint_px = match_endpoint_px
        self._active = {}  # id -> TrackedStroke ainda presente
        self._ordered = []  # todos os traços que estabilizaram
        self._next_id = 1
        self._next_order = 0
        self._last_corners = None
        self.breaks = 0
        self.samples = 0

    @property
    def ordered(self):
        """Traços estáveis (e apagados) em ordem de aparecimento."""
        return list(self._ordered)

    def final_strokes(self):
        """Traços presentes no fim da sessão, em ordem.

        Exclui os marcados ``erased``: é o desenho final, não o
        histórico. Para o histórico completo, use ``ordered``.
        """
        return [t for t in self._ordered if t.status == "stable"]

    def _corner_jump(self, corners):
        if corners is None or self._last_corners is None:
            return False
        try:
            dist = max(
                math.hypot(float(a[0]) - float(b[0]),
                           float(a[1]) - float(b[1]))
                for a, b in zip(self._last_corners, corners)
            )
        except (TypeError, ValueError, IndexError):
            return False
        return dist > SHEET_JUMP_LIMIT_PX

    def update(self, strokes, frame_index, timestamp_ms=None,
               corners=None, sheet_locked=True):
        """Alimenta uma amostra e devolve os traços ativos.

        ``strokes`` é a lista de ``Stroke`` extraídos da máscara
        retificada desta amostra. Sem folha travada, a amostra é
        ignorada. Com salto de folha, os traços ativos deixam de
        ser candidatos a casamento e a sessão conta uma quebra.
        """
        if not sheet_locked:
            return list(self._active.values())
        if self._corner_jump(corners):
            self.breaks += 1
            self._active = {}
        if corners is not None:
            self._last_corners = corners
        self.samples += 1

        candidates = list(strokes or [])
        matched_ids = set()
        used = set()
        # Casamento guloso pelo centroide mais próximo, com
        # reforço por pontas quando o traço cresceu.
        pairs = []
        for tracked in self._active.values():
            for i, stroke in enumerate(candidates):
                if i in used or not stroke.points:
                    continue
                centroid_dist = math.hypot(
                    tracked.centroid[0] - _centroid(stroke.points)[0],
                    tracked.centroid[1] - _centroid(stroke.points)[1],
                )
                endpoint_dist = _min_endpoint_distance(
                    tracked.points, stroke.points
                )
                if (centroid_dist <= self.match_centroid_px
                        or endpoint_dist <= self.match_endpoint_px
                        and centroid_dist <= 2 * self.match_centroid_px):
                    pairs.append((centroid_dist, tracked.id, i))
        pairs.sort()
        match_by_index = {}
        for _dist, tracked_id, i in pairs:
            if tracked_id in matched_ids or i in used:
                continue
            matched_ids.add(tracked_id)
            used.add(i)
            match_by_index[i] = tracked_id

        # Traços ativos não casados: estáveis viram erased e
        # mantêm a posição na ordem; pendentes nunca entraram na
        # ordem e são simplesmente descartados.
        for tracked_id, tracked in list(self._active.items()):
            if tracked_id not in matched_ids:
                if tracked.status == "stable":
                    tracked.status = "erased"
                del self._active[tracked_id]

        newly_stable = []
        for i, stroke in enumerate(candidates):
            if not stroke.points:
                continue
            tracked_id = match_by_index.get(i)
            if tracked_id is not None:
                tracked = self._active[tracked_id]
                tracked.points = list(stroke.points)
                tracked.closed = bool(stroke.closed)
                tracked.last_seen_frame = frame_index
                tracked.last_seen_ms = timestamp_ms
                tracked.seen_count += 1
            else:
                tracked = TrackedStroke(
                    id=self._next_id,
                    points=list(stroke.points),
                    closed=bool(stroke.closed),
                    first_seen_frame=frame_index,
                    first_seen_ms=timestamp_ms,
                    last_seen_frame=frame_index,
                    last_seen_ms=timestamp_ms,
                    seen_count=1,
                )
                self._next_id += 1
                self._active[tracked.id] = tracked
            if (tracked.status == "pending"
                    and tracked.seen_count >= self.stable_after):
                tracked.status = "stable"
                newly_stable.append(tracked)
        # A ordem é atribuída por primeiro aparecimento, não pela
        # ordem da lista de entrada desta amostra (que, no app, é
        # por centroide): dois traços que estabilizam juntos
        # entram na ordem em que apareceram.
        for tracked in sorted(newly_stable,
                              key=lambda t: (t.first_seen_frame, t.id)):
            tracked.order = self._next_order
            self._next_order += 1
            self._ordered.append(tracked)
        return list(self._active.values())


def _tracked_to_entry(tracked, scale):
    points_mm = []
    if scale is not None:
        points_mm = [
            [round(x, 3), round(y, 3)]
            for x, y in (export_module.px_to_mm(p, scale)
                         for p in tracked.points)
        ]
    length = 0.0
    for (x1, y1), (x2, y2) in zip(tracked.points, tracked.points[1:]):
        length += math.hypot(x2 - x1, y2 - y1)
    if tracked.closed and len(tracked.points) >= 2:
        (x1, y1), (x2, y2) = tracked.points[-1], tracked.points[0]
        length += math.hypot(x2 - x1, y2 - y1)
    entry = {
        # Campos v1 intactos: um leitor v1 que só lê version,
        # width, height, strokes e points continua funcionando.
        "points": [[float(x), float(y)] for x, y in tracked.points],
        "length": float(length),
        "closed": bool(tracked.closed),
        "id": tracked.id,
        "order": tracked.order,
        "first_seen_frame": tracked.first_seen_frame,
        "first_seen_ms": tracked.first_seen_ms,
        "last_seen_frame": tracked.last_seen_frame,
        "last_seen_ms": tracked.last_seen_ms,
        "status": tracked.status,
    }
    if scale is not None:
        entry["points_mm"] = points_mm
        entry["length_mm"] = round(length * scale.mm_per_px, 3)
    return entry


def session_to_dict(tracker, scale=None, paper="A4",
                    include_erased=True):
    """Sessão em JSON v2, compatível com leitores v1.

    Os traços saem em ordem de aparecimento estável. ``version``
    sobe para 2 porque há campos novos, mas ``width``/``height``/
    ``points`` continuam presentes e em pixels, exatamente como
    no v1.
    """
    if scale is None:
        scale = export_module.sheet_scale(paper=paper)
    strokes = tracker.ordered
    if not include_erased:
        strokes = [t for t in strokes if t.status == "stable"]
    return {
        "version": 2,
        "width": scale.canvas_width_px,
        "height": scale.canvas_height_px,
        "paper": scale.paper,
        "units": "mm",
        "scale_mm_per_px": scale.mm_per_px,
        "canvas_px": [scale.canvas_width_px, scale.canvas_height_px],
        "sheet_mm": [round(scale.effective_width_mm, 2),
                     round(scale.effective_height_mm, 2)],
        "coordinate_system": "sheet_top_left_y_down",
        "session_breaks": tracker.breaks,
        "strokes": [_tracked_to_entry(t, scale) for t in strokes],
    }


def session_to_svg(tracker, scale=None, paper="A4",
                   stroke_width_mm=0.5, include_erased=True):
    """Sessão em SVG em mm, com paths na ordem da sequência."""
    if scale is None:
        scale = export_module.sheet_scale(paper=paper)
    strokes = tracker.ordered
    if not include_erased:
        strokes = [t for t in strokes if t.status == "stable"]
    svg_strokes = [
        strokes_module.Stroke(points=list(t.points), closed=t.closed,
                              id=t.id, order=t.order)
        for t in strokes
    ]
    return export_module.strokes_to_svg_scaled(
        svg_strokes, scale=scale, stroke_width_mm=stroke_width_mm
    )


def export_session(tracker, base_path, scale=None, paper="A4",
                   stroke_width_mm=0.5):
    """Escreve base_path.svg (mm, ordem da sessão) e base_path.json (v2)."""
    if scale is None:
        scale = export_module.sheet_scale(paper=paper)
    base = Path(base_path)
    svg_path = base.with_suffix(".svg")
    json_path = base.with_suffix(".json")
    svg_path.write_text(
        session_to_svg(tracker, scale=scale,
                       stroke_width_mm=stroke_width_mm),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(session_to_dict(tracker, scale=scale), indent=2),
        encoding="utf-8",
    )
    return svg_path, json_path
