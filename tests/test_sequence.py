"""Testes da etapa 5 (ordem temporal), sintéticos, sem hardware.

O rastreador recebe Stroke prontos, não câmera: cada teste alimenta
amostras na ordem em que os traços "apareceriam" e confere a ordem
registrada. stable_after=3 nos testes, para não repetir 12 amostras
à toa; o padrão de produção continua 12.
"""

import json

from draw import sequence as sequence_module
from draw.strokes import Stroke


def _line(x1, y1, x2, y2):
    return Stroke(points=[(float(x1), float(y1)), (float(x2), float(y2))])


def _square(cx, cy, half=20):
    return Stroke(
        points=[
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ],
        closed=True,
    )


def _feed(tracker, frames):
    """frames: lista de (frame_index, [strokes]); devolve o tracker."""
    for frame_index, strokes in frames:
        tracker.update(strokes, frame_index=frame_index,
                       timestamp_ms=frame_index * 100)
    return tracker


def test_order_is_appearance_not_centroid():
    """Linha embaixo aparece primeiro, círculo em cima depois.

    Por centroide o círculo viria primeiro (y menor). A sessão
    tem que registrar [linha, círculo, braço], na ordem em que
    apareceram, com first_seen 1, 5 e 9.
    """
    tracker = sequence_module.SequenceTracker(stable_after=3)
    line = _line(10, 200, 110, 200)
    circle = _square(60, 60)
    arm = _line(300, 300, 330, 330)
    _feed(tracker, [
        (1, [line]),
        (2, [line]),
        (3, [line]),
        (4, [line]),
        (5, [line, circle]),
        (6, [line, circle]),
        (7, [line, circle]),
        (8, [line, circle]),
        (9, [line, circle, arm]),
        (10, [line, circle, arm]),
        (11, [line, circle, arm]),
    ])
    ordered = tracker.ordered
    assert [t.order for t in ordered] == [0, 1, 2]
    assert [t.first_seen_frame for t in ordered] == [1, 5, 9]
    assert ordered[0].points == line.points
    assert ordered[1].closed is True
    assert ordered[2].points == arm.points


def test_flicker_never_becomes_order():
    tracker = sequence_module.SequenceTracker(stable_after=3)
    line = _line(10, 200, 110, 200)
    flicker = _line(300, 300, 330, 300)
    _feed(tracker, [
        (1, [line]),
        (2, [line, flicker]),  # flicker aparece 1 amostra só
        (3, [line]),
        (4, [line]),
    ])
    assert [t.points for t in tracker.ordered] == [line.points]
    assert tracker.ordered[0].status == "stable"


def test_stability_threshold_is_exact():
    tracker = sequence_module.SequenceTracker(stable_after=3)
    line = _line(10, 200, 110, 200)
    tracker.update([line], frame_index=1)
    tracker.update([line], frame_index=2)
    assert tracker.ordered == []
    tracker.update([line], frame_index=3)
    assert len(tracker.ordered) == 1
    assert tracker.ordered[0].order == 0


def test_growing_stroke_keeps_id_and_order():
    tracker = sequence_module.SequenceTracker(stable_after=3)
    short = _line(10, 200, 60, 200)
    grown = _line(10, 200, 110, 200)
    _feed(tracker, [
        (1, [short]),
        (2, [short]),
        (3, [short]),
        (4, [grown]),
        (5, [grown]),
    ])
    assert len(tracker.ordered) == 1
    tracked = tracker.ordered[0]
    assert tracked.order == 0
    assert tracked.points == grown.points
    assert tracked.last_seen_frame == 5


def test_sheet_jump_breaks_matching():
    tracker = sequence_module.SequenceTracker(stable_after=3)
    line = _line(10, 200, 110, 200)
    corners = [(0, 0), (100, 0), (100, 100), (0, 100)]
    jumped = [(200, 0), (300, 0), (300, 100), (200, 100)]
    tracker.update([line], frame_index=1, corners=corners)
    tracker.update([line], frame_index=2, corners=corners)
    tracker.update([line], frame_index=3, corners=corners)
    assert tracker.breaks == 0
    old_id = tracker.ordered[0].id
    # Mesma geometria, folha saltada: não pode casar com o traço antigo.
    tracker.update([line], frame_index=4, corners=jumped)
    assert tracker.breaks == 1
    tracker.update([line], frame_index=5, corners=jumped)
    tracker.update([line], frame_index=6, corners=jumped)
    assert len(tracker.ordered) == 2
    assert tracker.ordered[0].id == old_id
    assert tracker.ordered[1].id != old_id
    assert tracker.ordered[1].order == 1


def test_unlocked_sheet_is_ignored():
    tracker = sequence_module.SequenceTracker(stable_after=3)
    line = _line(10, 200, 110, 200)
    tracker.update([line], frame_index=1, sheet_locked=False)
    tracker.update([line], frame_index=2, sheet_locked=False)
    tracker.update([line], frame_index=3, sheet_locked=False)
    assert tracker.ordered == []
    assert tracker.samples == 0


def test_erase_and_redraw_gets_new_order():
    tracker = sequence_module.SequenceTracker(stable_after=3)
    line = _line(10, 200, 110, 200)
    _feed(tracker, [
        (1, [line]),
        (2, [line]),
        (3, [line]),
        (4, []),  # apagado
        (5, [line]),  # redesenhado: novo ato, nova ordem
        (6, [line]),
        (7, [line]),
    ])
    ordered = tracker.ordered
    assert len(ordered) == 2
    assert ordered[0].status == "erased"
    assert ordered[0].order == 0
    assert ordered[1].status == "stable"
    assert ordered[1].order == 1
    assert ordered[1].id != ordered[0].id
    assert [t.id for t in tracker.final_strokes()] == [ordered[1].id]


def test_session_json_v2_keeps_v1_fields():
    tracker = sequence_module.SequenceTracker(stable_after=3)
    line = _line(10, 200, 110, 200)
    circle = _square(60, 60)
    _feed(tracker, [
        (1, [line]),
        (2, [line]),
        (3, [line]),
        (4, [line, circle]),
        (5, [line, circle]),
        (6, [line, circle]),
    ])
    data = sequence_module.session_to_dict(tracker)
    assert data["version"] == 2
    # Campos que um leitor v1 lê: presentes e em pixels.
    assert data["width"] == 850
    assert data["height"] == 1200
    assert len(data["strokes"]) == 2
    assert data["strokes"][0]["points"] == [
        [float(x), float(y)] for x, y in line.points
    ]
    assert data["strokes"][0]["order"] == 0
    assert data["strokes"][1]["order"] == 1
    assert data["strokes"][0]["points_mm"][0][0] >= 0
    assert data["strokes"][0]["first_seen_frame"] == 1
    assert data["strokes"][0]["status"] == "stable"
    assert data["coordinate_system"] == "sheet_top_left_y_down"


def test_session_svg_follows_sequence_order(tmp_path):
    tracker = sequence_module.SequenceTracker(stable_after=3)
    line = _line(10, 200, 110, 200)
    circle = _square(60, 60)
    _feed(tracker, [
        (1, [line]),
        (2, [line]),
        (3, [line]),
        (4, [line, circle]),
        (5, [line, circle]),
        (6, [line, circle]),
    ])
    svg_path, json_path = sequence_module.export_session(
        tracker, tmp_path / "session"
    )
    svg = svg_path.read_text(encoding="utf-8")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # A linha (embaixo) vem antes do círculo (em cima) no SVG da
    # sessão, ao contrário da ordem por centroide do snapshot.
    assert svg.index('data-order="0"') < svg.index('data-order="1"')
    assert 'width="210.00mm"' in svg
    assert [s["order"] for s in data["strokes"]] == [0, 1]
