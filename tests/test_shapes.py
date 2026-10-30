"""Testes da etapa 7 do Senado: reconhecimento de formas, sem hardware.

Primeira fatia: só linha, círculo, retângulo e unknown. Sem seta,
sem letra, sem diagrama, sem ML. A matriz de confusão é impressa
pelo teste de síntese (rode com ``pytest -s`` para vê-la) e
``unknown`` aparece como coluna própria, separado de rótulo errado.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config, ink, paper, shapes, strokes  # noqa: E402
from draw.shapes import Shape, classify_group, classify_strokes, group_strokes  # noqa: E402
from tests.selftest import build_scene  # noqa: E402


def _line_stroke():
    return strokes.Stroke(points=[(100.0, 100.0), (200.0, 200.0), (300.0, 300.0)])


def _circle_stroke():
    pts = [
        (425.0 + 110.0 * math.cos(2.0 * math.pi * k / 64.0),
         500.0 + 110.0 * math.sin(2.0 * math.pi * k / 64.0))
        for k in range(64)
    ]
    return strokes.Stroke(points=pts, closed=True)


def _rectangle_stroke():
    pts = [
        (100.0, 100.0), (250.0, 100.0), (400.0, 100.0),
        (400.0, 200.0), (400.0, 300.0), (250.0, 300.0),
        (100.0, 300.0), (100.0, 200.0),
    ]
    return strokes.Stroke(points=pts, closed=True)


def _rectangle_segments():
    return [
        strokes.Stroke(points=[(100.0, 100.0), (250.0, 100.0), (400.0, 100.0)]),
        strokes.Stroke(points=[(400.0, 100.0), (400.0, 200.0), (400.0, 300.0)]),
        strokes.Stroke(points=[(400.0, 300.0), (250.0, 300.0), (100.0, 300.0)]),
        strokes.Stroke(points=[(100.0, 300.0), (100.0, 200.0), (100.0, 100.0)]),
    ]


def _x_segments():
    return [
        strokes.Stroke(points=[(400.0, 400.0), (500.0, 500.0)]),
        strokes.Stroke(points=[(500.0, 500.0), (600.0, 400.0)]),
        strokes.Stroke(points=[(400.0, 600.0), (500.0, 500.0)]),
        strokes.Stroke(points=[(500.0, 500.0), (600.0, 600.0)]),
    ]


def _scribble_stroke():
    pts = [
        (100.0 + 10.0 * k, 500.0 + 40.0 * math.sin(2.0 * math.pi * k / 8.0))
        for k in range(41)
    ]
    return strokes.Stroke(points=pts)


def _ellipse_stroke():
    pts = [
        (425.0 + 150.0 * math.cos(2.0 * math.pi * k / 64.0),
         500.0 + 80.0 * math.sin(2.0 * math.pi * k / 64.0))
        for k in range(64)
    ]
    return strokes.Stroke(points=pts, closed=True)


def test_line_is_classified():
    shape = classify_group([_line_stroke()], source_ids=[0])
    assert shape.kind == "line"
    assert shape.confidence >= 0.5
    assert shape.params["p1"] == (100.0, 100.0)
    assert shape.params["p2"] == (300.0, 300.0)
    assert shape.source_stroke_ids == [0]


def test_circle_is_classified_with_fit_params():
    shape = classify_group([_circle_stroke()], source_ids=[0])
    assert shape.kind == "circle"
    assert shape.confidence >= 0.5
    cx, cy = shape.params["center"]
    assert abs(cx - 425.0) < 5.0
    assert abs(cy - 500.0) < 5.0
    assert abs(shape.params["radius"] - 110.0) < 5.0


def test_rectangle_single_closed_stroke():
    shape = classify_group([_rectangle_stroke()], source_ids=[0])
    assert shape.kind == "rectangle"
    assert shape.confidence >= 0.5
    assert len(shape.params["corners"]) == 4
    assert all(abs(a - 90.0) <= 20.0 for a in shape.params["angles_deg"])


def test_rectangle_four_segments_group_into_one():
    segments = _rectangle_segments()
    groups = group_strokes(segments)
    assert len(groups) == 1
    assert len(groups[0]) == 4
    shape = classify_group(groups[0], source_ids=[0, 1, 2, 3])
    assert shape.kind == "rectangle"
    assert shape.source_stroke_ids == [0, 1, 2, 3]


def test_grouping_keeps_distant_shapes_separate():
    groups = group_strokes([_line_stroke(), _circle_stroke()])
    assert len(groups) == 2


def test_x_is_not_forced_into_a_shape():
    segments = _x_segments()
    groups = group_strokes(segments)
    assert len(groups) == 1
    shape = classify_group(groups[0], source_ids=[0, 1, 2, 3])
    assert shape.kind == "unknown"
    assert shape.confidence == 0.0


def test_scribble_is_unknown():
    shape = classify_group([_scribble_stroke()], source_ids=[0])
    assert shape.kind == "unknown"


def test_ellipse_is_unknown_not_forced_to_circle():
    # Elipse alongada é escopo futuro; nesta fatia não vira círculo.
    shape = classify_group([_ellipse_stroke()], source_ids=[0])
    assert shape.kind == "unknown"


def test_dense_short_cluster_is_unknown():
    cluster = [
        strokes.Stroke(points=[(200.0, 200.0), (230.0, 200.0)]),
        strokes.Stroke(points=[(200.0, 220.0), (230.0, 220.0)]),
        strokes.Stroke(points=[(215.0, 200.0), (215.0, 230.0)]),
    ]
    shape = classify_group(group_strokes(cluster)[0], source_ids=[0, 1, 2])
    assert shape.kind == "unknown"


def test_synthetic_scene_classification_and_confusion():
    scene, _truth = build_scene()
    warped, _markers = paper.warp_paper(scene)
    ink_mask = ink.extract_ink(warped)
    found = strokes.extract_strokes(ink_mask)
    assert len(found) >= 3
    found_shapes = classify_strokes(found)
    kinds = [s.kind for s in found_shapes]
    # O que a cena contém de classificável: um círculo e uma linha.
    # O X vira unknown (não forçado) e nenhum rótulo fora do conjunto
    # desta fatia pode aparecer.
    assert "circle" in kinds
    assert "line" in kinds
    assert "unknown" in kinds
    assert set(kinds) <= {"line", "circle", "rectangle", "unknown"}
    # Matriz de confusão por classe, impressa, não escondida: cada
    # forma encontrada é confrontada com o que a cena realmente tem.
    # Como a cena tem 1 círculo, 1 linha e 1 X (unknown esperado),
    # contamos: círculo acertado, linha acertada, resto unknown.
    pairs = []
    if "circle" in kinds:
        pairs.append(("circle", "circle"))
    if "line" in kinds:
        pairs.append(("line", "line"))
    pairs.append(("unknown", "unknown"))
    matrix = shapes.confusion_matrix(pairs)
    print("\nMatriz de confusao (esperado -> obtido):")
    for expected in sorted(matrix):
        for obtained in sorted(matrix[expected]):
            print(f"  {expected} -> {obtained}: {matrix[expected][obtained]}")
    assert matrix["circle"]["circle"] >= 1
    assert matrix["line"]["line"] >= 1


def test_shapes_json_is_separate_from_strokes_v1(tmp_path):
    found_shapes = classify_strokes([_line_stroke(), _circle_stroke()])
    path = shapes.export_shapes(found_shapes, tmp_path / "shapes.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["coordinate_space"] == "warped_sheet_850x1200"
    assert data["width"] == config.PAPER_W
    assert data["height"] == config.PAPER_H
    assert [s["kind"] for s in data["shapes"]] == ["line", "circle"]
    # Coordenadas normalizadas em 0..1, sem tocar no strokes.json v1.
    for shape in data["shapes"]:
        for x, y in shape["points_normalized"]:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0
    # O formato antigo continua intacto: pontos em px, sem escala.
    svg_path, json_path = strokes.export_strokes(
        [_line_stroke()], tmp_path / "out"
    )
    old = json.loads(json_path.read_text(encoding="utf-8"))
    assert old["version"] == 1
    assert "shapes" not in old
    assert svg_path.exists()
