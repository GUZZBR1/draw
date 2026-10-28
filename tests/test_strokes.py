"""Testes da v0.2, traço vetorial, sem webcam.

A cena sintética é a mesma do selftest: folha inclinada, com um
círculo, uma linha e um X desenhados por código. O X vira 4
segmentos, porque junções dividem traços, então o total esperado
é pelo menos 3 traços, e não exatamente 3.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config, ink, paper, strokes  # noqa: E402
from tests.selftest import build_scene  # noqa: E402


def test_empty_mask_has_no_strokes():
    mask = np.zeros((config.PAPER_H, config.PAPER_W), dtype=np.uint8)
    assert strokes.extract_strokes(mask) == []


def test_straight_line_is_unchanged_by_smoothing():
    points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    smoothed = strokes.smooth_points(points, passes=1)
    for (x1, y1), (x2, y2) in zip(points, smoothed):
        assert abs(x1 - x2) < 1e-9
        assert abs(y1 - y2) < 1e-9


def test_synthetic_scene_produces_strokes():
    scene, _truth = build_scene()
    warped, markers = paper.warp_paper(scene)
    assert warped is not None
    assert len(markers) == 4
    ink_mask = ink.extract_ink(warped)
    found = strokes.extract_strokes(ink_mask)
    assert len(found) >= 3
    for stroke in found:
        assert len(stroke.points) >= 2
        assert stroke.length >= 15.0
    # Ordem determinística: centroides em ordem crescente de y,
    # com x como desempate.
    centroids = [s.centroid for s in found]
    assert centroids == sorted(centroids, key=lambda c: (c[1], c[0]))


def test_svg_and_json_roundtrip(tmp_path):
    scene, _truth = build_scene()
    warped, _markers = paper.warp_paper(scene)
    ink_mask = ink.extract_ink(warped)
    found = strokes.extract_strokes(ink_mask)
    svg = strokes.strokes_to_svg(found)
    assert svg.startswith("<svg")
    assert svg.count("<path") == len(found)
    assert svg.strip().endswith("</svg>")
    svg_path, json_path = strokes.export_strokes(found, tmp_path / "out")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["width"] == config.PAPER_W
    assert data["height"] == config.PAPER_H
    assert len(data["strokes"]) == len(found)
    assert svg_path.read_text(encoding="utf-8") == svg
