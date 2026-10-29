"""Testes do avaliador de dataset (etapa 2 do Senado), sem webcam.

O avaliador roda o pipeline real sobre fotos com verdade declarada.
Aqui ele é provado sobre cenas sintéticas (as do selftest) e sobre
casos montados em diretórios temporários, nunca sobre fotos reais
commitadas: o manifesto do repo começa com zero casos reais.
"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_dataset", ROOT / "tools" / "evaluate_dataset.py"
)
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)


def _truth(**overrides):
    truth = {
        "id": "case-1",
        "split": "dev",
        "photo": "photo.png",
        "paper": {"type": "white", "size": "A4"},
        "lighting": "artificial",
        "consent": True,
        "license": "internal-only",
        "sheet_corners": [[0, 0], [100, 0], [100, 100], [0, 100]],
    }
    truth.update(overrides)
    return truth


def test_shipped_manifest_has_zero_real_cases_and_passes():
    report = ev.evaluate_manifest(ROOT / "dataset" / "manifest.json")
    assert report["results"] == []
    assert report["overall"]["cases"] == 0
    assert report["overall"]["pass"] is True


def test_validate_truth_rejects_missing_consent_and_license():
    errors = ev.validate_truth(_truth(consent=False), Path("."))
    assert any("consent" in e for e in errors)
    errors = ev.validate_truth(_truth(license=""), Path("."))
    assert any("license" in e for e in errors)


def test_validate_truth_rejects_bad_split_and_missing_corners():
    assert ev.validate_truth(_truth(split="train"), Path("."))
    assert ev.validate_truth(
        _truth(sheet_corners=None, expects_sheet=True), Path(".")
    )


def test_validate_truth_accepts_negative_without_corners(tmp_path):
    (tmp_path / "photo.png").write_bytes(b"")
    assert ev.validate_truth(
        _truth(sheet_corners=None, expects_sheet=False), tmp_path
    ) == []


def test_load_manifest_rejects_wrong_version(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest_version": 99, "cases": []}))
    try:
        ev.load_manifest(manifest)
    except ValueError as exc:
        assert "manifest_version" in str(exc)
    else:
        raise AssertionError("versão errada deveria ser recusada")


def test_load_manifest_rejects_case_without_truth(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "manifest_version": 1,
        "baseline": {"min_detection_rate": 1.0,
                     "max_corner_error_px": 12.0, "min_ink_recall": 0.90},
        "cases": [{"id": "ghost", "split": "dev"}],
    }))
    try:
        ev.load_manifest(manifest)
    except ValueError as exc:
        assert "truth.json ausente" in str(exc)
    else:
        raise AssertionError("caso sem truth deveria ser recusado")


def test_synthetic_evaluation_passes_all_four_cases():
    report = ev.evaluate_synthetic()
    by_id = {r["id"]: r for r in report["results"]}
    assert set(by_id) == {
        "synthetic-markerless", "synthetic-nested",
        "synthetic-tray-negative", "synthetic-blank-sheet",
    }
    assert report["overall"]["pass"] is True
    for case_id in ("synthetic-markerless", "synthetic-nested"):
        result = by_id[case_id]
        assert result["detected"] is True
        assert result["corner_error_max_px"] <= 12.0
        assert result["ink_recall"] >= 0.90
        assert result["stroke_count"] > 0
    assert by_id["synthetic-tray-negative"]["detected"] is False
    assert by_id["synthetic-blank-sheet"]["detected"] is False
    assert by_id["synthetic-blank-sheet"]["verdict"] == "blank_maybe"


def test_synthetic_evaluation_is_deterministic():
    first = ev.evaluate_synthetic()
    second = ev.evaluate_synthetic()
    assert first["results"] == second["results"]


def test_synthetic_cli_exits_zero(capsys):
    assert ev.main(["--synthetic"]) == 0
    assert "PASS" in capsys.readouterr().out


def _write_case(case_dir, scene, truth, ink_mask=None):
    case_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(case_dir / "photo.png"), scene)
    if ink_mask is not None:
        cv2.imwrite(str(case_dir / "ink.png"), ink_mask)
        truth = dict(truth, ink_mask="ink.png")
    (case_dir / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False), encoding="utf-8"
    )


def _markerless_case(tmp_path):
    builders = ev._load_selftest_builders()
    scene, gt_ink, gt_corners = builders.build_markerless_scene()
    case_dir = tmp_path / "cases" / "real-dev-1"
    _write_case(case_dir, scene, _truth(
        id="real-dev-1", sheet_corners=gt_corners.tolist(),
        classes_present=["circle", "line"],
    ), ink_mask=gt_ink)
    return case_dir, gt_ink


def _manifest(tmp_path, cases):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "manifest_version": 1,
        "baseline": {"min_detection_rate": 1.0,
                     "max_corner_error_px": 12.0, "min_ink_recall": 0.90},
        "cases": cases,
    }, ensure_ascii=False), encoding="utf-8")
    return manifest


def test_real_case_end_to_end_with_ink_mask(tmp_path):
    _markerless_case(tmp_path)
    manifest = _manifest(tmp_path, [
        {"id": "real-dev-1", "split": "dev", "dir": "cases/real-dev-1"},
    ])
    report = ev.evaluate_manifest(manifest)
    assert report["overall"]["pass"] is True
    result = report["results"][0]
    assert result["detected"] is True
    assert result["corner_error_max_px"] <= 12.0
    assert result["ink_recall"] >= 0.90


def test_split_filter_keeps_dev_and_test_separate(tmp_path):
    case_dir, _ = _markerless_case(tmp_path)
    builders = ev._load_selftest_builders()
    negatives = builders.build_negative_scenes()
    neg_dir = tmp_path / "cases" / "real-test-neg"
    _write_case(neg_dir, negatives["bandeja branca"], _truth(
        id="real-test-neg", split="test", expects_sheet=False,
        sheet_corners=None,
    ))
    manifest = _manifest(tmp_path, [
        {"id": "real-dev-1", "split": "dev", "dir": "cases/real-dev-1"},
        {"id": "real-test-neg", "split": "test", "dir": "cases/real-test-neg"},
    ])
    dev_only = ev.evaluate_manifest(manifest, split="dev")
    assert [r["id"] for r in dev_only["results"]] == ["real-dev-1"]
    assert dev_only["by_split"]["test"]["cases"] == 0
    test_only = ev.evaluate_manifest(manifest, split="test")
    assert [r["id"] for r in test_only["results"]] == ["real-test-neg"]
    assert test_only["overall"]["pass"] is True


def test_cli_invalid_manifest_exits_two(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    assert ev.main([str(manifest)]) == 2


def test_cli_writes_markdown_report(tmp_path):
    _markerless_case(tmp_path)
    manifest = _manifest(tmp_path, [
        {"id": "real-dev-1", "split": "dev", "dir": "cases/real-dev-1"},
    ])
    out = tmp_path / "relatorio.md"
    assert ev.main([str(manifest), "--report", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "real-dev-1" in text
    assert "PASS" in text


def test_blank_frame_is_load_error_not_detection():
    result = ev.evaluate_frame(None, _truth())
    assert result.status == "error"
    assert result.failure_class == "load"
