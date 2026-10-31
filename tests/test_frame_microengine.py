import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "frame_microengine.py"
SPEC = importlib.util.spec_from_file_location("frame_microengine", MODULE_PATH)
engine = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(engine)


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _manifest(tmp_path: Path, groups_per_stratum: int = 4) -> tuple[list[dict[str, str]], Path]:
    rows = []
    source_root = tmp_path / "sources"
    for session_index, session in enumerate(("session-a", "session-b")):
        frame_index = 0
        for priority, status in (("high", "review_selected"), ("low", "not_selected_redundant")):
            for group_index in range(groups_per_stratum):
                frame_index += 1
                image = np.full((12, 16, 3), session_index * 50 + frame_index, dtype=np.uint8)
                path = source_root / session / f"{frame_index:04}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(path), image)
                rows.append({
                    "session_id": session,
                    "frame_index": str(frame_index),
                    "source_path": str(path.relative_to(source_root)),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "selection_status": status,
                    "review_group_id": f"{priority}-group-{group_index}",
                    "take_id": f"take-{group_index}",
                    "category_tags": "hand;paper",
                    "priority_score": str(0.9 if priority == "high" else 0.1),
                })
    return rows, source_root


def _counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["session_id"]] = counts.get(row["session_id"], 0) + 1
    return counts


def _renumber_frames(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    renumbered = [dict(row) for row in rows]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in renumbered:
        grouped.setdefault(row["session_id"], []).append(row)
    for session_rows in grouped.values():
        session_rows.sort(key=lambda row: int(row["frame_index"]))
        for index, row in enumerate(session_rows, start=1):
            row["frame_index"] = str(index)
    return renumbered


def _fill_reviews(output: Path, labels: dict[str, str], tags: str = "hand;paper") -> list[Path]:
    paths = []
    packet = list(csv.DictReader((output / "blind_packet.csv").open(encoding="utf-8")))
    for reviewer_index in range(1, 4):
        path = output / f"filled_{reviewer_index}.csv"
        rows = []
        for item in packet:
            rows.append({
                "reviewer_id": f"r{reviewer_index}",
                "blind_id": item["blind_id"],
                "useful": labels.get(item["blind_id"], "no"),
                "visible_tags": tags,
                "confidence": "high",
                "notes": "",
            })
        _write_csv(path, rows)
        paths.append(path)
    return paths


def test_sample_is_deterministic_and_excludes_reviewed_ids(tmp_path: Path) -> None:
    rows, _ = _manifest(tmp_path)
    kwargs = dict(seed=42, low_per_session=2, high_per_session=2, expected_session_counts=_counts(rows))
    first = engine.sample_manifest(rows, **kwargs)
    second = engine.sample_manifest(rows, **kwargs)
    assert [(r["session_id"], r["frame_index"]) for r in first] == [
        (r["session_id"], r["frame_index"]) for r in second
    ]
    assert len({r["audit_group_id"] for r in first if r["session_id"] == "session-a" and r["priority"] == "high"}) == 2
    excluded = {(str(first[0]["session_id"]), str(first[0]["frame_index"]))}
    remaining = engine.sample_manifest(rows, excluded_ids=excluded, **kwargs)
    assert excluded.isdisjoint((r["session_id"], r["frame_index"]) for r in remaining)
    assert all(0 < r["frame_inclusion_probability"] <= 1 and r["sample_weight"] >= 1 for r in first)


def test_blind_packet_has_no_source_or_prediction_fields(tmp_path: Path) -> None:
    rows, source_root = _manifest(tmp_path)
    manifest = tmp_path / "selection.csv"
    _write_csv(manifest, rows)
    output = tmp_path / "audit"
    engine.build_blind_audit(
        manifest, output, source_root=source_root, low_per_session=2, high_per_session=2, seed=7,
        expected_session_counts=_counts(rows),
    )
    packet = list(csv.DictReader((output / "blind_packet.csv").open(encoding="utf-8")))
    assert set(packet[0]) == {"blind_id", "image_path"}
    mapping = list(csv.DictReader((output / "audit_mapping.private.csv").open(encoding="utf-8")))
    assert [row["blind_id"] for row in packet] != [row["blind_id"] for row in mapping]
    packet_text = json.dumps(packet).lower()
    assert all(value not in packet_text for value in ("session", "frame", "score", "prediction", "priority", "selection"))
    assert len(list((output / "images").glob("*.png"))) == len(packet)
    assert len(mapping) == len(packet)
    reviewer_orders = []
    for index in range(1, 4):
        template = list(csv.DictReader((output / f"labels_reviewer_{index}.csv").open(encoding="utf-8")))
        assert len(template) == len(packet)
        reviewer_orders.append([row["blind_id"] for row in template])
        assert all(not row["useful"] and not row["visible_tags"] for row in template)
    assert reviewer_orders == [[row["blind_id"] for row in packet]] * 3


def test_duplicate_source_ids_and_checksum_mismatch_fail_closed(tmp_path: Path) -> None:
    rows, source_root = _manifest(tmp_path)
    with pytest.raises(ValueError, match="duplicate source ID"):
        engine.validate_manifest(rows + [dict(rows[0])], expected_session_counts=_counts(rows))

    manifest = tmp_path / "selection.csv"
    changed = [dict(row) for row in rows]
    changed = [dict(row, source_sha256="0" * 64) for row in rows]
    _write_csv(manifest, changed)
    with pytest.raises(ValueError, match="checksum mismatch"):
        engine.build_blind_audit(
            manifest, tmp_path / "bad-audit", source_root=source_root,
            low_per_session=1, high_per_session=1, seed=3,
            expected_session_counts=_counts(changed),
        )


def test_manifest_requires_both_complete_project_sessions_by_default(tmp_path: Path) -> None:
    assert engine.EXPECTED_SESSION_COUNTS == {
        "session_20261029_130013_453419": 2518,
        "session_20261029_151126_089557": 13327,
    }
    rows, _ = _manifest(tmp_path)
    with pytest.raises(ValueError, match="manifest source census mismatch"):
        engine.validate_manifest(rows)

    expected = _counts(rows)
    one_session = [row for row in rows if row["session_id"] == "session-a"]
    with pytest.raises(ValueError, match=r"missing_sessions=\['session-b'\]"):
        engine.validate_manifest(one_session, expected_session_counts=expected)

    incomplete = [row for row in rows if row is not rows[0]]
    with pytest.raises(ValueError, match="session-a.*expected.*actual"):
        engine.validate_manifest(incomplete, expected_session_counts=expected)

    wrong_index = [dict(row) for row in rows]
    wrong_index[0]["frame_index"] = "9999"
    with pytest.raises(ValueError, match="frame-index census mismatch"):
        engine.validate_manifest(wrong_index, expected_session_counts=expected)

    non_integer = [dict(row) for row in rows]
    non_integer[0]["frame_index"] = "frame-1"
    with pytest.raises(ValueError, match="non-integer frame_index"):
        engine.validate_manifest(non_integer, expected_session_counts=expected)


def test_reviewer_disagreement_and_uncertainty_are_conservative(tmp_path: Path) -> None:
    rows, source_root = _manifest(tmp_path)
    manifest = tmp_path / "selection.csv"
    _write_csv(manifest, rows)
    output = tmp_path / "audit"
    engine.build_blind_audit(
        manifest, output, source_root=source_root, low_per_session=2, high_per_session=2, seed=9,
        expected_session_counts=_counts(rows),
    )
    mapping_rows = list(csv.DictReader((output / "audit_mapping.private.csv").open(encoding="utf-8")))
    low_id = next(row["blind_id"] for row in mapping_rows if row["priority"] == "low")
    label_pattern = {low_id: "no"}
    paths = _fill_reviews(output, label_pattern)
    # One uncertain vote prevents the low-priority frame from being counted as a definite negative.
    for reviewer_index, path in enumerate(paths, start=1):
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        next(row for row in rows if row["blind_id"] == low_id)["useful"] = ("yes", "no", "uncertain")[reviewer_index - 1]
        _write_csv(path, rows)
    report = engine.evaluate_reviews(output / "audit_mapping.private.csv", paths, bootstrap_samples=200, seed=1)
    low_rows = [row for row in report["reviewed_rows"] if row["blind_id"] == low_id]
    assert low_rows[0]["consensus_useful"] == "uncertain"
    assert low_rows[0]["possible_useful"] is True
    assert report["status"] in {"inconclusive", "fail"}
    assert report["semantic_status"] == "three-agent image-level consensus; not pixel ground truth"


def test_zero_positive_metrics_are_defined_and_not_passed(tmp_path: Path) -> None:
    rows, source_root = _manifest(tmp_path)
    manifest = tmp_path / "selection.csv"
    _write_csv(manifest, rows)
    output = tmp_path / "audit"
    engine.build_blind_audit(
        manifest, output, source_root=source_root, low_per_session=2, high_per_session=2, seed=11,
        expected_session_counts=_counts(rows),
    )
    paths = _fill_reviews(output, {})
    report = engine.evaluate_reviews(output / "audit_mapping.private.csv", paths, bootstrap_samples=200, seed=2)
    assert all(stratum["consensus_useful_yes"] == 0 for stratum in report["strata"].values())
    assert all(bound == pytest.approx(1.0 - 0.05 ** (1 / 2)) for bound in report["low_priority_miss_upper_bounds_by_session"].values())
    assert report["high_minus_low_absolute_rate_difference_ci"]["lower_95"] == pytest.approx(0.0)
    assert report["status"] != "pass"


def test_group_census_below_requested_count_can_pass_with_other_gates(tmp_path: Path) -> None:
    rows, source_root = _manifest(tmp_path, groups_per_stratum=60)
    census_rows = _renumber_frames([
        row for row in rows
        if not (row["session_id"] == "session-a" and row["selection_status"] == "review_selected"
                and row["review_group_id"] != "high-group-0")
    ])
    sampled = engine.sample_manifest(census_rows, low_per_session=60, high_per_session=50, seed=5,
                                     expected_session_counts=_counts(census_rows))
    census_high = [row for row in sampled if row["session_id"] == "session-a" and row["priority"] == "high"]
    assert len(census_high) == 1
    assert census_high[0]["requested_groups"] == 50
    assert census_high[0]["actual_groups"] == 1
    assert census_high[0]["census_of_groups"] is True
    assert census_high[0]["insufficient_groups"] is False
    assert census_high[0]["group_inclusion_probability"] == 1.0
    assert census_high[0]["frame_inclusion_probability"] == 1.0
    assert census_high[0]["sample_weight"] == 1.0

    manifest = tmp_path / "short-selection.csv"
    _write_csv(manifest, census_rows)
    output = tmp_path / "short-audit"
    summary = engine.build_blind_audit(
        manifest, output, source_root=source_root,
        low_per_session=60, high_per_session=50, seed=5,
        expected_session_counts=_counts(census_rows),
    )
    stratum_key = "session-a:high"
    assert summary["sample_coverage_status"] == "includes_group_census"
    assert summary["strata"][stratum_key] == {
        "requested_groups": 50,
        "eligible_groups": 1,
        "actual_groups": 1,
        "census_of_groups": True,
        "insufficient_groups": False,
    }

    # A group census below the requested count is not itself an inconclusive
    # condition; the low-miss and weighted high-low gates decide the result.
    mapping = list(csv.DictReader((output / "audit_mapping.private.csv").open(encoding="utf-8")))
    labels = {
        row["blind_id"]: ("yes" if row["priority"] == "high" else "no")
        for row in mapping
    }
    reviewers = _fill_reviews(output, labels)
    report = engine.evaluate_reviews(output / "audit_mapping.private.csv", reviewers, bootstrap_samples=200, seed=5)
    short_report = report["strata"][stratum_key]
    assert short_report["requested_groups"] == 50
    assert short_report["actual_groups"] == 1
    assert short_report["census_of_groups"] is True
    assert short_report["insufficient_groups"] is False
    assert report["gates"]["sample_group_coverage_adequate"] is True
    assert report["gates"]["low_priority_possible_useful_upper95_at_most_5pct_each_session"] is True
    assert report["gates"]["high_minus_low_difference_lower95_above_20pp"] is True
    assert report["high_minus_low_absolute_rate_difference_ci"]["required_lower_95"] == 0.20
    assert report["status"] == "pass"


def test_empty_stratum_is_insufficient_and_evaluation_inconclusive(tmp_path: Path) -> None:
    rows, source_root = _manifest(tmp_path, groups_per_stratum=4)
    no_high = _renumber_frames([
        row for row in rows
        if not (row["session_id"] == "session-a" and row["selection_status"] == "review_selected")
    ])
    manifest = tmp_path / "empty-stratum.csv"
    _write_csv(manifest, no_high)
    output = tmp_path / "empty-audit"
    summary = engine.build_blind_audit(
        manifest, output, source_root=source_root, low_per_session=2, high_per_session=2, seed=5,
        expected_session_counts=_counts(no_high),
    )
    assert summary["strata"]["session-a:high"]["eligible_groups"] == 0
    assert summary["strata"]["session-a:high"]["insufficient_groups"] is True
    assert summary["sample_coverage_status"] == "insufficient_groups"
    mapping = list(csv.DictReader((output / "audit_mapping.private.csv").open(encoding="utf-8")))
    labels = {
        row["blind_id"]: ("yes" if row["priority"] == "high" else "no")
        for row in mapping
    }
    reviewers = _fill_reviews(output, labels)
    report = engine.evaluate_reviews(output / "audit_mapping.private.csv", reviewers, bootstrap_samples=100, seed=5)
    assert report["status"] == "inconclusive"
    assert report["gates"]["complete_session_priority_strata"] is False


def test_positive_difference_below_20pp_gate_does_not_pass(tmp_path: Path) -> None:
    rows, source_root = _manifest(tmp_path, groups_per_stratum=60)
    manifest = tmp_path / "below-threshold.csv"
    _write_csv(manifest, rows)
    output = tmp_path / "below-threshold-audit"
    engine.build_blind_audit(
        manifest, output, source_root=source_root,
        low_per_session=60, high_per_session=60, seed=12,
        expected_session_counts=_counts(rows),
    )
    mapping = list(csv.DictReader((output / "audit_mapping.private.csv").open(encoding="utf-8")))
    high_ids_by_session: dict[str, list[str]] = {}
    for row in mapping:
        if row["priority"] == "high":
            high_ids_by_session.setdefault(row["session_id"], []).append(row["blind_id"])
    labels = {
        blind_id: ("yes" if blind_id in set(sorted(ids)[:15]) else "no")
        for ids in high_ids_by_session.values()
        for blind_id in ids
    }
    # Every low-priority frame is a confirmed negative, so the miss gate passes.
    labels.update({row["blind_id"]: "no" for row in mapping if row["priority"] == "low"})
    reviewers = _fill_reviews(output, labels)
    report = engine.evaluate_reviews(output / "audit_mapping.private.csv", reviewers, bootstrap_samples=4000, seed=12)
    ci = report["high_minus_low_absolute_rate_difference_ci"]
    assert 0 < ci["lower_95"] < 0.20
    assert report["gates"]["high_minus_low_difference_lower95_above_20pp"] is False
    assert report["gates"]["low_priority_possible_useful_upper95_at_most_5pct_each_session"] is True
    assert report["status"] == "inconclusive"
