#!/usr/bin/env python3
"""Create and score blinded audits for the frame-triage ranking engine.

The tool never deletes source frames and never treats a low-priority frame as
negative training data.  It samples one frame from each selected review group,
keeps the source mapping in a private CSV, and gives reviewers only blind IDs
and re-encoded image copies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID_FIELDS = ("session_id", "frame_index")
EXPECTED_SESSION_COUNTS = {
    "session_20261029_130013_453419": 2518,
    "session_20261029_151126_089557": 13327,
}
REVIEW_FIELDS = (
    "reviewer_id", "blind_id", "useful", "visible_tags", "confidence", "notes",
)
PUBLIC_PACKET_FIELDS = ("blind_id", "image_path")
MAPPING_FIELDS = (
    "blind_id", "session_id", "frame_index", "source_path", "source_sha256",
    "priority", "selection_status", "review_group_id", "population_groups",
    "population_frames", "requested_groups", "actual_groups", "sampled_groups",
    "census_of_groups", "insufficient_groups", "group_inclusion_probability",
    "frame_inclusion_probability", "sample_weight",
)
VALID_USEFUL = {"yes", "no", "uncertain"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _source_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("session_id", "").strip(), row.get("frame_index", "").strip()


def validate_manifest(
    rows: list[dict[str, str]],
    *,
    expected_session_counts: dict[str, int] | None = None,
) -> None:
    """Fail closed on source identity, hashes, and the complete project census."""
    seen: set[tuple[str, str]] = set()
    frame_indices: dict[str, set[int]] = defaultdict(set)
    for line, row in enumerate(rows, start=2):
        key = _source_key(row)
        if not all(key):
            raise ValueError(f"manifest row {line} has an empty source ID")
        if key in seen:
            raise ValueError(f"duplicate source ID in manifest: {key[0]}:{key[1]}")
        seen.add(key)
        raw_index = key[1]
        try:
            frame_index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"manifest row {line} has a non-integer frame_index: {raw_index!r}") from exc
        if str(frame_index) != raw_index:
            raise ValueError(f"manifest row {line} has a non-canonical integer frame_index: {raw_index!r}")
        frame_indices[key[0]].add(frame_index)
        if not row.get("source_path", "").strip():
            raise ValueError(f"manifest row {line} has no source_path")
        digest = row.get("source_sha256", "").strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"manifest row {line} has an invalid source_sha256")
    expected = EXPECTED_SESSION_COUNTS if expected_session_counts is None else expected_session_counts
    actual: dict[str, int] = defaultdict(int)
    for row in rows:
        actual[row["session_id"].strip()] += 1
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = {
            session: {"expected": expected[session], "actual": actual.get(session, 0)}
            for session in expected
            if actual.get(session, 0) != expected[session]
        }
        raise ValueError(
            "manifest source census mismatch: "
            f"missing_sessions={missing}, unexpected_sessions={unexpected}, counts={mismatched}"
        )
    index_mismatches = {}
    for session, expected_count in expected.items():
        expected_indices = set(range(1, expected_count + 1))
        actual_indices = frame_indices.get(session, set())
        if actual_indices != expected_indices:
            index_mismatches[session] = {
                "missing_indices": sorted(expected_indices - actual_indices)[:20],
                "unexpected_indices": sorted(actual_indices - expected_indices)[:20],
            }
    if index_mismatches:
        raise ValueError(f"manifest frame-index census mismatch: {index_mismatches}")


def _priority(row: dict[str, str]) -> str | None:
    status = row.get("selection_status", "").strip()
    if status == "review_selected":
        return "high"
    if status.startswith("not_selected"):
        return "low"
    return None  # overflow and other statuses are deliberately not sampled here


def _group_key(row: dict[str, str]) -> str:
    group = row.get("review_group_id", "").strip()
    if group:
        return group
    take = row.get("take_id", "").strip()
    if take:
        return f"take:{take}"
    return f"frame:{row['frame_index']}"


def _reviewed_ids(path: Path | None) -> set[tuple[str, str]]:
    if path is None:
        return set()
    rows = _read_csv(path)
    required = set(SOURCE_ID_FIELDS)
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"reviewed-ID CSV must include {sorted(required)}")
    return {_source_key(row) for row in rows if all(_source_key(row))}


def sample_manifest(
    rows: list[dict[str, str]],
    *,
    seed: int = 20261030,
    low_per_session: int = 60,
    high_per_session: int = 50,
    excluded_ids: set[tuple[str, str]] | None = None,
    expected_session_counts: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    """Sample one frame per review group from high/low strata in each session.

    Inclusion probabilities account for group sampling and the uniformly chosen
    frame inside each sampled group.  Groups, not adjacent frames, are the
    sampling units for the audit. If the request exceeds the eligible group
    count, all available groups are selected and the stratum is marked as a
    group census. An empty stratum remains insufficient.
    """
    validate_manifest(rows, expected_session_counts=expected_session_counts)
    excluded_ids = excluded_ids or set()
    rng = random.Random(seed)
    sessions = sorted({row["session_id"] for row in rows})
    if not sessions:
        raise ValueError("manifest contains no capture sessions")

    sampled: list[dict[str, object]] = []
    for session in sessions:
        for priority, target in (("high", high_per_session), ("low", low_per_session)):
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                if row["session_id"] != session or _priority(row) != priority:
                    continue
                if _source_key(row) in excluded_ids:
                    continue
                groups[_group_key(row)].append(row)
            for group_rows in groups.values():
                group_rows.sort(key=lambda item: int(item["frame_index"]))
            group_ids = sorted(groups)
            actual_target = min(target, len(group_ids))
            selected_groups = sorted(rng.sample(group_ids, actual_target))
            group_probability = actual_target / len(group_ids) if group_ids else 0.0
            for group_id in selected_groups:
                candidates = groups[group_id]
                row = dict(rng.choice(candidates))
                frame_probability = group_probability / len(candidates)
                row.update({
                    "priority": priority,
                    "audit_group_id": group_id,
                    "population_groups": len(group_ids),
                    "population_frames": sum(len(group) for group in groups.values()),
                    "requested_groups": target,
                    "actual_groups": actual_target,
                    "sampled_groups": actual_target,
                    "census_of_groups": bool(group_ids) and actual_target == len(group_ids),
                    "insufficient_groups": not group_ids,
                    "group_inclusion_probability": group_probability,
                    "frame_inclusion_probability": frame_probability,
                    "sample_weight": 1.0 / frame_probability,
                })
                sampled.append(row)
    sampled.sort(key=lambda row: (str(row["session_id"]), str(row["priority"]), int(row["frame_index"])))
    return sampled


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_source(source: str, source_root: Path) -> Path:
    path = Path(source)
    return path if path.is_absolute() else source_root / path


def build_blind_audit(
    manifest_path: Path,
    output_dir: Path,
    *,
    source_root: Path = ROOT,
    reviewed_ids_path: Path | None = None,
    seed: int = 20261030,
    low_per_session: int = 60,
    high_per_session: int = 50,
    reviewer_count: int = 3,
    expected_session_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    """Create blind PNGs, the hidden map, and three empty reviewer templates."""
    if reviewer_count != 3:
        raise ValueError("the audit contract requires exactly three reviewers")
    rows = _read_csv(manifest_path)
    excluded_ids = _reviewed_ids(reviewed_ids_path)
    selected = sample_manifest(
        rows,
        seed=seed,
        low_per_session=low_per_session,
        high_per_session=high_per_session,
        excluded_ids=excluded_ids,
        expected_session_counts=expected_session_counts,
    )
    sessions = sorted({row["session_id"] for row in rows})
    stratum_counts: dict[str, dict[str, object]] = {}
    for session in sessions:
        for priority, requested in (("high", high_per_session), ("low", low_per_session)):
            eligible = [
                row for row in rows
                if row["session_id"] == session and _priority(row) == priority
                and _source_key(row) not in excluded_ids
            ]
            eligible_groups = {_group_key(row) for row in eligible}
            actual = min(requested, len(eligible_groups))
            stratum_counts[f"{session}:{priority}"] = {
                "requested_groups": requested,
                "eligible_groups": len(eligible_groups),
                "actual_groups": actual,
                "census_of_groups": bool(eligible_groups) and actual == len(eligible_groups),
                "insufficient_groups": not eligible_groups,
            }
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    packet: list[dict[str, object]] = []
    mapping: list[dict[str, object]] = []
    templates: list[dict[str, object]] = []

    for row in selected:
        source = _resolve_source(str(row["source_path"]), source_root)
        if not source.is_file():
            raise FileNotFoundError(f"source image is missing: {source}")
        actual_hash = _sha256(source)
        expected_hash = str(row["source_sha256"]).lower()
        if actual_hash != expected_hash:
            raise ValueError(f"source checksum mismatch: {row['session_id']}:{row['frame_index']}")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"source image cannot be decoded: {source}")
        blind_id = hashlib.sha256(
            f"{seed}:{row['session_id']}:{row['frame_index']}:{expected_hash}".encode()
        ).hexdigest()[:16]
        destination = image_dir / f"{blind_id}.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"failed to write blind image: {destination}")
        packet.append({"blind_id": blind_id, "image_path": f"images/{blind_id}.png"})
        mapping.append({
            "blind_id": blind_id,
            "session_id": row["session_id"],
            "frame_index": row["frame_index"],
            "source_path": row["source_path"],
            "source_sha256": expected_hash,
            "priority": row["priority"],
            "selection_status": row["selection_status"],
            "review_group_id": row["audit_group_id"],
            "population_groups": row["population_groups"],
            "population_frames": row["population_frames"],
            "requested_groups": row["requested_groups"],
            "actual_groups": row["actual_groups"],
            "sampled_groups": row["sampled_groups"],
            "census_of_groups": row["census_of_groups"],
            "insufficient_groups": row["insufficient_groups"],
            "group_inclusion_probability": row["group_inclusion_probability"],
            "frame_inclusion_probability": row["frame_inclusion_probability"],
            "sample_weight": row["sample_weight"],
        })
        for reviewer_num in range(1, reviewer_count + 1):
            templates.append({
                "reviewer_id": f"reviewer_{reviewer_num}",
                "blind_id": blind_id,
                "useful": "",
                "visible_tags": "",
                "confidence": "",
                "notes": "",
            })

    # Shuffle presentation order with a recorded, reproducible seed.  The
    # private mapping remains in source order, while all reviewers get the same
    # blinded order.
    packet_order = random.Random(seed ^ 0xA11D17).sample(
        [row["blind_id"] for row in packet], k=len(packet)
    )
    packet_by_id = {str(row["blind_id"]): row for row in packet}
    templates_by_reviewer: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in templates:
        templates_by_reviewer[str(row["reviewer_id"])][str(row["blind_id"])] = row
    packet = [packet_by_id[blind_id] for blind_id in packet_order]
    _write_csv(output_dir / "blind_packet.csv", packet, PUBLIC_PACKET_FIELDS)
    mapping_path = output_dir / "audit_mapping.private.csv"
    _write_csv(mapping_path, mapping, MAPPING_FIELDS)
    try:
        mapping_path.chmod(0o600)
    except OSError:
        pass
    for reviewer_num in range(1, reviewer_count + 1):
        reviewer_rows = [
            templates_by_reviewer[f"reviewer_{reviewer_num}"][blind_id]
            for blind_id in packet_order
        ]
        _write_csv(output_dir / f"labels_reviewer_{reviewer_num}.csv", reviewer_rows, REVIEW_FIELDS)

    summary = {
        "schema_version": 1,
        "status": "awaiting_blind_review",
        "seed": seed,
        "source_frames": len(rows),
        "sample_frames": len(selected),
        "reviewers_required": reviewer_count,
        "priority_counts": {
            priority: sum(row["priority"] == priority for row in selected)
            for priority in ("high", "low")
        },
        "sessions": {
            session: {
                priority: sum(row["session_id"] == session and row["priority"] == priority for row in selected)
                for priority in ("high", "low")
            }
            for session in sorted({str(row["session_id"]) for row in selected})
        },
        "strata": stratum_counts,
        "sample_coverage_status": (
            "insufficient_groups" if any(value["insufficient_groups"] for value in stratum_counts.values())
            else "includes_group_census" if any(value["census_of_groups"] for value in stratum_counts.values())
            else "complete"
        ),
        "mapping_file": mapping_path.name,
        "blind_packet_fields": list(PUBLIC_PACKET_FIELDS),
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _binomial_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided exact binomial upper confidence limit, for small audit samples."""
    if n <= 0:
        return 1.0
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n")
    if k == n:
        return 1.0

    def cdf(probability: float) -> float:
        return sum(
            math.comb(n, index) * probability**index * (1 - probability) ** (n - index)
            for index in range(k + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _binomial_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided exact binomial lower confidence limit."""
    if n <= 0 or k <= 0:
        return 0.0
    if k == n:
        return alpha ** (1 / n)
    # P_p(X >= k) is monotone increasing in p; find the value equal to alpha.
    def upper_tail(probability: float) -> float:
        return sum(
            math.comb(n, index) * probability**index * (1 - probability) ** (n - index)
            for index in range(k, n + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if upper_tail(middle) < alpha:
            low = middle
        else:
            high = middle
    return high


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _parse_tags(value: str) -> set[str]:
    return {tag.strip().lower() for tag in value.replace(",", ";").split(";") if tag.strip()}


def evaluate_reviews(
    mapping_path: Path,
    reviewer_paths: list[Path],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 20261030,
) -> dict[str, object]:
    """Adjudicate three blinded reviewer exports and score the ranking.

    Useful/uncertain votes are counted conservatively as possible misses in the
    low-priority group.  The high-low comparison uses an absolute rate difference
    with a review-group-stratified bootstrap; it never divides by the low rate.
    """
    if len(reviewer_paths) != 3:
        raise ValueError("exactly three reviewer CSV files are required")
    mapping = _read_csv(mapping_path)
    blind_ids = [row.get("blind_id", "") for row in mapping]
    if len(set(blind_ids)) != len(blind_ids) or not all(blind_ids):
        raise ValueError("mapping has duplicate or empty blind IDs")
    mapping_by_id = {row["blind_id"]: row for row in mapping}
    responses: dict[str, list[dict[str, str]]] = defaultdict(list)
    reviewer_ids: set[str] = set()
    expected = set(mapping_by_id)
    for reviewer_path in reviewer_paths:
        rows = _read_csv(reviewer_path)
        if not rows or not set(REVIEW_FIELDS).issubset(rows[0]):
            raise ValueError(f"reviewer CSV is missing fields: {reviewer_path}")
        file_reviewer_ids = {row.get("reviewer_id", "").strip() for row in rows}
        if len(file_reviewer_ids) != 1 or not next(iter(file_reviewer_ids)):
            raise ValueError(f"reviewer CSV must contain one reviewer_id: {reviewer_path}")
        reviewer_id = next(iter(file_reviewer_ids))
        if reviewer_id in reviewer_ids:
            raise ValueError(f"reviewer_id appears in multiple files: {reviewer_id}")
        reviewer_ids.add(reviewer_id)
        ids = [row.get("blind_id", "").strip() for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate blind ID in reviewer file: {reviewer_path}")
        if set(ids) != expected:
            raise ValueError(f"reviewer IDs do not match mapping: {reviewer_path}")
        for row in rows:
            useful = row.get("useful", "").strip().lower()
            if useful not in VALID_USEFUL:
                raise ValueError(f"invalid useful label {useful!r} for {row.get('blind_id')}")
            row["useful"] = useful
            row["reviewer_id"] = reviewer_id
            responses[row["blind_id"]].append(row)
    if len(reviewer_ids) != 3:
        raise ValueError("three distinct reviewers are required")

    evaluated: list[dict[str, object]] = []
    agreement_pairs = 0
    agreement_matches = 0
    tag_counts: CounterLike = defaultdict(int)
    for blind_id in sorted(expected):
        votes = responses[blind_id]
        labels = [row["useful"] for row in votes]
        yes_count = labels.count("yes")
        no_count = labels.count("no")
        uncertain_count = labels.count("uncertain")
        if yes_count >= 2:
            consensus = "yes"
        elif no_count >= 2:
            consensus = "no"
        else:
            consensus = "uncertain"
        possible_useful = any(label in {"yes", "uncertain"} for label in labels)
        tags: set[str] = set()
        for vote in votes:
            tags.update(_parse_tags(vote.get("visible_tags", "")))
        for tag in tags:
            if sum(tag in _parse_tags(vote.get("visible_tags", "")) for vote in votes) >= 2:
                tag_counts[tag] += 1
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                agreement_pairs += 1
                agreement_matches += int(labels[left] == labels[right])
        mapping_row = mapping_by_id[blind_id]
        evaluated.append({
            "blind_id": blind_id,
            "session_id": mapping_row["session_id"],
            "priority": mapping_row["priority"],
            "review_group_id": mapping_row["review_group_id"],
            "requested_groups": int(mapping_row["requested_groups"]),
            "actual_groups": int(mapping_row["actual_groups"]),
            "census_of_groups": mapping_row["census_of_groups"].strip().lower() == "true",
            "insufficient_groups": mapping_row["insufficient_groups"].strip().lower() == "true",
            "consensus_useful": consensus,
            "possible_useful": possible_useful,
            "yes_votes": yes_count,
            "no_votes": no_count,
            "uncertain_votes": uncertain_count,
            "tags": sorted(tag for tag in tags if sum(tag in _parse_tags(vote.get("visible_tags", "")) for vote in votes) >= 2),
            "sample_weight": float(mapping_row["sample_weight"]),
        })

    sessions = sorted({str(row["session_id"]) for row in mapping})
    by_stratum: dict[str, dict[str, object]] = {}
    weighted_possible_rates: dict[tuple[str, str], float] = {}
    low_ucb: dict[str, float] = {}
    for session in sessions:
        for priority in ("high", "low"):
            sample = [row for row in evaluated if row["session_id"] == session and row["priority"] == priority]
            if not sample:
                continue
            yes = sum(row["consensus_useful"] == "yes" for row in sample)
            possible = sum(bool(row["possible_useful"]) for row in sample)
            count = len(sample)
            rate = yes / count
            possible_rate = possible / count
            group_count = len({row["review_group_id"] for row in sample})
            by_stratum[f"{session}:{priority}"] = {
                "sample_frames": count,
                "sample_groups": group_count,
                "requested_groups": int(sample[0]["requested_groups"]),
                "actual_groups": int(sample[0]["actual_groups"]),
                "census_of_groups": any(bool(row["census_of_groups"]) for row in sample),
                "insufficient_groups": any(bool(row["insufficient_groups"]) for row in sample),
                "consensus_useful_yes": yes,
                "possible_useful_or_uncertain": possible,
                "consensus_useful_rate": rate,
                "possible_useful_rate": possible_rate,
                "sample_weight_sum": sum(float(row["sample_weight"]) for row in sample),
                "weighted_consensus_useful_rate": (
                    sum(float(row["sample_weight"]) for row in sample if row["consensus_useful"] == "yes")
                    / sum(float(row["sample_weight"]) for row in sample)
                ),
                "weighted_possible_useful_rate": (
                    sum(float(row["sample_weight"]) for row in sample if row["possible_useful"])
                    / sum(float(row["sample_weight"]) for row in sample)
                ),
                "one_sided_95_upper_possible_useful_rate": _binomial_upper(possible, count),
                "upper_bound_note": "exact unweighted frame bound; use weighted rate for population estimate",
                "tags_positive_in_2_of_3": {
                    tag: sum(tag in row["tags"] for row in sample)
                    for tag in sorted(tag_counts)
                },
            }
            weighted_possible_rates[(session, priority)] = float(by_stratum[f"{session}:{priority}"]["weighted_possible_useful_rate"])
            if priority == "low":
                low_ucb[session] = _binomial_upper(possible, count)

    # Bootstrap the absolute, inverse-probability-weighted high-minus-low rate
    # by review group within each session. This remains defined if low is zero.
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    if bootstrap_samples > 0 and all((session, priority) in weighted_possible_rates for session in sessions for priority in ("high", "low")):
        for _ in range(bootstrap_samples):
            differences_by_session = []
            for session in sessions:
                rates = []
                for priority in ("high", "low"):
                    sample = [row for row in evaluated if row["session_id"] == session and row["priority"] == priority]
                    draw = rng.integers(0, len(sample), size=len(sample))
                    drawn = [sample[int(index)] for index in draw]
                    weight_sum = sum(float(row["sample_weight"]) for row in drawn)
                    rates.append(
                        sum(float(row["sample_weight"]) for row in drawn if row["possible_useful"])
                        / weight_sum
                    )
                differences_by_session.append(rates[0] - rates[1])
            # Sessions receive equal weight here because the sample is explicitly
            # session-stratified; report session-specific rates above.
            differences.append(float(np.mean(differences_by_session)))
    diff_ci = {
        "lower_95": _quantile(differences, 0.025) if differences else None,
        "upper_95": _quantile(differences, 0.975) if differences else None,
            "definition": "cluster bootstrap of inverse-probability-weighted absolute (high - low) possible-useful rate, within session",
        "required_lower_95": 0.20,
        "required_lower_95_unit": "absolute rate difference (20 percentage points)",
        "samples": bootstrap_samples if differences else 0,
    }
    agreement = agreement_matches / agreement_pairs if agreement_pairs else 0.0
    rare_tags = {
        tag: {
            "positive_frames": count,
            "support_frames_reviewed": len(evaluated),
            "status": "supported" if count >= 50 else "insufficient_support",
        }
        for tag, count in sorted(tag_counts.items())
    }
    complete_strata = all(f"{session}:{priority}" in by_stratum for session in sessions for priority in ("high", "low"))
    adequate_group_coverage = complete_strata and all(
        not bool(stratum["insufficient_groups"]) for stratum in by_stratum.values()
    )
    equal_weight_low_strata = all(
        len({round(float(row["sample_weight"]), 12) for row in evaluated
             if row["session_id"] == session and row["priority"] == "low"}) <= 1
        for session in sessions
    )
    low_gate = bool(low_ucb) and all(value <= 0.05 for value in low_ucb.values()) and equal_weight_low_strata
    enrichment_gate = diff_ci["lower_95"] is not None and diff_ci["lower_95"] > 0.20
    if not adequate_group_coverage or agreement < 0.80:
        status = "inconclusive"
    elif low_gate and enrichment_gate:
        status = "pass"
    elif any(value > 0.05 for value in low_ucb.values()) or (diff_ci["upper_95"] is not None and diff_ci["upper_95"] <= 0):
        status = "fail"
    else:
        status = "inconclusive"
    return {
        "schema_version": 1,
        "status": status,
        "semantic_status": "three-agent image-level consensus; not pixel ground truth",
        "reviewers": sorted(reviewer_ids),
        "frames_evaluated": len(evaluated),
        "reviewer_useful_label_pairwise_agreement": agreement,
        "strata": by_stratum,
        "low_priority_miss_upper_bounds_by_session": low_ucb,
        "high_minus_low_absolute_rate_difference_ci": diff_ci,
        "rare_category_support": rare_tags,
        "gates": {
            "low_priority_possible_useful_upper95_at_most_5pct_each_session": low_gate,
            "low_priority_bound_equal_weights": equal_weight_low_strata,
            "high_minus_low_difference_lower95_above_20pp": enrichment_gate,
            "reviewer_agreement_at_least_0_80": agreement >= 0.80,
            "complete_session_priority_strata": complete_strata,
            "sample_group_coverage_adequate": adequate_group_coverage,
        },
        "limits": [
            "This validates ranking only on the sampled captured sessions.",
            "No frame is deleted and low priority is not a training negative.",
            "Rare-category claims require at least 50 consensus-positive frames.",
            "Masks need a separate pixel-level review and are not validated by image tags.",
        ],
        "reviewed_rows": evaluated,
    }


# CounterLike keeps annotations readable without making collections.Counter a
# runtime dependency in the public type surface.
CounterLike = dict[str, int]


def _sample_cli(args: argparse.Namespace) -> int:
    summary = build_blind_audit(
        args.manifest,
        args.output_dir,
        source_root=args.source_root,
        reviewed_ids_path=args.reviewed_ids,
        seed=args.seed,
        low_per_session=args.low_per_session,
        high_per_session=args.high_per_session,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _evaluate_cli(args: argparse.Namespace) -> int:
    report = evaluate_reviews(args.mapping, args.reviewers, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample_parser = subparsers.add_parser(
        "sample-audit",
        help="create a blinded, stratified audit packet",
        description=(
            "Create a blind audit packet after validating the complete project census: "
            "session IDs session_20261029_130013_453419 (2518 frames) and "
            "session_20261029_151126_089557 (13327 frames). The CLI has no census override."
        ),
    )
    sample_parser.add_argument("--manifest", type=Path, required=True)
    sample_parser.add_argument("--output-dir", type=Path, required=True)
    sample_parser.add_argument("--source-root", type=Path, default=ROOT)
    sample_parser.add_argument("--reviewed-ids", type=Path)
    sample_parser.add_argument("--seed", type=int, default=20261030)
    sample_parser.add_argument("--low-per-session", type=int, default=60)
    sample_parser.add_argument("--high-per-session", type=int, default=50)
    sample_parser.set_defaults(func=_sample_cli)

    evaluate_parser = subparsers.add_parser("evaluate-audit", help="score three independent review CSVs")
    evaluate_parser.add_argument("--mapping", type=Path, required=True)
    evaluate_parser.add_argument("--reviewer-csv", dest="reviewers", type=Path, action="append", required=True)
    evaluate_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluate_parser.add_argument("--seed", type=int, default=20261030)
    evaluate_parser.add_argument("--output", type=Path)
    evaluate_parser.set_defaults(func=_evaluate_cli)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
