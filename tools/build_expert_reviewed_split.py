#!/usr/bin/env python3
"""Build a session-isolated dataset from independently reviewed mask records.

This command is deliberately fail-closed. It never promotes the existing
pseudo-label proposal without two native-resolution overlay reviews and a
third adjudication for risky or disputed cases. Its status means
"expert-consensus reviewed"; it does not mean human ground truth.

Review CSV schema (one row per reviewer and frame):
  reviewer_id, session_id, frame_index, phase, outcome, source_path,
  source_sha256, mask_path, mask_sha256, width, height, issues, confidence,
  notes, reviewed_at_utc, corrected_mask_path, corrected_mask_sha256,
  displayed_native_resolution

`phase` is blind_source, overlay_review, or adjudication. `outcome` is
accept, edit, reject, or uncertain. The split freeze JSON is created before
reviews with `--write-freeze-manifest`, explicit `--train-session` values,
and `--holdout-session`. Previously trained sessions are rejected as holdout;
use the resulting manifest unchanged for packaging.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "captures/triage/pseudo_training_candidate/samples.csv"
TRAIN_SESSION = "session_20261029_151126_089557"
USED_FOR_TRAINING_SESSIONS = {
    "session_20261029_130013_453419",
    TRAIN_SESSION,
}
ALLOWED_VALUES = {0, 1, 2, 3, 255}
RISK_ISSUES = {
    "user_added_ink", "printed_text", "hand_over_ink", "occlusion",
    "uncertain_boundary", "printed_vs_handwritten",
}
REVIEW_FIELDS = {
    "reviewer_id", "session_id", "frame_index", "phase", "outcome",
    "source_path", "source_sha256", "mask_path", "mask_sha256", "width",
    "height", "issues", "confidence", "notes", "reviewed_at_utc",
    "corrected_mask_path", "corrected_mask_sha256",
    "displayed_native_resolution",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def key_of(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["session_id"]), int(row["frame_index"])


def resolve_repo_path(value: str, *, candidate_root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        candidates = []
        if candidate_root is not None:
            candidates.append(candidate_root / path)
        candidates.append(ROOT / path)
        resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[-1].resolve())
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"reviewed dataset paths must stay inside the repository: {resolved}")
    return resolved


def read_samples(path: Path) -> tuple[dict[tuple[str, int], dict[str, str]], Path, str]:
    path = path.resolve()
    fields, rows = csv_rows(path)
    required = {
        "session_id", "frame_index", "source_path", "source_sha256", "image_path",
        "mask_path", "mask_sha256", "width", "height", "visible_tags_hint_only",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(f"samples CSV is missing columns: {', '.join(sorted(missing))}")
    samples: dict[tuple[str, int], dict[str, str]] = {}
    session_dims: dict[str, tuple[int, int]] = {}
    for row in rows:
        key = key_of(row)
        if key in samples:
            raise ValueError(f"duplicate source ID in samples CSV: {key[0]} frame {key[1]}")
        source = resolve_repo_path(row["source_path"])
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"source frame cannot be decoded: {source}")
        height, width = image.shape[:2]
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise ValueError(f"samples dimensions disagree with source for {key}")
        if key[0] in session_dims and session_dims[key[0]] != (width, height):
            raise ValueError(f"session contains mixed source dimensions: {key[0]}")
        session_dims[key[0]] = (width, height)
        source_hash = sha256(source)
        if source_hash != row["source_sha256"]:
            raise ValueError(f"source SHA-256 mismatch for {key}")
        mask = resolve_repo_path(row["mask_path"], candidate_root=path.parent)
        if not mask.is_file() or sha256(mask) != row["mask_sha256"]:
            raise ValueError(f"source mask missing or SHA-256 mismatch for {key}")
        row["_resolved_source"] = str(source)
        row["_resolved_mask"] = str(mask)
        samples[key] = row
    if not samples:
        raise ValueError("samples CSV contains no source IDs")
    return samples, path.parent, sha256(path)


def make_freeze(
    samples: dict[tuple[str, int], dict[str, str]],
    sample_hash: str,
    train_sessions: set[str],
    holdout_session: str,
    previously_trained_sessions: set[str] | None = None,
) -> dict[str, Any]:
    previously_trained_sessions = set(previously_trained_sessions or ())
    used_for_training = USED_FOR_TRAINING_SESSIONS | previously_trained_sessions
    all_sessions = {session for session, _ in samples}
    if not train_sessions or holdout_session not in all_sessions:
        raise ValueError("split requires training sessions and a holdout session present in samples")
    if holdout_session in used_for_training:
        raise ValueError(
            f"session {holdout_session} has already been used for training; "
            "it cannot be an independent holdout"
        )
    if train_sessions & {holdout_session} or train_sessions | {holdout_session} != all_sessions:
        raise ValueError("explicit train/holdout sessions must be disjoint and cover the sample sessions exactly")
    train_ids = sorted(
        [{"session_id": s, "frame_index": i} for s, i in samples if s in train_sessions],
        key=lambda row: (row["session_id"], row["frame_index"]),
    )
    holdout_ids = sorted(
        [{"session_id": s, "frame_index": i} for s, i in samples if s == holdout_session],
        key=lambda row: (row["session_id"], row["frame_index"]),
    )
    train_keys = [(row["session_id"], row["frame_index"]) for row in train_ids]
    calibration: dict[tuple[str, int], None] = {}
    # Favor the risky visual categories without exposing the hints to reviewers.
    for issue in ("user_added_ink", "printed_text", "hand_over_ink", "occlusion", "uncertain_boundary", "hand_on_paper", "pen"):
        candidate = next(
            (key for key in train_keys if issue in parse_tags(samples[key].get("visible_tags_hint_only", ""))),
            None,
        )
        if candidate is not None:
            calibration[candidate] = None
    remainder = [key for key in train_keys if key not in calibration]
    remainder.sort(key=lambda key: hashlib.sha256(f"{key[0]}:{key[1]}:{sample_hash}".encode()).hexdigest())
    for key in remainder[:30 - len(calibration)]:
        calibration[key] = None
    calibration_ids = [
        {"session_id": session, "frame_index": index}
        for session, index in sorted(calibration, key=lambda item: item[1])
    ]
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_samples_sha256": sample_hash,
        "train_session_ids": sorted(train_sessions),
        "previously_trained_session_ids": sorted(previously_trained_sessions),
        "holdout_session_id": holdout_session,
        "train_source_ids": train_ids,
        "holdout_source_ids": holdout_ids,
        "calibration_source_ids": calibration_ids,
        "split_label": "near_domain_session_holdout",
        "notes": "Frozen before mask review/training. All sessions previously used for training are forbidden as holdout.",
    }


def validate_freeze(freeze_path: Path, samples: dict[tuple[str, int], dict[str, str]], sample_hash: str) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != 2:
        raise ValueError("split freeze manifest schema_version must be 2")
    if freeze.get("source_samples_sha256") != sample_hash:
        raise ValueError("split freeze manifest was created from a different samples.csv")
    train_sessions = set(freeze.get("train_session_ids", []))
    previously_trained_sessions = set(freeze.get("previously_trained_session_ids", []))
    used_for_training = USED_FOR_TRAINING_SESSIONS | previously_trained_sessions
    holdout_session = freeze.get("holdout_session_id")
    all_sessions = {session for session, _ in samples}
    if holdout_session in used_for_training:
        raise ValueError("frozen holdout session was already used for training")
    if not train_sessions or train_sessions & {holdout_session} or train_sessions | {holdout_session} != all_sessions:
        raise ValueError("frozen train/holdout sessions do not exactly partition the source sessions")
    train = {(str(row["session_id"]), int(row["frame_index"])) for row in freeze.get("train_source_ids", [])}
    holdout = {(str(row["session_id"]), int(row["frame_index"])) for row in freeze.get("holdout_source_ids", [])}
    calibration = {(str(row["session_id"]), int(row["frame_index"])) for row in freeze.get("calibration_source_ids", [])}
    if not isinstance(holdout_session, str) or not train or not holdout or train & holdout:
        raise ValueError("freeze manifest must have non-empty, disjoint train and holdout source ID lists")
    if train != {key for key in samples if key[0] in train_sessions}:
        raise ValueError("frozen training IDs must exactly match selected frames from training sessions")
    if holdout != {key for key in samples if key[0] == holdout_session}:
        raise ValueError("frozen holdout IDs must exactly match selected frames from the holdout session")
    if not 20 <= len(calibration) <= 30 or not calibration.issubset(train):
        raise ValueError("freeze manifest must contain 20–30 calibration IDs from the training session only")
    return train, holdout


def parse_timestamp(value: str, *, key: tuple[str, int], reviewer: str, phase: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid reviewed_at_utc for {key}/{reviewer}/{phase}: {value!r}") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"reviewed_at_utc must include a timezone for {key}/{reviewer}/{phase}")
    return timestamp


def parse_tags(value: str) -> set[str]:
    return {part.strip().lower() for part in value.replace(",", ";").split(";") if part.strip()}


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def read_reviews(path: Path, samples: dict[tuple[str, int], dict[str, str]], candidate_root: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    fields, rows = csv_rows(path.resolve())
    missing = REVIEW_FIELDS - set(fields)
    if missing:
        raise ValueError(f"review CSV is missing columns: {', '.join(sorted(missing))}")
    reviews: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, int, str, str]] = set()
    for row in rows:
        key = key_of(row)
        if key not in samples:
            raise ValueError(f"review CSV contains source ID not present in frozen samples: {key}")
        reviewer = row["reviewer_id"].strip()
        if not reviewer:
            raise ValueError(f"blank reviewer_id for {key}")
        stage = row["phase"].strip()
        if stage not in {"blind_source", "overlay_review", "adjudication"}:
            raise ValueError(f"invalid review phase {stage!r} for {key}")
        outcome = row["outcome"].strip()
        if outcome not in {"accept", "edit", "reject", "uncertain"}:
            raise ValueError(f"invalid review outcome {outcome!r} for {key}")
        if not row["reviewed_at_utc"].strip():
            raise ValueError(f"missing reviewed_at_utc for {key}/{reviewer}/{stage}")
        if not truthy(row["displayed_native_resolution"]):
            raise ValueError(f"review was not recorded at native resolution for {key}/{reviewer}/{stage}")
        sample = samples[key]
        source = resolve_repo_path(row["source_path"])
        base_mask = resolve_repo_path(row["mask_path"], candidate_root=candidate_root)
        if source != Path(sample["_resolved_source"]) or row["source_sha256"].strip() != sample["source_sha256"]:
            raise ValueError(f"review source ID/path/hash mismatch for {key}/{reviewer}")
        is_adjudication = stage == "adjudication"
        allowed_review_masks = {Path(sample["_resolved_mask"])}
        # An adjudicator may explicitly accept one corrected candidate from an
        # earlier overlay reviewer; all other phases must remain tied to the
        # original proposal being reviewed.
        for other in reviews.get(key, []):
            if other.get("phase") == "overlay_review" and other.get("_final_mask"):
                allowed_review_masks.add(Path(other["_final_mask"]))
        if (not is_adjudication and base_mask != Path(sample["_resolved_mask"])) or base_mask not in allowed_review_masks:
            raise ValueError(f"review mask path mismatch for {key}/{reviewer}/{stage}")
        if row["mask_sha256"].strip() != sha256(base_mask):
            raise ValueError(f"review proposal path/hash mismatch for {key}/{reviewer}")
        if (int(row["width"]), int(row["height"])) != (
            int(sample["width"]), int(sample["height"])
        ):
            raise ValueError(f"review resolution mismatch for {key}/{reviewer}")
        duplicate_key = (key[0], str(key[1]), reviewer, stage)
        if duplicate_key in seen:
            raise ValueError(f"duplicate review event for {key}/{reviewer}/{stage}")
        seen.add(duplicate_key)
        row["_resolved_source"] = str(source)
        row["_resolved_mask"] = str(base_mask)
        if outcome == "edit":
            corrected_path = row["corrected_mask_path"].strip()
            corrected_hash = row["corrected_mask_sha256"].strip()
            if not corrected_path or not corrected_hash:
                raise ValueError(f"edited review needs corrected_mask_path and corrected_mask_sha256: {key}/{reviewer}")
            edited_mask = resolve_repo_path(corrected_path, candidate_root=candidate_root)
            if not edited_mask.is_file() or sha256(edited_mask) != corrected_hash:
                raise ValueError(f"corrected mask missing or checksum mismatch: {key}/{reviewer}")
            row["_final_mask"] = str(edited_mask)
            row["_final_mask_sha256"] = corrected_hash
        elif row["corrected_mask_path"].strip() or row["corrected_mask_sha256"].strip():
            raise ValueError(f"non-edit review must not specify a corrected mask: {key}/{reviewer}")
        else:
            row["_final_mask"] = str(base_mask)
            row["_final_mask_sha256"] = sha256(base_mask)
        reviews[key].append(row)
    return reviews


def load_final_mask(mask_path: Path, key: tuple[str, int], sample: dict[str, str]) -> tuple[np.ndarray, dict[int, int]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise ValueError(f"final mask must decode as single-channel: {key} {mask_path}")
    height, width = int(sample["height"]), int(sample["width"])
    if mask.shape != (height, width):
        raise ValueError(f"final mask dimension mismatch for {key}: {mask.shape[::-1]} != {(width, height)}")
    values, counts = np.unique(mask, return_counts=True)
    invalid = sorted(set(int(value) for value in values) - ALLOWED_VALUES)
    if invalid:
        raise ValueError(f"final mask contains invalid values {invalid}: {mask_path}")
    pixel_counts = {int(value): int(count) for value, count in zip(values, counts)}
    if sum(value for label, value in pixel_counts.items() if label != 255) == 0:
        raise ValueError(f"final mask is entirely unknown (255): {key}")
    return mask, pixel_counts


def approved_mask(key: tuple[str, int], sample: dict[str, str], review_rows: list[dict[str, str]]) -> tuple[Path, str, list[dict[str, str]]] | None:
    blind = [row for row in review_rows if row["phase"] == "blind_source"]
    overlay = [row for row in review_rows if row["phase"] == "overlay_review"]
    if len({row["reviewer_id"].strip() for row in blind}) < 2:
        return None
    if len({row["reviewer_id"].strip() for row in overlay}) < 2:
        return None
    overlay = sorted(overlay, key=lambda row: row["reviewer_id"].strip())
    blind_by_reviewer = {row["reviewer_id"].strip(): row for row in blind}
    for row in overlay:
        reviewer = row["reviewer_id"].strip()
        blind_row = blind_by_reviewer.get(reviewer)
        if blind_row is None:
            return None
        if parse_timestamp(blind_row["reviewed_at_utc"], key=key, reviewer=reviewer, phase="blind_source") >= parse_timestamp(row["reviewed_at_utc"], key=key, reviewer=reviewer, phase="overlay_review"):
            return None
    if any(row["outcome"] not in {"accept", "edit"} for row in overlay):
        return None
    if len({row["_final_mask_sha256"] for row in overlay}) > 1:
        disagreement = True
    else:
        disagreement = len({row["outcome"] for row in overlay}) > 1

    tags = parse_tags(sample.get("visible_tags_hint_only", ""))
    review_issues = set().union(*(parse_tags(row.get("issues", "")) for row in blind + overlay))
    needs_adjudication = bool((tags | review_issues) & RISK_ISSUES) or disagreement
    adjudications = [row for row in review_rows if row["phase"] == "adjudication"]
    if needs_adjudication:
        if len(adjudications) != 1:
            return None
        adjudicator = adjudications[0]
        if adjudicator["reviewer_id"].strip() in {row["reviewer_id"].strip() for row in overlay}:
            return None
        if adjudicator["outcome"] not in {"accept", "edit"}:
            return None
        # The adjudicator's viewed mask_path is the explicitly selected final
        # candidate for accept; an edit selects corrected_mask_path/hash.
        final_path = Path(adjudicator["_final_mask"])
        final_hash = adjudicator["_final_mask_sha256"]
    else:
        if adjudications:
            if len(adjudications) != 1 or adjudications[0]["reviewer_id"].strip() in {row["reviewer_id"].strip() for row in overlay}:
                return None
            if adjudications[0]["outcome"] not in {"accept", "edit"}:
                return None
            final_path = Path(adjudications[0]["_final_mask"])
            final_hash = adjudications[0]["_final_mask_sha256"]
        else:
            final_path = Path(overlay[0]["_final_mask"])
            final_hash = overlay[0]["_final_mask_sha256"]
            if any(Path(row["_final_mask"]) != final_path for row in overlay):
                return None
    if not final_path.is_file() or sha256(final_path) != final_hash:
        raise ValueError(f"selected final mask hash mismatch for {key}")
    return final_path, final_hash, overlay + adjudications


def copy_split(
    destination: Path,
    split_name: str,
    keys: set[tuple[str, int]],
    samples: dict[tuple[str, int], dict[str, str]],
    reviews: dict[tuple[str, int], list[dict[str, str]]],
    min_coverage: float,
) -> tuple[list[dict[str, Any]], Counter[int]]:
    image_dir = destination / split_name / "images"
    mask_dir = destination / split_name / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    accepted_rows: list[dict[str, Any]] = []
    totals: Counter[int] = Counter()
    excluded: list[tuple[str, int]] = []

    for key in sorted(keys, key=lambda item: (item[0], item[1])):
        sample = samples[key]
        review_rows = reviews.get(key, [])
        decision = approved_mask(key, sample, review_rows)
        if decision is None:
            excluded.append(key)
            continue
        final_mask_path, final_hash, accepted_reviews = decision
        _, pixel_counts = load_final_mask(final_mask_path, key, sample)
        source_path = Path(sample["_resolved_source"])
        stem = f"{key[0]}_frame_{key[1]:08d}"
        image_out = image_dir / f"{stem}{source_path.suffix.lower()}"
        mask_out = mask_dir / f"{stem}.png"
        shutil.copy2(source_path, image_out)
        shutil.copy2(final_mask_path, mask_out)
        totals.update(pixel_counts)
        accepted_rows.append({
            "session_id": key[0],
            "frame_index": key[1],
            "split": split_name,
            "label_status": "expert_consensus_reviewed",
            "truth_status": "agent consensus; not human ground truth",
            "source_path": str(source_path.relative_to(ROOT)),
            "source_sha256": sample["source_sha256"],
            "image_path": str(image_out.relative_to(destination)),
            "image_sha256": sha256(image_out),
            "proposal_mask_path": sample["mask_path"],
            "accepted_mask_path": str(final_mask_path.relative_to(ROOT)),
            "mask_path": str(mask_out.relative_to(destination)),
            "mask_sha256": sha256(mask_out),
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "class_0_pixels": pixel_counts.get(0, 0),
            "class_1_pixels": pixel_counts.get(1, 0),
            "class_2_pixels": pixel_counts.get(2, 0),
            "class_3_pixels": pixel_counts.get(3, 0),
            "ignored_pixels_255": pixel_counts.get(255, 0),
            "reviewer_ids": ";".join(row["reviewer_id"].strip() for row in accepted_reviews),
            "review_rows": json.dumps([
                {field: row[field] for field in REVIEW_FIELDS}
                for row in accepted_reviews
            ], ensure_ascii=False, sort_keys=True),
        })

    if not keys or not accepted_rows:
        raise ValueError(f"no frames passed expert-review gates for split {split_name}")
    coverage = len(accepted_rows) / len(keys)
    if coverage < min_coverage:
        raise ValueError(f"{split_name} accepted coverage {coverage:.3f} is below required {min_coverage:.3f}; excluded {len(excluded)} frames")
    for class_id in range(4):
        if totals[class_id] <= 0:
            raise ValueError(f"split {split_name} has no reviewed support for class {class_id}")
    samples_path = destination / f"{split_name}_samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(accepted_rows[0]))
        writer.writeheader()
        writer.writerows(accepted_rows)
    return accepted_rows, totals


def build(args: argparse.Namespace) -> dict[str, Any]:
    samples, candidate_root, sample_hash = read_samples(args.samples)
    if args.write_freeze_manifest:
        if not args.holdout_session or not args.train_session:
            raise ValueError("freeze creation requires --train-session (repeatable) and --holdout-session")
        path = args.write_freeze_manifest.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("freeze manifest must be written inside the repository")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen split manifest: {path}")
        freeze = make_freeze(
            samples,
            sample_hash,
            set(args.train_session),
            args.holdout_session,
            set(args.previously_trained_session or ()),
        )
        path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {
            "freeze_manifest": str(path.relative_to(ROOT)),
            "source_samples_sha256": sample_hash,
            "train_frames": len(freeze["train_source_ids"]),
            "holdout_frames": len(freeze["holdout_source_ids"]),
        }

    if not args.freeze_manifest:
        raise ValueError("--freeze-manifest is required for packaging; create it first with --write-freeze-manifest")
    train_ids, holdout_ids = validate_freeze(args.freeze_manifest.resolve(), samples, sample_hash)
    review_path = args.reviews.resolve()
    reviews = read_reviews(review_path, samples, candidate_root)
    out = args.output_dir.resolve()
    safe_root = (ROOT / "captures/triage/training_ready").resolve()
    if not out.is_relative_to(safe_root) or out == safe_root:
        raise ValueError(f"--output-dir must be a new child of {safe_root}")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output: {out}")
    out.mkdir(parents=True)
    try:
        train_rows, train_counts = copy_split(out, "train", train_ids, samples, reviews, args.min_coverage)
        holdout_rows, holdout_counts = copy_split(out, "holdout", holdout_ids, samples, reviews, args.min_coverage)
        metadata = {
            "schema_version": 1,
            "dataset_kind": "expert_consensus_reviewed_session_split",
            "semantic_status": "expert_consensus_reviewed; not human ground truth",
            "source_samples_csv": str(args.samples.resolve().relative_to(ROOT)),
            "source_samples_sha256": sample_hash,
            "review_csv": str(review_path.relative_to(ROOT)),
            "review_csv_sha256": sha256(review_path),
            "split_freeze_manifest": str(args.freeze_manifest.resolve().relative_to(ROOT)),
            "split_freeze_sha256": sha256(args.freeze_manifest.resolve()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "classes": {"0": "background", "1": "visible paper", "2": "visible hand", "3": "user-added visible ink"},
            "ignore_index": 255,
            "unknown_pixels_are_preserved": True,
            "train": {"session_ids": sorted({row["session_id"] for row in train_rows}), "frames": len(train_rows), "class_pixel_support": {str(i): train_counts[i] for i in range(4)}, "ignored_pixels": train_counts[255]},
            "holdout": {"session_id": args.freeze_manifest and json.loads(args.freeze_manifest.read_text(encoding="utf-8"))["holdout_session_id"], "frames": len(holdout_rows), "class_pixel_support": {str(i): holdout_counts[i] for i in range(4)}, "ignored_pixels": holdout_counts[255], "scope": "single-session holdout; selected/event-balanced frames do not estimate natural prevalence"},
            "excluded_unreviewed_or_rejected": {
                "train": len(train_ids) - len(train_rows),
                "holdout": len(holdout_ids) - len(holdout_rows),
            },
        }
        (out / "dataset_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return metadata
    except Exception:
        # Output is a newly created, unique run directory. Remove only the empty
        # or partial directory created by this invocation after a failed build.
        shutil.rmtree(out)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES, help="frozen pseudo-candidate samples.csv")
    parser.add_argument("--reviews", type=Path, help="per-review CSV using the documented schema; required for packaging")
    parser.add_argument("--freeze-manifest", type=Path, help="immutable split freeze JSON, required for packaging")
    parser.add_argument("--train-session", action="append", help="session already assigned to training; repeat once per session when freezing")
    parser.add_argument("--previously-trained-session", action="append", help="other session used by earlier training runs; repeat as needed when freezing")
    parser.add_argument("--holdout-session", help="new session reserved entirely for holdout; previously trained sessions are rejected")
    parser.add_argument("--write-freeze-manifest", type=Path, help="create a new split-freeze JSON from exact selected IDs and exit; refuses overwrite")
    parser.add_argument("--output-dir", type=Path, help="new child of captures/triage/training_ready/<run-id>")
    parser.add_argument("--min-coverage", type=float, default=1.0, help="minimum approved fraction in each frozen split (default: 1.0; fail closed)")
    args = parser.parse_args()
    if not 0 < args.min_coverage <= 1:
        parser.error("--min-coverage must be in (0, 1]")
    if args.write_freeze_manifest:
        try:
            print(json.dumps(build(args), indent=2, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if not args.reviews or not args.freeze_manifest or not args.output_dir:
        parser.error("packaging requires --reviews, --freeze-manifest, and --output-dir")
    try:
        print(json.dumps(build(args), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
