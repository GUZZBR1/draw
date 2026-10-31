#!/usr/bin/env python3
"""Package the 19 triple-reviewed real-frame masks for an exploratory pilot.

The package records agent consensus as review evidence; it does not claim
human ground truth or complete labels. Unknown pixels (255) are preserved.
No holdout is created because the only candidate holdout masks failed review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SESSION = "session_20261029_151126_089557"
EXPECTED_IDS = {
    1, 47, 156, 1222, 1481, 1497, 1640, 4173, 5502, 5939,
    6375, 6523, 6729, 7464, 11228, 12290, 12725, 12802, 13093,
}
ALLOWED_VALUES = {0, 1, 2, 3, 255}
DEFAULT_CANDIDATES = ROOT / "captures/triage/pseudo_training_candidate/calibration_19_consensus_candidates.csv"
DEFAULT_SOURCE_REVIEW = ROOT / "captures/triage/pseudo_training_candidate/calibration_30_expert_source_review.csv"
DEFAULT_OUTPUT = ROOT / "captures/triage/training_ready/expert_consensus_20261030_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str, *, base: Path = ROOT) -> Path:
    path = Path(value)
    resolved = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"input path escapes repository: {resolved}")
    return resolved


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def build(candidates_path: Path, source_review_path: Path, output_dir: Path) -> dict[str, object]:
    candidates_path = candidates_path.resolve()
    source_review_path = source_review_path.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(ROOT / "captures/triage/training_ready"):
        raise ValueError("output must be a child of captures/triage/training_ready/")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    fields, candidates = read_csv(candidates_path)
    required = {"session_id", "frame_index", "source_path", "source_sha256", "mask_path", "mask_sha256", "width", "height"}
    if missing := required - set(fields):
        raise ValueError(f"candidate manifest is missing fields: {sorted(missing)}")
    candidate_by_id: dict[int, dict[str, str]] = {}
    for row in candidates:
        if row["session_id"] != SESSION:
            raise ValueError(f"unexpected session in candidate manifest: {row['session_id']}")
        frame_id = int(row["frame_index"])
        if frame_id in candidate_by_id:
            raise ValueError(f"duplicate candidate source ID: {SESSION}/{frame_id}")
        candidate_by_id[frame_id] = row
    if set(candidate_by_id) != EXPECTED_IDS:
        raise ValueError(
            f"candidate IDs differ from frozen triple-consensus set; "
            f"missing={sorted(EXPECTED_IDS-set(candidate_by_id))}, "
            f"extra={sorted(set(candidate_by_id)-EXPECTED_IDS)}"
        )

    review_fields, review_rows = read_csv(source_review_path)
    review_required = {"session_id", "frame_index", "reviewer_a", "reviewer_b", "source_review_status"}
    if missing := review_required - set(review_fields):
        raise ValueError(f"source review manifest is missing fields: {sorted(missing)}")
    source_reviews = {
        int(row["frame_index"]): row for row in review_rows
        if row["session_id"] == SESSION and int(row["frame_index"]) in EXPECTED_IDS
    }
    if set(source_reviews) != EXPECTED_IDS:
        raise ValueError("not all triple-consensus frames have two-reviewer blind source records")
    for frame_id, row in source_reviews.items():
        if row["reviewer_a"] != "vision_A" or row["reviewer_b"] != "vision_B" or row["source_review_status"] != "two_agent_source_consensus":
            raise ValueError(f"unexpected blind source review record for frame {frame_id}")

    image_dir = output_dir / "train" / "images"
    mask_dir = output_dir / "train" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    rows_out: list[dict[str, object]] = []
    totals: Counter[int] = Counter()
    try:
        for frame_id in sorted(EXPECTED_IDS):
            row = candidate_by_id[frame_id]
            source = repo_path(row["source_path"])
            mask_path = repo_path(row["mask_path"], base=candidates_path.parent)
            if not source.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"missing image or proposal for {SESSION}/{frame_id}")
            source_hash, mask_hash = sha256(source), sha256(mask_path)
            if source_hash != row["source_sha256"] or mask_hash != row["mask_sha256"]:
                raise ValueError(f"source/proposal hash mismatch for {SESSION}/{frame_id}")
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None or mask is None:
                raise ValueError(f"image or mask decode failed for {SESSION}/{frame_id}")
            height, width = image.shape[:2]
            if (width, height) != (1280, 720) or (width, height) != (int(row["width"]), int(row["height"])):
                raise ValueError(f"unexpected image resolution for {SESSION}/{frame_id}: {(width, height)}")
            if mask.ndim != 2 or mask.shape != (height, width):
                raise ValueError(f"mask must be single-channel and image-sized for {SESSION}/{frame_id}")
            labels, counts = np.unique(mask, return_counts=True)
            if invalid := sorted(set(map(int, labels)) - ALLOWED_VALUES):
                raise ValueError(f"invalid values {invalid} in {mask_path}")
            pixel_counts = {int(label): int(count) for label, count in zip(labels, counts)}
            totals.update(pixel_counts)

            stem = f"{SESSION}_frame_{frame_id:08d}"
            image_out = image_dir / f"{stem}{source.suffix.lower()}"
            mask_out = mask_dir / f"{stem}.png"
            shutil.copy2(source, image_out)
            shutil.copy2(mask_path, mask_out)
            blind = source_reviews[frame_id]
            rows_out.append({
                "session_id": SESSION,
                "frame_index": frame_id,
                "split": "train",
                "dataset_role": "exploratory_expert_consensus_train_pilot",
                "label_status": "expert_consensus_reviewed",
                "truth_status": "agent consensus; not human ground truth",
                "overlay_reviewers": "vision_A;vision_B;vision_C",
                "blind_source_reviewers": f"{blind['reviewer_a']};{blind['reviewer_b']}",
                "review_provenance": "transcribed from independent specialist reports; per-frame wall-clock timestamps were not supplied",
                "source_path": str(source.relative_to(ROOT)),
                "source_sha256": source_hash,
                "image_path": str(image_out.relative_to(output_dir)),
                "image_sha256": sha256(image_out),
                "proposal_mask_path": str(mask_path.relative_to(ROOT)),
                "mask_path": str(mask_out.relative_to(output_dir)),
                "mask_sha256": sha256(mask_out),
                "width": width,
                "height": height,
                "class_0_pixels": pixel_counts.get(0, 0),
                "class_1_pixels": pixel_counts.get(1, 0),
                "class_2_pixels": pixel_counts.get(2, 0),
                "class_3_pixels": pixel_counts.get(3, 0),
                "ignored_pixels_255": pixel_counts.get(255, 0),
            })

        if any(totals[class_id] == 0 for class_id in range(4)):
            raise ValueError(f"pilot is missing one or more class labels: {dict(totals)}")
        if totals[255] == 0:
            raise ValueError("expected preserved unknown pixels (255) in partial reviewed labels")
        samples_path = output_dir / "samples.csv"
        with samples_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
        metadata = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_kind": "expert_consensus_train_pilot",
            "semantic_status": "agent consensus; not human ground truth",
            "label_status": "19 masks accepted by independent overlay reviewers vision_A, vision_B, and vision_C",
            "review_evidence": {
                "blind_source_csv": str(source_review_path.relative_to(ROOT)),
                "blind_source_csv_sha256": sha256(source_review_path),
                "candidate_csv": str(candidates_path.relative_to(ROOT)),
                "candidate_csv_sha256": sha256(candidates_path),
                "overlay_reviewers": ["vision_A", "vision_B", "vision_C"],
                "per_frame_timestamps": "not supplied; not fabricated",
                "important_limit": "expert visual review is treated as validated by the user for this workflow; it is not human ground truth",
            },
            "split": {
                "train_session_id": SESSION,
                "train_resolution": "1280x720",
                "frames": len(rows_out),
                "holdout": None,
                "holdout_note": "No holdout masks were approved; do not report real-session evaluation from this pilot.",
            },
            "classes": {"0": "background", "1": "visible paper", "2": "visible hand", "3": "user-added visible ink"},
            "mask_values": [0, 1, 2, 3, 255],
            "ignore_index": 255,
            "unknown_pixels_preserved": True,
            "class_pixel_counts": {str(label): totals.get(label, 0) for label in (0, 1, 2, 3, 255)},
            "sources_unchanged": True,
            "holdout_review_disposition": "20 candidate frames reviewed by A/B/C; none had dual-review acceptance at overlay stage; v3 also had zero accepts from reviewer B",
        }
        (output_dir / "dataset_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return metadata
    except Exception:
        shutil.rmtree(output_dir)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--source-review", type=Path, default=DEFAULT_SOURCE_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.candidates, args.source_review, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
