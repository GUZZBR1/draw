#!/usr/bin/env python3
"""Build a private, review-only candidate bundle from owner capture frames."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ALLOWED_MASK_VALUES = {0, 1, 2, 3, 255}
EXPECTED_SESSIONS = {
    "session_20261029_130013_453419": 2518,
    "session_20261031_125745_232414": 1,
    "session_20261031_130204_930563": 116,
    "session_20261031_131529_833914": 873,
    "session_20261031_132028_204773": 805,
    "session_background_20261031_125441_816941": 633,
    "session_background_live_20261031_125918": 112,
    "session_background_sampled_20261031_125519_372124": 28,
}
TRAINING_USED_SESSIONS = {"session_20261029_130013_453419"}
USER_PRIVATE_AUTHORIZATION = "user confirmed owner of depicted person/content and authorized private Kaggle upload on 2026-10-31"
MASK_CLASSES = {
    "0": "background",
    "1": "visible paper (including pre-existing print/text)",
    "2": "visible hand",
    "3": "user-added visible ink",
    "255": "uncertain / ignore; never background",
}
FRAME_FIELDS = [
    "frame_id", "image_path_in_archive", "session_id", "frame_index", "timestamp_ns",
    "take_id", "take_uid", "width", "height", "source_sha256", "source_manifest_path",
    "annotation_status", "annotation_authority", "mask_status", "pseudo_mask_path_in_archive",
    "pseudo_mask_sha256", "pseudo_mask_method", "pseudo_mask_version", "pseudo_mask_pixel_counts",
    "invalid_mask", "training_approved", "review_split", "source_split_role",
    "session_used_for_prior_training", "external_exact_duplicate_of", "previous_frame_id",
    "previous_frame_delta_ns", "next_frame_id", "next_frame_delta_ns", "derived_copy_count",
    "derived_copy_paths", "proposal_queue_status", "proposal_status_snapshot",
    "proposal_status_record", "proposal_mask_included", "scene_subject", "scene_action",
    "scene_tag_status", "scene_label_note", "capture_context_status", "source_license",
    "privacy_scope", "privacy_authorization_status",
]
EXCLUDED_FIELDS = [
    "frame_id", "session_id", "frame_index", "source_sha256", "annotation_status",
    "exclusion_reason", "capture_context_status", "external_exact_duplicate_of",
    "pseudo_mask_sha256", "pseudo_mask_method", "pseudo_mask_version", "pseudo_mask_status",
    "derived_copy_count", "derived_copy_paths", "previous_frame_id", "previous_frame_delta_ns",
    "next_frame_id", "next_frame_delta_ns", "proposal_queue_status", "proposal_status_snapshot",
    "proposal_status_record", "proposal_mask_included", "scene_subject", "scene_action",
    "scene_tag_status", "scene_label_note",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_source(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise SystemExit(f"Unsafe source path: {relative}")
    result = (root / rel).resolve()
    if not result.is_relative_to(root.resolve()):
        raise SystemExit(f"Source path escapes source root: {relative}")
    return result


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--derived-copies", type=Path, required=True)
    parser.add_argument("--reference-samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    crosswalk = read_csv(args.crosswalk)
    canonical_rows = read_csv(args.canonical_manifest)
    canonical = {r["frame_id"]: r for r in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise SystemExit("Canonical manifest has duplicate frame IDs")
    if len({r["source_frame_id"] for r in crosswalk}) != len(crosswalk):
        raise SystemExit("Crosswalk has duplicate frame IDs")
    if len(crosswalk) != sum(EXPECTED_SESSIONS.values()):
        raise SystemExit(f"Expected 5086 crosswalk rows, found {len(crosswalk)}")

    remote_rows = read_csv(args.reference_samples)
    remote_by_id: dict[str, dict[str, str]] = {}
    for r in remote_rows:
        if r.get("session_id") and r.get("frame_index"):
            rid = f"{r['session_id']}_F{int(r['frame_index']):08d}"
            if rid in remote_by_id:
                raise SystemExit(f"Duplicate ID in remote reference samples: {rid}")
            remote_by_id[rid] = r
    if len(remote_by_id) != len(remote_rows):
        raise SystemExit("Existing remote pilot sample IDs are missing")

    derived_by_id: dict[str, list[str]] = defaultdict(list)
    for item in read_csv(args.derived_copies):
        derived_path = safe_source(root, item["derived_copy_path"])
        if not derived_path.is_file():
            raise SystemExit(f"Missing derived-copy lineage file: {derived_path}")
        derived_by_id[item["canonical_frame_id"]].append(item["derived_copy_path"])

    # Per-session frame ordering is used to expose adjacent-frame leakage/lineage.
    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in crosswalk:
        by_session[item["session_id"]].append(item)
    neighbor_map: dict[str, tuple[str, str, str, str]] = {}
    for session, items in by_session.items():
        items.sort(key=lambda r: int(r["frame_index"]))
        for i, item in enumerate(items):
            prev = items[i - 1] if i else None
            nxt = items[i + 1] if i + 1 < len(items) else None
            pdelta = str(int(item["timestamp_ns"]) - int(prev["timestamp_ns"])) if prev else ""
            ndelta = str(int(nxt["timestamp_ns"]) - int(item["timestamp_ns"])) if nxt else ""
            neighbor_map[item["source_frame_id"]] = (
                prev["source_frame_id"] if prev else "", pdelta,
                nxt["source_frame_id"] if nxt else "", ndelta,
            )

    images_zip = out / "images.zip"
    masks_zip = out / "pseudo_masks.zip"
    frames: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    ids: set[str] = set()
    source_hashes: dict[str, str] = {}
    full_status_counts: Counter[str] = Counter()
    full_session_counts: Counter[str] = Counter()
    included_status_counts: Counter[str] = Counter()
    excluded_status_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    excluded_sessions: Counter[str] = Counter()
    exclusion_reasons: Counter[str] = Counter()
    external_duplicates: Counter[str] = Counter()
    duplicate_masks_verified = 0
    missing_masks = invalid_masks = 0
    frames_csv_cache: dict[str, tuple[str, dict[str, dict[str, str]]]] = {}
    pseudo_methods: Counter[str] = Counter()

    with zipfile.ZipFile(images_zip, "w", compression=zipfile.ZIP_STORED) as archive, zipfile.ZipFile(
        masks_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as mask_archive:
        for item in sorted(crosswalk, key=lambda r: (r["session_id"], int(r["frame_index"]))):
            frame_id, session = item["source_frame_id"], item["session_id"]
            if frame_id in ids:
                raise SystemExit(f"Duplicate frame ID: {frame_id}")
            ids.add(frame_id)
            if session not in EXPECTED_SESSIONS:
                raise SystemExit(f"Unexpected session: {session}")
            full_session_counts[session] += 1
            image_path = safe_source(root, item["source_path"])
            if not image_path.is_file():
                raise SystemExit(f"Missing original image: {image_path}")
            image_hash = sha256(image_path)
            if image_hash != item["source_sha256"] or image_hash in source_hashes:
                raise SystemExit(f"Crosswalk hash mismatch or internal duplicate: {frame_id}")
            source_hashes[image_hash] = frame_id
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                if image.format != "JPEG" or (width, height) != (int(item["width"]), int(item["height"])):
                    raise SystemExit(f"Image format/dimensions mismatch: {frame_id}")

            source_record = canonical.get(frame_id)
            if source_record is None or source_record["source_sha256"] != image_hash:
                raise SystemExit(f"Canonical source ID/hash mismatch: {frame_id}")
            if item["canonical_source_sha256_match"].lower() != "true":
                raise SystemExit(f"Crosswalk did not verify canonical source: {frame_id}")
            frames_csv_rel = item["frames_csv_path"]
            if frames_csv_rel not in frames_csv_cache:
                csv_path = safe_source(root, frames_csv_rel)
                csv_rows = read_csv(csv_path)
                indexed = {str(int(r["frame_index"])): r for r in csv_rows}
                if len(indexed) != len(csv_rows):
                    raise SystemExit(f"Duplicate frame index in {frames_csv_rel}")
                frames_csv_cache[frames_csv_rel] = (sha256(csv_path), indexed)
            csv_hash, indexed = frames_csv_cache[frames_csv_rel]
            recorded = indexed.get(str(int(item["frame_index"])))
            if csv_hash != item["frames_csv_sha256"] or not recorded:
                raise SystemExit(f"Session frames.csv hash/ID mismatch: {frame_id}")
            if (recorded["filename"], recorded["timestamp_ns"], recorded["width"], recorded["height"]) != (
                image_path.name, item["timestamp_ns"], str(width), str(height)
            ):
                raise SystemExit(f"Session frames.csv relationship mismatch: {frame_id}")
            if source_record["annotation_status"] != item["canonical_annotation_status"]:
                raise SystemExit(f"Annotation status drift: {frame_id}")
            if source_record["training_approved"].lower() != "false" or item["training_approved"].lower() != "false":
                raise SystemExit(f"Unexpected training approval: {frame_id}")

            annotation = item["canonical_annotation_status"]
            full_status_counts[annotation] += 1
            remote = remote_by_id.get(frame_id)
            duplicate_of = ""
            if remote:
                if remote.get("source_sha256") != image_hash:
                    raise SystemExit(f"Remote pilot ID has different source bytes: {frame_id}")
                duplicate_of = "guzzbr/draw-real-seg-pilot-v1"
                external_duplicates[duplicate_of] += 1

            pseudo_hash = item["canonical_pseudo_mask_sha256"]
            pseudo_method = pseudo_version = ""
            pseudo_status = "missing"
            pixel_counts: Counter[int] = Counter()
            pseudo_path = item["canonical_pseudo_mask_path"]
            if pseudo_path:
                if annotation not in {"pseudo_supervised", "pseudo_all_unknown"}:
                    raise SystemExit(f"Mask exists for an unsupported annotation status: {frame_id}")
                mask_path = safe_source(root, pseudo_path)
                if not mask_path.is_file():
                    raise SystemExit(f"Missing pseudo mask file: {frame_id}")
                if sha256(mask_path) != pseudo_hash:
                    raise SystemExit(f"Pseudo-mask hash mismatch: {frame_id}")
                with Image.open(mask_path) as mask:
                    mask.load()
                    if mask.mode != "L" or mask.size != (width, height):
                        invalid_masks += 1
                        pseudo_status = "invalid_dimensions_or_channels"
                    else:
                        pixel_counts.update(mask.getdata())
                        if not set(pixel_counts).issubset(ALLOWED_MASK_VALUES):
                            invalid_masks += 1
                            pseudo_status = "invalid_pixel_values"
                        elif annotation == "pseudo_all_unknown" and set(pixel_counts) != {255}:
                            invalid_masks += 1
                            pseudo_status = "invalid_all_unknown_mask"
                        else:
                            pseudo_status = "valid_exploratory_pseudo"
                if not remote or not remote.get("mask_sha256") or remote["mask_sha256"] != pseudo_hash:
                    raise SystemExit(f"Remote mask SHA-256 missing/mismatch: {frame_id}")
                duplicate_masks_verified += 1
                remote_proposal_path = remote.get("mask_proposal_path", "")
                if not remote_proposal_path:
                    raise SystemExit(f"Remote pseudo-mask method path missing: {frame_id}")
                remote_proposal_file = safe_source(root, remote_proposal_path)
                if not remote_proposal_file.is_file():
                    raise SystemExit(f"Pseudo-mask proposal source missing: {frame_id}")
                pseudo_version = Path(remote_proposal_path).parts[-2]
                if pseudo_version != "reviewed_v3":
                    raise SystemExit(f"Unexpected pseudo-mask version for {frame_id}: {pseudo_version}")
                pseudo_method = "mask proposal reviewed_v3 + thumbnail-level visual triage; fine pixels unverified"
                pseudo_methods[f"{pseudo_method} / {pseudo_version}"] += 1
                mask_archive.write(mask_path, f"{frame_id}.png")
            else:
                missing_masks += 1

            neighbors = neighbor_map[frame_id]
            derived_paths = sorted(derived_by_id.get(frame_id, []))
            derived_count = len(derived_paths)
            context_status = item["capture_context_status"] or "not_documented"
            queue_status = item["proposal_queue_status"] or "not_recorded"
            proposal_snapshot = item["proposal_inference_status_snapshot"] or "not_recorded"
            scene_status = "provided" if item["primary_subject"] or item["primary_action"] else "not_provided"
            include = session in EXPECTED_SESSIONS

            if include:
                image_archive_path = f"{session}/{image_path.name}"
                archive.write(image_path, image_archive_path)
                included_status_counts[annotation] += 1
                session_counts[session] += 1
                frames.append({
                    "frame_id": frame_id,
                    "image_path_in_archive": image_archive_path,
                    "session_id": session,
                    "frame_index": str(int(item["frame_index"])),
                    "timestamp_ns": item["timestamp_ns"],
                    "take_id": source_record.get("take_id", "") or "not_recorded",
                    "take_uid": source_record.get("take_uid", "") or "not_recorded",
                    "width": str(width), "height": str(height), "source_sha256": image_hash,
                    "source_manifest_path": "captures/manifests/source_frames_20261031_after_user_review.csv",
                    "annotation_status": annotation,
                    "annotation_authority": "none; no pixel mask" if annotation == "unlabeled" else "heuristic pseudo; review only",
                    "mask_status": "missing" if not pseudo_path else pseudo_status,
                    "pseudo_mask_path_in_archive": f"{frame_id}.png" if pseudo_path else "",
                    "pseudo_mask_sha256": pseudo_hash,
                    "pseudo_mask_method": pseudo_method if pseudo_path else "not_applicable; no mask",
                    "pseudo_mask_version": pseudo_version if pseudo_path else "not_applicable",
                    "pseudo_mask_pixel_counts": json.dumps({str(k): pixel_counts[k] for k in sorted(pixel_counts)}, separators=(",", ":")),
                    "invalid_mask": "false",
                    "training_approved": "false", "review_split": "review_only",
                    "source_split_role": item["split_role"] or "unassigned",
                    "session_used_for_prior_training": str(session in TRAINING_USED_SESSIONS).lower(),
                    "external_exact_duplicate_of": duplicate_of,
                    "previous_frame_id": neighbors[0], "previous_frame_delta_ns": neighbors[1],
                    "next_frame_id": neighbors[2], "next_frame_delta_ns": neighbors[3],
                    "derived_copy_count": str(derived_count),
                    "derived_copy_paths": json.dumps(derived_paths, separators=(",", ":")),
                    "proposal_queue_status": queue_status,
                    "proposal_status_snapshot": proposal_snapshot,
                    "proposal_status_record": "recorded" if proposal_snapshot != "not_recorded" else "not_recorded",
                    "proposal_mask_included": "false",
                    "scene_subject": item["primary_subject"], "scene_action": item["primary_action"],
                    "scene_tag_status": scene_status,
                    "scene_label_note": "scene tags are separate from pixel masks",
                    "capture_context_status": context_status,
                    "source_license": "other; user-confirmed owner content; private use only",
                    "privacy_scope": "private owner capture; private Kaggle dataset only",
                    "privacy_authorization_status": USER_PRIVATE_AUTHORIZATION,
                })
            else:
                reason = "source outside the user-owned capture pool; excluded from release"
                exclusion_reasons[reason] += 1
                excluded_status_counts[annotation] += 1
                excluded_sessions[session] += 1
                excluded.append({
                    "frame_id": frame_id, "session_id": session,
                    "frame_index": str(int(item["frame_index"])), "source_sha256": image_hash,
                    "annotation_status": annotation, "exclusion_reason": reason,
                    "capture_context_status": context_status,
                    "external_exact_duplicate_of": duplicate_of,
                    "pseudo_mask_sha256": pseudo_hash,
                    "pseudo_mask_method": pseudo_method or "not_applicable",
                    "pseudo_mask_version": pseudo_version or "not_applicable",
                    "pseudo_mask_status": pseudo_status,
                    "derived_copy_count": str(derived_count),
                    "derived_copy_paths": json.dumps(derived_paths, separators=(",", ":")),
                    "previous_frame_id": neighbors[0], "previous_frame_delta_ns": neighbors[1],
                    "next_frame_id": neighbors[2], "next_frame_delta_ns": neighbors[3],
                    "proposal_queue_status": queue_status,
                    "proposal_status_snapshot": proposal_snapshot,
                    "proposal_status_record": "recorded" if proposal_snapshot != "not_recorded" else "not_recorded",
                    "proposal_mask_included": "false",
                    "scene_subject": item["primary_subject"], "scene_action": item["primary_action"],
                    "scene_tag_status": scene_status,
                    "scene_label_note": "not_provided" if scene_status == "not_provided" else "scene tags are separate from pixel masks",
                })

    if dict(full_status_counts) != {"pseudo_supervised": 286, "unlabeled": 4789, "pseudo_all_unknown": 11}:
        raise SystemExit(f"Full inventory annotation counts differ: {dict(full_status_counts)}")
    if dict(full_session_counts) != EXPECTED_SESSIONS:
        raise SystemExit(f"Full inventory session counts differ: {dict(full_session_counts)}")
    if dict(included_status_counts) != dict(full_status_counts):
        raise SystemExit(f"Release annotation counts differ: {dict(included_status_counts)}")
    if len(frames) + len(excluded) != 5086 or dict(external_duplicates) != {"guzzbr/draw-real-seg-pilot-v1": 297}:
        raise SystemExit("Frame/external-overlap reconciliation failed")
    if duplicate_masks_verified != 297 or invalid_masks != 0:
        raise SystemExit(f"Pseudo-mask validation counts differ: verified={duplicate_masks_verified}, invalid={invalid_masks}")
    if dict(session_counts) != EXPECTED_SESSIONS:
        raise SystemExit(f"Included session counts differ: {dict(session_counts)}")
    with zipfile.ZipFile(images_zip) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != 5086:
            raise SystemExit("Image archive integrity/count mismatch")
    with zipfile.ZipFile(masks_zip) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != 297:
            raise SystemExit("Pseudo-mask archive integrity/count mismatch")

    write_rows(out / "frames.csv", FRAME_FIELDS, frames)
    write_rows(out / "privacy_exclusions.csv", EXCLUDED_FIELDS, excluded)

    # Session-neighbor relations over all canonical sessions; no images are copied for these.
    times_by_session: dict[str, list[int]] = defaultdict(list)
    count_by_session: Counter[str] = Counter()
    for record in canonical_rows:
        count_by_session[record["session_id"]] += 1
        if record.get("timestamp_ns"):
            times_by_session[record["session_id"]].append(int(record["timestamp_ns"]))
    session_intervals = {
        sid: (min(values), max(values)) for sid, values in times_by_session.items() if values
    }
    ordered_sessions = sorted(session_intervals, key=lambda sid: session_intervals[sid][0])
    session_lineage = {}
    for sid in sorted(EXPECTED_SESSIONS):
        idx = ordered_sessions.index(sid)
        session_lineage[sid] = {
            "frame_count": count_by_session[sid],
            "start_timestamp_ns": session_intervals[sid][0],
            "end_timestamp_ns": session_intervals[sid][1],
            "previous_source_session": ordered_sessions[idx - 1] if idx else None,
            "next_source_session": ordered_sessions[idx + 1] if idx + 1 < len(ordered_sessions) else None,
            "used_in_prior_training": sid in TRAINING_USED_SESSIONS,
            "included_in_release_candidate": sid in EXPECTED_SESSIONS,
        }
    external_neighbor_ids = {
        neighbor
        for entry in session_lineage.values()
        for neighbor in (entry["previous_source_session"], entry["next_source_session"])
        if neighbor and neighbor not in EXPECTED_SESSIONS
    }
    external_neighbor_details = {
        sid: {
            "frame_count": count_by_session[sid],
            "start_timestamp_ns": session_intervals[sid][0],
            "end_timestamp_ns": session_intervals[sid][1],
        }
        for sid in sorted(external_neighbor_ids)
    }
    selected_ids = {r["source_frame_id"] for r in crosswalk}
    derived_record_count = sum(len(paths) for paths in derived_by_id.values())
    selected_derived_ids = selected_ids.intersection(derived_by_id)

    metadata = {
        "title": "Draw Real Capture Review Pool v1",
        "id": "guzzbr/draw-real-capture-pool-v1",
        "isPrivate": True,
        "licenses": [{"name": "other"}],
        "description": (
            "PRIVATE OWNER CAPTURE DATA. Review-only pool from 5,086 original frames in 8 owner sessions. "
            "Pseudo-masks are exploratory; no frame is approved ground truth or holdout."
        ),
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "dataset_kind": "private_real_capture_review_pool_candidate",
        "release_id": "20261031-v4-preflight",
        "kaggle_slug": "guzzbr/draw-real-capture-pool-v1",
        "isPrivate": True,
        "technical_preflight_passed": True,
        "independent_review_status": "pending",
        "eligible_for_upload": False,
        "privacy_review_basis": {
            "source": "user-owned original capture directory with source manifests and per-session frames.csv hashes",
            "private_use_authorization": USER_PRIVATE_AUTHORIZATION,
            "included_sessions_context": "user directly confirmed all included person/content is theirs and authorized private upload",
            "external_sources": "none included; third-party document corpora and uncertain external files are excluded",
        },
        "source_manifest": "captures/manifests/source_frames_20261031_after_user_review.csv",
        "source_manifest_sha256": sha256(args.canonical_manifest),
        "crosswalk": "captures/triage/local_training_capture_categorization_20261031_v1/capture_frame_crosswalk.csv",
        "crosswalk_sha256": sha256(args.crosswalk),
        "derived_copies_manifest": "captures/manifests/derived_source_copies_20261031.csv",
        "derived_copies_sha256": sha256(args.derived_copies),
        "existing_pilot_samples_sha256": sha256(args.reference_samples),
        "source_inventory": {
            "frame_count": len(crosswalk), "session_count": len(EXPECTED_SESSIONS),
            "session_counts": dict(sorted(full_session_counts.items())), "annotation_status_counts": dict(sorted(full_status_counts.items())),
            "unique_frame_ids": len(ids), "unique_source_sha256": len(source_hashes),
            "internal_exact_duplicate_source_sha256": 0,
        },
        "release_candidate": {
            "frame_count": len(frames), "session_counts": dict(sorted(session_counts.items())),
            "annotation_status_counts": dict(sorted(included_status_counts.items())),
            "all_review_split": "review_only", "training_approved_count": 0,
            "human_ground_truth_count": 0, "missing_mask_count": missing_masks,
            "invalid_mask_count": invalid_masks, "pseudo_masks_included": 297,
            "proposal_masks_included": 0, "scene_tag_status": "not_provided status recorded per frame",
        },
        "privacy_exclusions": {
            "frame_count": len(excluded), "session_counts": dict(sorted(excluded_sessions.items())),
            "annotation_status_counts": dict(sorted(excluded_status_counts.items())),
            "reason_counts": dict(sorted(exclusion_reasons.items())),
            "note": "No user-owned source frames excluded; no third-party data files are included.",
        },
        "pseudo_masks": {
            "included": 297,
            "pseudo_supervised": 286, "pseudo_all_unknown": 11, "invalid_mask_count": invalid_masks,
            "missing_mask_count_in_full_inventory": missing_masks,
            "verified_remote_mask_hash_count": duplicate_masks_verified,
            "method_version_counts": dict(sorted(pseudo_methods.items())),
            "class_legend": MASK_CLASSES,
        },
        "known_exact_overlap_with_existing_dataset": {
            "slug": "guzzbr/draw-real-seg-pilot-v1", "frame_count": external_duplicates["guzzbr/draw-real-seg-pilot-v1"],
            "pseudo_mask_hashes_verified": duplicate_masks_verified,
            "prior_training_session_included_for_review_only": True,
        },
        "lineage": {
            "adjacent_frames": "previous/next frame IDs and timestamp deltas are recorded per source frame",
            "derived_annotation_copies": "paths/counts are recorded per source frame; only original images are packaged",
            "derived_copy_inventory": {
                "manifest_record_count": derived_record_count,
                "selected_frames_with_derived_copies": len(selected_derived_ids),
                "derived_records_outside_this_5086_frame_pool": derived_record_count - sum(len(derived_by_id[sid]) for sid in selected_derived_ids),
            },
            "session_neighbors_by_capture_time": session_lineage,
            "external_neighbor_session_details": external_neighbor_details,
            "prior_training_session": sorted(TRAINING_USED_SESSIONS),
            "holdout": "none; all sessions are review_only, including the session previously used for pseudo-label training",
        },
        "mask_contract": {"classes": MASK_CLASSES, "allowed_values": [0, 1, 2, 3, 255], "ignore_index": 255},
        "distinctions": {
            "missing_mask_count": missing_masks,
            "invalid_mask_count": invalid_masks,
            "note": "Missing masks are intentionally unlabeled; they are not empty masks and are not invalid masks. This count describes the full source inventory.",
        },
        "archive_sha256": {"images.zip": sha256(images_zip), "pseudo_masks.zip": sha256(masks_zip), "frames.csv": sha256(out / "frames.csv"),
                           "privacy_exclusions.csv": sha256(out / "privacy_exclusions.csv")},
        "archive_bytes": {"images.zip": images_zip.stat().st_size, "pseudo_masks.zip": masks_zip.stat().st_size,
                          "frames.csv": (out / "frames.csv").stat().st_size,
                          "privacy_exclusions.csv": (out / "privacy_exclusions.csv").stat().st_size},
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Draw Real Capture Review Pool v1\n\n"
        "**PRIVATE USER-OWNED CAPTURES. Review-only; pseudo-masks are exploratory, not ground truth.**\n\n"
        "This bundle contains 5,086 original JPEGs from 8 user-owned capture sessions and 297 separate "
        "pseudo-masks. All frames are `review_only`; none is `training_approved`, no human ground truth exists, "
        "and there is no train/validation/test/holdout split.\n\n"
        "The user explicitly confirmed the depicted person and capture content are theirs and authorized private "
        "Kaggle upload. No third-party corpus or external source file is included. Kaggle privacy is `private`.\n\n"
        "## Mask contract and status\n\n"
        "Mask classes are: 0 background; 1 visible paper (including "
        "pre-existing print/text); 2 visible hand; 3 user-added visible ink; 255 uncertain/ignore (never background).\n\n"
        "Missing masks are intentionally unlabeled; they are not empty masks. Invalid masks are tracked separately. "
        "Scene tags and model proposal statuses are explicit metadata and are not pixel labels.\n\n"
        "The 297 exact image/mask overlaps with `guzzbr/draw-real-seg-pilot-v1` are cross-referenced per frame "
        "with verified source and mask hashes. Their method/version is `reviewed_v3` plus thumbnail-level visual "
        "triage; fine pixels remain unverified. The overlap is retained for a complete review inventory and is not "
        "a holdout. The session previously used for pseudo-label training remains marked in `frames.csv`.\n\n"
        "Neighbor frame IDs and timestamp deltas, nearest source-session relationships, and derived-copy references "
        "are recorded for lineage review. The 90 derived-copy manifest rows do not refer to these 5,086 frame IDs.\n\n"
        "No temporal pairs, model proposal masks, third-party document corpora, or training kernels are included.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(out), "manifest": manifest,
        "candidate_frames": len(frames), "excluded_frames": len(excluded),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
