"""State and freeze snapshots for an N-cycle ink-model improvement loop.

This first implementation deliberately stops before training or Kaggle.
It turns a cycle into a unique, resumable, provenance-checked dataset
snapshot. Candidate training and promotion are separate gates and are not
performed by this module.

Examples (from the repository root)::

    python3 -m tools.runner.improvement_loop init \
        --workdir runs/ink-loop/2026-11-01 --cycles 5
    python3 -m tools.runner.improvement_loop freeze \
        --workdir runs/ink-loop/2026-11-01 --cycle 1 \
        --dataset-manifest captures/triage/training_ready/run/dataset_manifest.json
    python3 -m tools.runner.improvement_loop status \
        --workdir runs/ink-loop/2026-11-01

An exploratory or agent-consensus manifest can be frozen for bookkeeping,
but is marked exploratory and can never satisfy a real-quality promotion gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MAX_CYCLES = 100
ROOT = Path(__file__).resolve().parents[2]


class LoopError(ValueError):
    """Input is invalid or a workflow gate is not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoopError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise LoopError(f"expected a JSON object: {path}")
    return data


def _session_ids(section: dict[str, Any], field: str) -> set[str]:
    if field == "train":
        value = section.get("session_ids")
        if not isinstance(value, list) or not value:
            raise LoopError("dataset train.session_ids must be a non-empty list")
        return {str(item) for item in value if str(item)}
    value = section.get("session_id")
    if not value:
        raise LoopError("dataset holdout.session_id is required")
    return {str(value)}


def _verify_repo_artifact(relative_path: str, expected_sha: str, label: str) -> Path:
    if not relative_path or not expected_sha:
        raise LoopError(f"dataset manifest is missing {label} path/hash")
    artifact = (ROOT / relative_path).resolve(strict=True)
    if not artifact.is_relative_to(ROOT):
        raise LoopError(f"{label} escapes the repository")
    if not artifact.is_file():
        raise LoopError(f"{label} is not a file: {artifact}")
    actual_sha = sha256_file(artifact)
    if actual_sha != expected_sha:
        raise LoopError(f"{label} hash mismatch: expected {expected_sha}, got {actual_sha}")
    return artifact


def _verify_dataset_file(base: Path, relative_path: str, expected_sha: str, label: str) -> str:
    if not relative_path or not expected_sha:
        raise LoopError(f"dataset sample is missing {label} path/hash")
    candidate = (base / relative_path).resolve(strict=True)
    if not candidate.is_relative_to(base):
        raise LoopError(f"{label} escapes its dataset/source root")
    if not candidate.is_file():
        raise LoopError(f"{label} is not a file: {candidate}")
    actual_sha = sha256_file(candidate)
    if actual_sha != expected_sha:
        raise LoopError(f"{label} hash mismatch: expected {expected_sha}, got {actual_sha}")
    return actual_sha


def inspect_dataset_manifest(path: Path) -> dict[str, Any]:
    """Validate the current expert-reviewed dataset manifest shape.

    Agent consensus is retained as an exploratory training source, but the
    resulting snapshot is explicitly ineligible for a real-quality claim.
    """
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise LoopError(f"dataset manifest must stay inside {ROOT}")
    raw_manifest = path.read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as exc:
        raise LoopError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise LoopError(f"expected a JSON object: {path}")
    if manifest.get("schema_version") != 1:
        raise LoopError("dataset manifest schema_version must be 1")
    train = manifest.get("train")
    holdout = manifest.get("holdout")
    if not isinstance(train, dict) or not isinstance(holdout, dict):
        raise LoopError("dataset manifest must contain train and holdout objects")
    train_sessions = _session_ids(train, "train")
    holdout_sessions = _session_ids(holdout, "holdout")
    if train_sessions & holdout_sessions:
        raise LoopError("train and holdout sessions overlap")
    if int(train.get("frames", 0)) < 1 or int(holdout.get("frames", 0)) < 1:
        raise LoopError("train and holdout must each contain at least one reviewed frame")

    train_support = train.get("class_pixel_support", {})
    holdout_support = holdout.get("class_pixel_support", {})
    if int(train_support.get("3", 0)) <= 0:
        raise LoopError("train split has no supervised ink pixels (class 3)")
    if int(holdout_support.get("3", 0)) <= 0:
        raise LoopError("holdout has no reviewed ink pixels (class 3)")
    if manifest.get("ignore_index") != 255 or not manifest.get("unknown_pixels_are_preserved"):
        raise LoopError("manifest must preserve uncertain pixels as ignore_index 255")

    sample_path = _verify_repo_artifact(
        str(manifest.get("source_samples_csv", "")),
        str(manifest.get("source_samples_sha256", "")), "source samples CSV")
    review_path = _verify_repo_artifact(
        str(manifest.get("review_csv", "")),
        str(manifest.get("review_csv_sha256", "")), "review CSV")
    freeze_path = _verify_repo_artifact(
        str(manifest.get("split_freeze_manifest", "")),
        str(manifest.get("split_freeze_sha256", "")), "split freeze manifest")
    freeze = _load_json(freeze_path)
    if freeze.get("holdout_session_id") not in holdout_sessions:
        raise LoopError("frozen split does not identify the manifest holdout session")
    if set(freeze.get("train_session_ids", [])) != train_sessions:
        raise LoopError("frozen train sessions differ from the dataset manifest")
    if holdout_sessions.intersection(set(freeze.get("previously_trained_session_ids", []))):
        raise LoopError("holdout session was used for training previously")

    packaged_samples = path.parent / "samples.csv"
    if not packaged_samples.is_file():
        raise LoopError(f"packaged samples.csv missing beside dataset manifest: {packaged_samples}")
    rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "holdout": []}
    sample_provenance: list[dict[str, str]] = []
    source_ids: dict[str, set[tuple[str, str]]] = {"train": set(), "holdout": set()}
    with packaged_samples.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split = row.get("split", "")
            if split not in rows_by_split:
                raise LoopError(f"unexpected split in packaged samples.csv: {split!r}")
            session_id = row.get("session_id", "")
            expected_sessions = train_sessions if split == "train" else holdout_sessions
            if session_id not in expected_sessions:
                raise LoopError(f"sample {session_id}/{row.get('frame_index')} has wrong split")
            source_id = (session_id, row.get("frame_index", ""))
            if not source_id[0] or not source_id[1] or source_id in source_ids[split]:
                raise LoopError(f"missing or duplicate sample ID in packaged samples: {source_id}")
            source_ids[split].add(source_id)
            source_hash = _verify_dataset_file(
                ROOT, row.get("source_path", ""), row.get("source_sha256", ""), "source image")
            image_hash = _verify_dataset_file(
                path.parent, row.get("image_path", ""), row.get("image_sha256", ""), "packaged image")
            if image_hash != source_hash:
                raise LoopError(f"packaged image differs from source for {source_id}")
            mask_hash = _verify_dataset_file(
                path.parent, row.get("mask_path", ""), row.get("mask_sha256", ""), "mask")
            rows_by_split[split].append(row)
            sample_provenance.append({
                "session_id": session_id,
                "frame_index": source_id[1],
                "split": split,
                "source_sha256": source_hash,
                "mask_sha256": mask_hash,
            })
    if len(rows_by_split["train"]) != int(train["frames"]):
        raise LoopError("packaged train sample count differs from dataset manifest")
    if len(rows_by_split["holdout"]) != int(holdout["frames"]):
        raise LoopError("packaged holdout sample count differs from dataset manifest")
    frozen_train_ids = {
        (str(item["session_id"]), str(item["frame_index"]))
        for item in freeze.get("train_source_ids", [])
    }
    frozen_holdout_ids = {
        (str(item["session_id"]), str(item["frame_index"]))
        for item in freeze.get("holdout_source_ids", [])
    }
    if not source_ids["train"].issubset(frozen_train_ids) or not source_ids["holdout"].issubset(frozen_holdout_ids):
        raise LoopError("packaged sample IDs fall outside the frozen split manifest")
    excluded = manifest.get("excluded_unreviewed_or_rejected", {})
    expected_excluded_train = len(frozen_train_ids) - len(source_ids["train"])
    expected_excluded_holdout = len(frozen_holdout_ids) - len(source_ids["holdout"])
    if (int(excluded.get("train", -1)) != expected_excluded_train
            or int(excluded.get("holdout", -1)) != expected_excluded_holdout):
        raise LoopError("excluded sample counts differ from frozen split membership")

    semantic_status = str(manifest.get("semantic_status", "")).lower()
    human_reviewed = (
        "human" in semantic_status
        and "not human" not in semantic_status
        and manifest.get("human_adjudication_status") == "approved"
    )
    if "agent" in semantic_status or "consensus" in semantic_status:
        human_reviewed = False
    # This first version records evidence but has no independent evaluator
    # capable of validating adjudication and test isolation end to end.
    # Never let manifest-provided booleans open the quality/promotion gate.
    holdout_is_independent = False

    provenance = {
        "packaged_samples_sha256": sha256_file(packaged_samples),
        "source_samples_sha256": sha256_file(sample_path),
        "review_csv_sha256": sha256_file(review_path),
        "split_freeze_sha256": sha256_file(freeze_path),
        "training_session_ids": sorted(train_sessions),
        "holdout_session_ids": sorted(holdout_sessions),
        "sample_files": sample_provenance,
    }
    fingerprint = hashlib.sha256(json.dumps(
        provenance, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "data_fingerprint": fingerprint,
        "dataset_kind": manifest.get("dataset_kind", "unknown"),
        "semantic_status": manifest.get("semantic_status", "unspecified"),
        **provenance,
        "training_frames": int(train["frames"]),
        "holdout_frames": int(holdout["frames"]),
        "training_ink_pixels": int(train_support["3"]),
        "holdout_ink_pixels": int(holdout_support["3"]),
        "human_reviewed": human_reviewed,
        "independent_real_holdout": holdout_is_independent,
        "quality_claim_eligible": human_reviewed and holdout_is_independent,
    }


def initialize(workdir: Path, cycles: int) -> dict[str, Any]:
    if cycles < 1 or cycles > MAX_CYCLES:
        raise LoopError(f"cycles must be between 1 and {MAX_CYCLES}")
    workdir = workdir.resolve()
    if not workdir.is_relative_to(ROOT) or workdir == ROOT:
        raise LoopError(f"workdir must be a child directory of {ROOT}")
    state_path = workdir / "loop_state.json"
    if state_path.exists():
        raise LoopError(f"refusing to overwrite existing loop: {state_path}")
    state = {
        "schema_version": SCHEMA_VERSION,
        "target": "pixel-level-ink-segmentation",
        "requested_cycles": cycles,
        "completed_snapshots": 0,
        "training_started": False,
        "kaggle_started": False,
        "promotion_allowed": False,
        "cycles": [
            {"number": index, "state": "awaiting_reviewed_snapshot"}
            for index in range(1, cycles + 1)
        ],
    }
    _atomic_json(state_path, state)
    return state


def freeze_snapshot(workdir: Path, cycle_number: int, dataset_manifest: Path) -> dict[str, Any]:
    workdir = workdir.resolve(strict=True)
    if not workdir.is_relative_to(ROOT) or workdir == ROOT:
        raise LoopError(f"workdir must be a child directory of {ROOT}")
    state_path = workdir / "loop_state.json"
    state = _load_json(state_path)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise LoopError("unsupported loop state schema")
    cycles = state.get("cycles", [])
    if not 1 <= cycle_number <= len(cycles):
        raise LoopError(f"cycle must be between 1 and {len(cycles)}")

    entry = cycles[cycle_number - 1]
    if entry.get("state") != "awaiting_reviewed_snapshot":
        raise LoopError(f"cycle {cycle_number} is already {entry.get('state')}")
    if cycle_number > 1 and cycles[cycle_number - 2].get("state") not in {
        "snapshot_frozen_exploratory", "snapshot_frozen_quality_eligible"
    }:
        raise LoopError("freeze cycles in order; the previous cycle lacks a frozen snapshot")

    evidence = inspect_dataset_manifest(dataset_manifest)
    for prior in cycles[:cycle_number - 1]:
        prior_snapshot = prior.get("snapshot", {})
        if prior_snapshot.get("data_fingerprint") == evidence["data_fingerprint"]:
            raise LoopError("this dataset snapshot was already used by an earlier cycle")

    snapshot = {
        "schema_version": 1,
        "cycle": cycle_number,
        "target": state["target"],
        "dataset": evidence,
        "state": "snapshot_frozen_exploratory",
        "training_executed": False,
        "kaggle_executed": False,
        "promotion_allowed": False,
    }
    snapshot_path = workdir / f"cycle_{cycle_number:03d}_snapshot.json"
    if snapshot_path.exists():
        existing = _load_json(snapshot_path)
        if existing.get("dataset", {}).get("data_fingerprint") != evidence["data_fingerprint"]:
            raise LoopError(f"refusing to replace frozen snapshot: {snapshot_path}")
        snapshot = existing
    else:
        _atomic_json(snapshot_path, snapshot)

    entry.update({
        "state": snapshot["state"],
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "snapshot": {
            "dataset_manifest_sha256": evidence["sha256"],
            "data_fingerprint": evidence["data_fingerprint"],
        },
        "quality_claim_eligible": evidence["quality_claim_eligible"],
    })
    state["completed_snapshots"] = sum(
        item.get("state", "").startswith("snapshot_frozen") for item in cycles
    )
    state["next_action"] = (
        "review_and_freeze_next_distinct_dataset_snapshot"
        if state["completed_snapshots"] < state["requested_cycles"]
        else "training_and_quality_gates_not_yet_implemented"
    )
    _atomic_json(state_path, state)
    return snapshot


def status(workdir: Path) -> dict[str, Any]:
    workdir = workdir.resolve(strict=True)
    if not workdir.is_relative_to(ROOT) or workdir == ROOT:
        raise LoopError(f"workdir must be a child directory of {ROOT}")
    state = _load_json(workdir / "loop_state.json")
    for entry in state.get("cycles", []):
        snapshot_path = entry.get("snapshot_path")
        if not snapshot_path:
            continue
        snapshot_file = Path(snapshot_path).resolve(strict=True)
        if not snapshot_file.is_relative_to(workdir):
            raise LoopError("frozen snapshot path escapes the loop workdir")
        if sha256_file(snapshot_file) != entry.get("snapshot_sha256"):
            raise LoopError(f"frozen snapshot hash changed: {snapshot_file}")
        frozen = _load_json(snapshot_file)
        current = inspect_dataset_manifest(Path(frozen["dataset"]["path"]))
        if current["sha256"] != frozen["dataset"].get("sha256"):
            raise LoopError(f"dataset manifest changed after freeze: {frozen['dataset']['path']}")
        if current["data_fingerprint"] != frozen["dataset"].get("data_fingerprint"):
            raise LoopError(f"dataset provenance changed after freeze: {frozen['dataset']['path']}")
    return {
        "requested_cycles": state.get("requested_cycles"),
        "completed_snapshots": state.get("completed_snapshots", 0),
        "cycles": state.get("cycles", []),
        "training_started": state.get("training_started", False),
        "kaggle_started": state.get("kaggle_started", False),
        "promotion_allowed": state.get("promotion_allowed", False),
        "next_action": state.get("next_action", "freeze_first_dataset_snapshot"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create a resumable N-cycle run")
    init_parser.add_argument("--workdir", type=Path, required=True)
    init_parser.add_argument("--cycles", type=int, required=True)

    freeze_parser = subparsers.add_parser("freeze", help="freeze one reviewed dataset snapshot")
    freeze_parser.add_argument("--workdir", type=Path, required=True)
    freeze_parser.add_argument("--cycle", type=int, required=True)
    freeze_parser.add_argument("--dataset-manifest", type=Path, required=True)

    status_parser = subparsers.add_parser("status", help="show loop progress")
    status_parser.add_argument("--workdir", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = initialize(args.workdir, args.cycles)
        elif args.command == "freeze":
            result = freeze_snapshot(args.workdir, args.cycle, args.dataset_manifest)
        else:
            result = status(args.workdir)
    except (LoopError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
