import hashlib
import json

import pytest

from tools.runner import improvement_loop as loop


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        path.write_text(data, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(tmp_path, *, semantic_status="agent consensus; not human ground truth"):
    samples = tmp_path / "source_samples.csv"
    review = tmp_path / "review.csv"
    freeze = tmp_path / "freeze.json"
    source_train = tmp_path / "source_train.jpg"
    source_holdout = tmp_path / "source_holdout.jpg"
    train_image = tmp_path / "train/images/train_frame.jpg"
    train_mask = tmp_path / "train/masks/train_mask.png"
    holdout_image = tmp_path / "holdout/images/holdout_frame.jpg"
    holdout_mask = tmp_path / "holdout/masks/holdout_mask.png"
    source_train_sha = _write(source_train, "source train image")
    source_holdout_sha = _write(source_holdout, "source holdout image")
    train_image_sha = _write(train_image, "source train image")
    train_mask_sha = _write(train_mask, "packaged train mask")
    holdout_image_sha = _write(holdout_image, "source holdout image")
    holdout_mask_sha = _write(holdout_mask, "packaged holdout mask")
    sample_rows = [
        ["train_session_a", "1", "train", "source_train.jpg", source_train_sha,
         "train/images/train_frame.jpg", train_image_sha,
         "train/masks/train_mask.png", train_mask_sha],
        ["new_holdout_session", "2", "holdout", "source_holdout.jpg", source_holdout_sha,
         "holdout/images/holdout_frame.jpg", holdout_image_sha,
         "holdout/masks/holdout_mask.png", holdout_mask_sha],
    ]
    with (tmp_path / "dataset_manifest_samples.csv").open("w", newline="") as stream:
        import csv
        writer = csv.writer(stream)
        writer.writerow(["session_id", "frame_index", "split", "source_path", "source_sha256",
                         "image_path", "image_sha256", "mask_path", "mask_sha256"])
        writer.writerows(sample_rows)
    sample_sha = _write(samples, "frame_id,source_sha256,mask_sha256\n")
    review_sha = _write(review, "frame_id,reviewer,outcome\n")
    _write(freeze, {
        "schema_version": 2,
        "train_session_ids": ["train_session_a"],
        "previously_trained_session_ids": [],
        "holdout_session_id": "new_holdout_session",
        "train_source_ids": [
            {"session_id": "train_session_a", "frame_index": 1},
            {"session_id": "train_session_a", "frame_index": 3},
        ],
        "holdout_source_ids": [{"session_id": "new_holdout_session", "frame_index": 2}],
    })
    freeze_sha = hashlib.sha256(freeze.read_bytes()).hexdigest()
    manifest_sha = _write(tmp_path / "dataset_manifest.json", {
        "schema_version": 1,
        "dataset_kind": "expert_consensus_reviewed_session_split",
        "semantic_status": semantic_status,
        "source_samples_csv": "source_samples.csv",
        "source_samples_sha256": sample_sha,
        "review_csv": "review.csv",
        "review_csv_sha256": review_sha,
        "split_freeze_manifest": "freeze.json",
        "split_freeze_sha256": freeze_sha,
        "train": {
            "session_ids": ["train_session_a"], "frames": 1,
            "class_pixel_support": {"3": 250},
        },
        "holdout": {
            "session_id": "new_holdout_session", "frames": 1,
            "class_pixel_support": {"3": 100},
            "scope": "single-session holdout",
        },
        "excluded_unreviewed_or_rejected": {"train": 1, "holdout": 0},
        "ignore_index": 255,
        "unknown_pixels_are_preserved": True,
    })
    # The packaged samples CSV is distinct from the source samples CSV named
    # by the builder manifest and is the exact set the trainer will consume.
    (tmp_path / "dataset_manifest.json").parent.joinpath("samples.csv").write_bytes(
        (tmp_path / "dataset_manifest_samples.csv").read_bytes())
    return manifest_sha


def test_init_creates_exact_cycle_count_without_external_actions(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    workdir = tmp_path / "loop"
    state = loop.initialize(workdir, 5)
    assert len(state["cycles"]) == 5
    assert state["kaggle_started"] is False
    assert state["training_started"] is False
    assert state["promotion_allowed"] is False


def test_freeze_marks_agent_consensus_exploratory(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    manifest = _dataset(tmp_path)
    workdir = tmp_path / "loop"
    loop.initialize(workdir, 1)

    snapshot = loop.freeze_snapshot(workdir, 1, tmp_path / "dataset_manifest.json")

    assert snapshot["state"] == "snapshot_frozen_exploratory"
    assert snapshot["dataset"]["quality_claim_eligible"] is False
    saved = json.loads((workdir / "loop_state.json").read_text())
    assert saved["completed_snapshots"] == 1
    assert saved["kaggle_started"] is False


def test_manifest_booleans_cannot_open_quality_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    _dataset(tmp_path, semantic_status="human reviewed")
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["human_adjudication_status"] = "approved"
    manifest["holdout_independent"] = True
    manifest_path.write_text(json.dumps(manifest))
    workdir = tmp_path / "loop"
    loop.initialize(workdir, 1)

    snapshot = loop.freeze_snapshot(workdir, 1, manifest_path)

    assert snapshot["state"] == "snapshot_frozen_exploratory"
    assert snapshot["dataset"]["quality_claim_eligible"] is False


def test_prior_training_sessions_can_remain_in_incremental_train(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    _dataset(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["previously_trained_session_ids"] = ["train_session_a"]
    freeze_sha = _write(freeze_path, freeze)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split_freeze_sha256"] = freeze_sha
    _write(manifest_path, manifest)
    workdir = tmp_path / "loop"
    loop.initialize(workdir, 1)

    snapshot = loop.freeze_snapshot(workdir, 1, manifest_path)

    assert snapshot["dataset"]["training_session_ids"] == ["train_session_a"]


def test_freeze_rejects_train_holdout_session_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    _dataset(tmp_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["holdout"]["session_id"] = "train_session_a"
    manifest_path.write_text(json.dumps(manifest))
    workdir = tmp_path / "loop"
    loop.initialize(workdir, 1)

    with pytest.raises(loop.LoopError, match="overlap"):
        loop.freeze_snapshot(workdir, 1, manifest_path)


def test_freeze_rejects_manifest_split_that_disagrees_with_freeze_file(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    _dataset(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["train_session_ids"] = ["some_other_session"]
    freeze_sha = _write(freeze_path, freeze)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split_freeze_sha256"] = freeze_sha
    _write(manifest_path, manifest)
    workdir = tmp_path / "loop"
    loop.initialize(workdir, 1)

    with pytest.raises(loop.LoopError, match="frozen train sessions differ"):
        loop.freeze_snapshot(workdir, 1, manifest_path)


def test_freeze_requires_distinct_dataset_for_each_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    _dataset(tmp_path)
    workdir = tmp_path / "loop"
    loop.initialize(workdir, 2)
    manifest_path = tmp_path / "dataset_manifest.json"
    loop.freeze_snapshot(workdir, 1, manifest_path)

    with pytest.raises(loop.LoopError, match="already used"):
        loop.freeze_snapshot(workdir, 2, manifest_path)


def test_freeze_detects_source_manifest_hash_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    _dataset(tmp_path)
    (tmp_path / "review.csv").write_text("tampered\n")
    workdir = tmp_path / "loop"
    loop.initialize(workdir, 1)

    with pytest.raises(loop.LoopError, match="hash mismatch"):
        loop.freeze_snapshot(workdir, 1, tmp_path / "dataset_manifest.json")


def test_status_detects_changed_mask_after_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    _dataset(tmp_path)
    workdir = tmp_path / "loop"
    loop.initialize(workdir, 1)
    loop.freeze_snapshot(workdir, 1, tmp_path / "dataset_manifest.json")
    (tmp_path / "train/masks/train_mask.png").write_text("changed")

    with pytest.raises(loop.LoopError, match="mask hash mismatch"):
        loop.status(workdir)


def test_status_detects_changed_dataset_metadata_after_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    _dataset(tmp_path)
    workdir = tmp_path / "loop"
    manifest_path = tmp_path / "dataset_manifest.json"
    loop.initialize(workdir, 1)
    loop.freeze_snapshot(workdir, 1, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["semantic_status"] = "human adjudicated"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(loop.LoopError, match="manifest changed after freeze"):
        loop.status(workdir)


def test_freeze_rejects_overwrite_and_out_of_order(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    _dataset(tmp_path)
    workdir = tmp_path / "loop"
    loop.initialize(workdir, 2)
    manifest_path = tmp_path / "dataset_manifest.json"

    with pytest.raises(loop.LoopError, match="freeze cycles in order"):
        loop.freeze_snapshot(workdir, 2, manifest_path)

    loop.freeze_snapshot(workdir, 1, manifest_path)
    with pytest.raises(loop.LoopError, match="already"):
        loop.freeze_snapshot(workdir, 1, manifest_path)


def test_init_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    workdir = tmp_path / "loop"
    loop.initialize(workdir, 1)
    with pytest.raises(loop.LoopError, match="refusing to overwrite"):
        loop.initialize(workdir, 1)
