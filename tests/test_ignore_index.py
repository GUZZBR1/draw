"""Ignore-index contract for sparse review masks (255 = unknown)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from train import (  # noqa: E402
    CLASSES,
    IGNORE_INDEX,
    combined_loss,
    dice_loss_numpy,
    ink_tolerance_metrics as trainer_ink_tolerance_metrics,
    load_mask,
    per_class_metrics,
    require_valid_evaluation,
    validate_mask,
)
from tools import seg_metrics  # noqa: E402


def test_mask_validation_accepts_only_classes_and_ignore_index():
    assert validate_mask(np.array([[0, 1, 2, 3, 255]], dtype=np.uint8))
    with pytest.raises(ValueError):
        validate_mask(np.array([[4]], dtype=np.uint8))
    with pytest.raises(ValueError):
        seg_metrics.evaluate(np.array([[255]]), np.array([[0]]))
    with pytest.raises(ValueError):
        seg_metrics.evaluate(np.array([[0]]), np.array([[4]]))
    with pytest.raises(ValueError):
        seg_metrics.evaluate(np.array([[0]]), np.array([[256]]))


def test_load_mask_preserves_unknown_pixels(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "sparse.png"
    assert cv2.imwrite(str(path), np.array([[0, 255]], dtype=np.uint8))
    assert np.array_equal(load_mask(path), np.array([[0, 255]], dtype=np.int64))


def test_unknown_pixels_do_not_change_numpy_trainer_metrics_or_dice():
    target = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.uint8)
    pred = np.array([[0, 1, 3, 3], [1, 1, 2, 0]], dtype=np.uint8)
    prob = np.full((4, *target.shape), 0.1, dtype=np.float32)
    for cls in range(4):
        prob[cls][pred == cls] = 0.7

    sparse_target = target.copy()
    sparse_pred = pred.copy()
    sparse_target[0, 1] = IGNORE_INDEX
    sparse_target[1, 3] = IGNORE_INDEX
    sparse_pred[0, 1] = 3  # arbitrary output at ignored locations
    sparse_pred[1, 3] = 2

    actual = per_class_metrics(sparse_pred, sparse_target)
    valid = sparse_target != IGNORE_INDEX
    expected = per_class_metrics(pred[valid], target[valid])
    for cls in CLASSES.values():
        assert actual[cls] == expected[cls]
    assert dice_loss_numpy(prob, sparse_target) == pytest.approx(
        dice_loss_numpy(prob[:, sparse_target != IGNORE_INDEX].reshape(4, 1, -1), target[sparse_target != IGNORE_INDEX].reshape(1, -1))
    )


def test_metrics_unknown_pixels_are_excluded_and_support_is_reported():
    target = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    pred = np.array([[0, 1, 0, 3]], dtype=np.uint8)
    target_sparse = target.copy()
    target_sparse[0, 2] = IGNORE_INDEX
    pred_sparse = pred.copy()
    pred_sparse[0, 2] = 2  # arbitrary prediction in unknown area

    got = seg_metrics.evaluate(pred_sparse, target_sparse)
    expected = seg_metrics.evaluate(pred[:, [0, 1, 3]], target[:, [0, 1, 3]])
    for key in ("iou", "dice", "precision", "recall", "mean_iou", "mean_iou_present", "mean_dice", "accuracy"):
        assert got[key] == expected[key]
    assert got["valid_px"] == 3
    assert got["ignored_px"] == 1
    assert got["class_support_px"]["hand"] == 0
    assert got["iou"]["hand"] is None


def test_all_unknown_has_no_fabricated_scores():
    gt = np.full((3, 4), IGNORE_INDEX, dtype=np.uint8)
    pred = np.zeros_like(gt)
    metrics = seg_metrics.evaluate(pred, gt)
    assert metrics["valid_px"] == 0
    assert metrics["ignored_px"] == 12
    assert metrics["mean_iou"] is None
    assert metrics["mean_dice"] is None
    assert metrics["accuracy"] is None
    assert all(metrics[key][cls] is None for key in ("iou", "dice", "precision", "recall") for cls in seg_metrics.CLASSES)
    unknown_ink_prediction = np.full((3, 4), 3, dtype=np.uint8)
    unknown_tol = seg_metrics.ink_tolerance_metrics(unknown_ink_prediction, gt)
    assert unknown_tol["precision"] is None and unknown_tol["recall"] is None and unknown_tol["f1"] is None
    train_metrics = per_class_metrics(pred, gt)
    assert all(train_metrics[name][metric] is None for name in CLASSES.values() for metric in ("iou", "dice", "precision", "recall"))
    assert dice_loss_numpy(np.full((4, 3, 4), 0.25, dtype=np.float32), gt) == 0.0


def test_fully_labeled_metrics_keep_existing_values():
    gt = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.uint8)
    pred = np.array([[0, 1, 1, 0], [2, 3, 3, 3]], dtype=np.uint8)
    metrics = seg_metrics.evaluate(pred, gt)
    assert metrics["iou"] == {"background": 1 / 3, "paper": 1 / 3, "hand": 1 / 2, "ink": 2 / 3}
    assert metrics["accuracy"] == 5 / 8
    assert metrics["valid_px"] == 8 and metrics["ignored_px"] == 0


def test_torch_loss_ignores_unknown_and_is_safe_for_all_unknown():
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [[[[3.0, 0.0]], [[0.0, 3.0]], [[0.0, 0.0]], [[0.0, 0.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1]]], dtype=torch.long)
    weights = torch.ones(4)
    baseline = combined_loss(logits, target, weights)
    sparse = combined_loss(logits, torch.tensor([[[0, IGNORE_INDEX]]]), weights)
    single = combined_loss(logits[:, :, :, :1], target[:, :, :1], weights)
    assert sparse.item() == pytest.approx(single.item())
    assert baseline.item() != pytest.approx(sparse.item())

    all_unknown = combined_loss(logits, torch.full_like(target, IGNORE_INDEX), weights)
    assert torch.isfinite(all_unknown)
    assert all_unknown.item() == 0.0
    all_unknown.backward()
    assert logits.grad is not None


def test_validation_without_any_labeled_pixels_fails_before_checkpoint_selection():
    with pytest.raises(ValueError, match="não contém pixels rotulados válidos"):
        require_valid_evaluation({"valid_px": 0, "mean_iou": None})
    with pytest.raises(ValueError, match="não contém pixels rotulados válidos"):
        require_valid_evaluation({"valid_px": 10, "mean_iou": None})
    metrics = {"valid_px": 1, "mean_iou": 0.5}
    assert require_valid_evaluation(metrics) is metrics


def test_ink_tolerance_is_undefined_without_annotated_ink():
    no_ink = np.zeros((5, 5), dtype=np.uint8)
    pred = no_ink.copy()
    assert seg_metrics.ink_tolerance_metrics(pred, no_ink)["f1"] is None

    trainer_metrics = trainer_ink_tolerance_metrics(pred, no_ink)
    assert all(values["precision"] is None and values["recall"] is None for values in trainer_metrics.values())

    false_positive = no_ink.copy()
    false_positive[2, 2] = 3
    tolerance = seg_metrics.ink_tolerance_metrics(false_positive, no_ink)
    assert tolerance["precision"] == 0.0
    assert tolerance["recall"] is None
    assert tolerance["f1"] == 0.0
    trainer_tolerance = trainer_ink_tolerance_metrics(false_positive, no_ink)
    assert all(values["precision"] == 0.0 and values["recall"] is None for values in trainer_tolerance.values())

    ink = no_ink.copy()
    ink[2, 2] = 3
    supported = seg_metrics.ink_tolerance_metrics(ink, ink)
    assert supported["f1"] == 1.0


def test_scenario_aggregation_excludes_unsupported_ink_scores():
    no_ink = np.zeros((5, 5), dtype=np.uint8)
    ink = no_ink.copy()
    ink[2, 2] = 3
    result = seg_metrics.evaluate_scenarios(
        {"01_clean_sheet": no_ink, "02_sheet_with_drawing": ink},
        {"01_clean_sheet": no_ink, "02_sheet_with_drawing": ink},
    )
    assert result["per_scenario"]["01_clean_sheet"]["ink_tolerance"]["f1"] is None
    assert result["aggregate"]["mean_ink_tolerance_f1"] == 1.0
    assert result["aggregate"]["n_evaluated"] == 2
