#!/usr/bin/env python3
"""Compare pixel-mask proposals against a reviewed pilot (not ground truth).

Runs the OpenCV heuristic, an optional four-class ONNX model, and an
optional ink-specialist ONNX model on the same source images. Metrics ignore
unknown reference pixels; abstaining proposal pixels are also excluded and
reported through coverage so that abstention cannot be hidden.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_DATA_ROOT = ROOT
if not (LOCAL_DATA_ROOT / "captures/triage/training_ready/expert_consensus_20261030_v1/samples.csv").is_file():
    # Git worktrees intentionally do not contain the ignored capture data.
    # During local development, reuse the parent's original checkout if it
    # carries that data; published use should pass an explicit --pack path.
    candidate = ROOT.parents[1]
    if (candidate / "captures/triage/training_ready/expert_consensus_20261030_v1/samples.csv").is_file():
        LOCAL_DATA_ROOT = candidate
DEFAULT_PACK = LOCAL_DATA_ROOT / "captures/triage/training_ready/expert_consensus_20261030_v1"
DEFAULT_REVIEW = LOCAL_DATA_ROOT / "captures/triage/pseudo_training_candidate/calibration_30_expert_source_review.csv"
DEFAULT_SEGFORMER = Path("models/kaggle-real-pilot-v1/model.onnx")
DEFAULT_INK = Path("models/draw_ink_v2.onnx")
CLASSES = ("background", "paper", "hand", "ink")
COLORS = {0: (25, 25, 25), 1: (190, 190, 190), 2: (40, 80, 230), 3: (220, 100, 30), 255: (200, 0, 210)}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(image)
    for value, bgr in COLORS.items():
        color[mask == value] = bgr
    out = cv2.addWeighted(image, 0.62, color, 0.38, 0)
    unknown = mask == 255
    out[unknown] = (out[unknown].astype(np.uint16) * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
    return out


def confusion_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    if pred.shape != ref.shape or pred.ndim != 2:
        raise ValueError(f"mask dimensions differ: proposal {pred.shape}, reviewed {ref.shape}")
    if not np.isin(pred, (0, 1, 2, 3, 255)).all():
        raise ValueError("proposal has unexpected class values")
    if not np.isin(ref, (0, 1, 2, 3, 255)).all():
        raise ValueError("review mask has unexpected class values")
    ref_valid = ref != 255
    valid = ref_valid & (pred != 255)
    result = {
        "reference_labeled_pixels": int(ref_valid.sum()),
        "compared_pixels": int(valid.sum()),
        "coverage": float(valid.sum() / ref_valid.sum()) if ref_valid.any() else None,
        "classes": {},
    }
    for cls, name in enumerate(CLASSES):
        tp = int(((pred == cls) & (ref == cls) & valid).sum())
        fp = int(((pred == cls) & (ref != cls) & valid).sum())
        fn = int(((pred != cls) & (ref == cls) & valid).sum())
        den = tp + fp + fn
        result["classes"][name] = {
            "iou": tp / den if den else None,
            "dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
            "support_pixels": int(((ref == cls) & ref_valid).sum()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
    return result


def load_review_tags(path: Path) -> dict[tuple[str, str], set[str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        return {
            (row["session_id"], row["frame_index"]): {
                item for item in row.get("visible_tags", "").split(";") if item
            }
            for row in rows
        }


def infer_four_class(session, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # This is the preprocessing written into training/train.py: square 512,
    # RGB, ImageNet normalization. Restore class IDs at original dimensions.
    from training.train import IMAGENET_MEAN, IMAGENET_STD

    image = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: image.transpose(2, 0, 1)[None]})[0][0]
    # Match training evaluation: bilinearly resize logits to input resolution,
    # then classify. Upsampling argmax labels would change thin boundaries.
    logits = np.stack([
        cv2.resize(channel, (512, 512), interpolation=cv2.INTER_LINEAR)
        for channel in logits
    ])
    labels = logits.argmax(axis=0).astype(np.uint8)
    shifted = logits - logits.max(axis=0, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    confidence = probabilities.max(axis=0).astype(np.float32)
    size = (image_rgb.shape[1], image_rgb.shape[0])
    labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
    confidence = cv2.resize(confidence, size, interpolation=cv2.INTER_NEAREST)
    return labels, confidence


def cascade_refine(
    classical: np.ndarray,
    neural: np.ndarray,
    confidence: np.ndarray,
    context: dict[str, np.ndarray | bool],
    tags: set[str],
    minimum_confidence: float,
) -> np.ndarray:
    """Fill a conservative subset of OpenCV unknown pixels with ML proposals.

    Existing OpenCV decisions are immutable. Context from the first stage
    bounds predicted background, paper, and hand pixels. Class 3 is disabled
    on frames tagged with printed text because the model cannot separate print
    from user ink reliably in this pilot.
    """
    if classical.shape != neural.shape or classical.shape != confidence.shape:
        raise ValueError("cascade masks and confidence must have identical dimensions")
    paper_region = np.asarray(context["paper_region"], dtype=bool)
    paper_candidate = np.asarray(context["paper_candidate"], dtype=bool)
    paper_core = np.asarray(context["paper_core"], dtype=bool)
    likely_skin = np.asarray(context["likely_skin"], dtype=bool)
    if any(region.shape != classical.shape for region in (paper_region, paper_candidate, paper_core, likely_skin)):
        raise ValueError("OpenCV context masks must match the proposal dimensions")
    out = classical.copy()
    uncertain = classical == 255
    high_confidence = confidence >= minimum_confidence
    paper_detected = bool(context["paper_detected"])
    hand_tagged = bool({"hand_on_paper", "hand_elsewhere"} & tags)
    allowed = {
        0: paper_detected & ~paper_region & ~likely_skin,
        1: paper_candidate & ~likely_skin,
        2: hand_tagged & likely_skin,
        3: paper_core & ("user_added_ink" in tags) & ("printed_text" not in tags),
    }
    for class_id, region in allowed.items():
        if not isinstance(region, np.ndarray):
            region = np.full(classical.shape, bool(region), dtype=bool)
        fill = uncertain & high_confidence & (neural == class_id) & region
        out[fill] = class_id
    return out


def infer_ink(session, image_rgb: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    image = image_rgb.astype(np.float32) / 255.0
    logits = session.run([output_name], {input_name: image.transpose(2, 0, 1)[None]})[0][0]
    ink = (logits.argmax(axis=0) == 2).astype(np.uint8)
    return cv2.resize(ink, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)


def summarize(per_frame: list[dict]) -> dict:
    out = {"frames": len(per_frame), "mean_coverage": None, "classes": {}}
    coverages = [row["coverage"] for row in per_frame if row["coverage"] is not None]
    out["mean_coverage"] = float(np.mean(coverages)) if coverages else None
    for name in CLASSES:
        vals = [row["classes"][name] for row in per_frame]
        out["classes"][name] = {
            metric: float(np.mean([v[metric] for v in vals if v[metric] is not None]))
            if any(v[metric] is not None for v in vals) else None
            for metric in ("iou", "dice")
        }
        out["classes"][name]["reference_support_pixels"] = int(sum(v["support_pixels"] for v in vals))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--review-tags", type=Path, default=DEFAULT_REVIEW)
    segformer_default = Path(os.environ["DRAW_SEGFORMER_MODEL"]) if os.environ.get("DRAW_SEGFORMER_MODEL") else ROOT / DEFAULT_SEGFORMER
    parser.add_argument("--segformer-model", type=Path, default=segformer_default)
    parser.add_argument("--ink-model", type=Path, default=ROOT / DEFAULT_INK)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "captures/triage/pixelmask-pilot")
    parser.add_argument("--no-segformer", action="store_true")
    parser.add_argument("--no-ink-model", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true", help="reuse existing prediction PNGs in output-dir")
    parser.add_argument("--refine-confidence", type=float, default=0.90,
                        help="minimum SegFormer softmax confidence to fill OpenCV unknown pixels (default: 0.90)")
    args = parser.parse_args()
    for name in ("pack", "review_tags", "segformer_model", "ink_model", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.output_dir.is_relative_to(ROOT):
        parser.error("--output-dir must stay inside the repository")
    if not 0.5 <= args.refine_confidence <= 1.0:
        parser.error("--refine-confidence must be between 0.5 and 1")
    with (args.pack / "samples.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("pilot manifest has no samples")
    tags = load_review_tags(args.review_tags)

    four_class = None
    if not args.no_segformer:
        import onnxruntime as ort

        if args.segformer_model is None:
            parser.error("pass --segformer-model/DRAW_SEGFORMER_MODEL or use --no-segformer")
        if not args.segformer_model.is_file():
            parser.error(f"four-class model not found: {args.segformer_model}")
        four_class = ort.InferenceSession(str(args.segformer_model), providers=["CPUExecutionProvider"])
        shape = four_class.get_outputs()[0].shape
        if len(shape) != 4 or shape[1] != 4:
            parser.error(f"expected four-class logits output, got {shape}")
    ink_session = None
    if not args.no_ink_model:
        import onnxruntime as ort

        if not args.ink_model.is_file():
            parser.error(f"ink specialist not found: {args.ink_model}")
        ink_session = ort.InferenceSession(str(args.ink_model), providers=["CPUExecutionProvider"])

    from tools.propose_semantic_masks import propose_mask

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = args.output_dir / "prediction_cache.json"
    model_hashes = {
        "segformer": digest(args.segformer_model) if four_class is not None else None,
        "ink_specialist": digest(args.ink_model) if ink_session is not None else None,
    }
    tool_hash = digest(Path(__file__))
    proposer_hash = digest(ROOT / "tools/propose_semantic_masks.py")
    preprocessing_hash = digest(ROOT / "training/train.py")
    input_hashes = {}
    for row in rows:
        image_path = (args.pack / row["image_path"]).resolve()
        input_hashes[f"{row['session_id']}:{row['frame_index']}"] = digest(image_path)
    if args.reuse_predictions:
        if not cache_manifest_path.is_file():
            parser.error("--reuse-predictions requires prediction_cache.json from a completed matching run")
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if (
            cache_manifest.get("input_sha256") != input_hashes
            or cache_manifest.get("model_sha256") != model_hashes
            or cache_manifest.get("tool_sha256") != tool_hash
            or cache_manifest.get("proposer_sha256") != proposer_hash
            or cache_manifest.get("preprocessing_sha256") != preprocessing_hash
        ):
            parser.error("cached predictions do not match the current inputs, models, and tool version")
    metrics_by_method: dict[str, list[dict]] = defaultdict(list)
    review_rows = []
    tiles = []
    for row in rows:
        image_path = (args.pack / row["image_path"]).resolve()
        reference_path = (args.pack / row["mask_path"]).resolve()
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or reference is None:
            raise FileNotFoundError(f"unreadable pilot image/mask: {image_path}, {reference_path}")
        if digest(image_path) != row["image_sha256"] or digest(reference_path) != row["mask_sha256"]:
            raise ValueError(f"pilot checksum mismatch at {row['session_id']} frame {row['frame_index']}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        frame_tags = tags.get((row["session_id"], row["frame_index"]), set())
        proposal_cv, _, context = propose_mask(image_bgr, frame_tags, return_context=True)
        predictions = {"opencv": proposal_cv}
        stem = f"{row['session_id']}_frame_{int(row['frame_index']):08d}"
        if four_class is not None:
            cached = args.output_dir / f"{stem}_segformer.png"
            confidence_cache = args.output_dir / f"{stem}_segformer_confidence.png"
            cached_mask = cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE) if args.reuse_predictions else None
            if args.reuse_predictions and cached_mask is None:
                raise FileNotFoundError(f"cached segmentation prediction missing: {cached}")
            cached_confidence = cv2.imread(str(confidence_cache), cv2.IMREAD_UNCHANGED) if args.reuse_predictions else None
            if args.reuse_predictions and cached_confidence is None:
                raise FileNotFoundError(f"cached segmentation confidence missing: {confidence_cache}")
            if cached_mask is None:
                segformer_mask, confidence = infer_four_class(four_class, image_rgb)
            else:
                segformer_mask = cached_mask
                confidence_scale = 65535.0 if cached_confidence.dtype == np.uint16 else 255.0
                cached_confidence = cached_confidence.astype(np.float32)
                confidence = cached_confidence / confidence_scale
            predictions["segformer"] = segformer_mask
            confidence_path = args.output_dir / f"{stem}_segformer_confidence.png"
            if not args.reuse_predictions:
                encoded_confidence = np.clip(confidence * 65535.0, 0, 65535).astype(np.uint16)
                if not cv2.imwrite(str(confidence_path), encoded_confidence):
                    raise OSError(f"failed to write confidence map: {confidence_path}")
            # Abstaining proposal: accept only pixels where both methods
            # agree. It narrows review; it does not establish label truth.
            agreement = np.full(proposal_cv.shape, 255, dtype=np.uint8)
            agree = (proposal_cv == segformer_mask) & (proposal_cv != 255)
            agreement[agree] = proposal_cv[agree]
            predictions["agreement"] = agreement
            # Cascade: OpenCV is the first-stage mask and owns every pixel it
            # labeled. SegFormer may only fill its 255 abstentions when its
            # maximum softmax probability clears the requested threshold.
            predictions["cascade"] = cascade_refine(
                proposal_cv, segformer_mask, confidence, context,
                frame_tags, args.refine_confidence,
            )
        if ink_session is not None:
            proposal_hybrid = proposal_cv.copy()
            # Respect visible hand/unknown and only allow the specialist to
            # contribute inside the heuristic's confirmed paper region.
            cached = args.output_dir / f"{stem}_opencv_plus_inkv2.png"
            cached_mask = cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE) if args.reuse_predictions else None
            if args.reuse_predictions and cached_mask is None:
                raise FileNotFoundError(f"cached ink-specialist prediction missing: {cached}")
            if cached_mask is not None:
                proposal_hybrid = cached_mask
                ink = None
            else:
                ink = infer_ink(ink_session, image_rgb)
            if ink is not None:
                proposal_hybrid[(proposal_hybrid == 1) & (ink > 0)] = 3
            predictions["opencv_plus_inkv2"] = proposal_hybrid

        cv2.imwrite(str(args.output_dir / f"{stem}_reference.png"), reference)
        cv2.imwrite(str(args.output_dir / f"{stem}_image.jpg"), image_bgr)
        tile_parts = [cv2.resize(image_bgr, (320, 180))]
        labels = ["image"]
        for method, prediction in predictions.items():
            score = confusion_metrics(prediction, reference)
            metrics_by_method[method].append(score)
            prediction_path = args.output_dir / f"{stem}_{method}.png"
            overlay_path = args.output_dir / f"{stem}_{method}_overlay.jpg"
            cv2.imwrite(str(prediction_path), prediction)
            cv2.imwrite(str(overlay_path), overlay(image_bgr, prediction))
            tile_parts.append(cv2.resize(overlay(image_bgr, prediction), (320, 180)))
            labels.append(method)
            review_rows.append({
                "session_id": row["session_id"], "frame_index": row["frame_index"],
                "method": method, "coverage": score["coverage"],
                "filled_from_opencv_unknown_pixels": (
                    int(((proposal_cv == 255) & (prediction != 255)).sum())
                    if method == "cascade" else None
                ),
                "remaining_unknown_pixels": int((prediction == 255).sum()),
                **{f"{cls}_iou": score["classes"][cls]["iou"] for cls in CLASSES},
                **{f"{cls}_support_px": score["classes"][cls]["support_pixels"] for cls in CLASSES},
                "proposal_path": str(prediction_path.relative_to(ROOT)),
                "overlay_path": str(overlay_path.relative_to(ROOT)),
            })
        ref_tile = cv2.resize(overlay(image_bgr, reference), (320, 180))
        tile_parts.insert(1, ref_tile)
        labels.insert(1, "reviewed consensus")
        canvas = np.concatenate(tile_parts, axis=1)
        cv2.putText(canvas, f"frame {row['frame_index']}: " + " | ".join(labels), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(canvas)

    report = {
        "status": "exploratory proposal comparison",
        "reference_status": "three-agent visual review of masks copied byte-for-byte from the earlier OpenCV proposal; not independent pixel ground truth; no holdout",
        "samples": len(rows),
        "model_paths": {
            "segformer": str(args.segformer_model) if four_class is not None else None,
            "segformer_sha256": digest(args.segformer_model) if four_class is not None else None,
            "ink_specialist": str(args.ink_model) if ink_session is not None else None,
            "ink_specialist_sha256": digest(args.ink_model) if ink_session is not None else None,
        },
        "tool_sha256": tool_hash,
        "proposer_sha256": proposer_hash,
        "preprocessing_sha256": preprocessing_hash,
        "metric_rule": "reference 255 excluded; proposal 255 also abstains and is excluded; coverage reported separately; scores measure agreement with the reviewed OpenCV-derived masks, not independent accuracy",
        "cascade_rule": "OpenCV labels are preserved; SegFormer fills only OpenCV 255 pixels at or above refine_confidence within appearance-qualified paper/skin context; printed-text frames cannot receive new ink labels",
        "refine_confidence": args.refine_confidence,
        "methods": {method: summarize(values) for method, values in metrics_by_method.items()},
    }
    cache_manifest_path.write_text(json.dumps({
        "input_sha256": input_hashes,
        "model_sha256": model_hashes,
        "tool_sha256": tool_hash,
        "proposer_sha256": proposer_hash,
        "preprocessing_sha256": preprocessing_hash,
    }, indent=2), encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    if tiles:
        sheet = np.concatenate(tiles, axis=0)
        cv2.imwrite(str(args.output_dir / "proposal_review_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(json.dumps(report, indent=2))
    print(f"Artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
