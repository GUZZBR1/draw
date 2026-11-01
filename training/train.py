"""Treinamento do modelo de segmentação do Draw (SegFormer-B0, 4 classes).

Escopo: dados -> treino -> validação -> exportação ONNX -> documentação.
FORA do escopo (Codex, depois): webcam, integração no app, tracking
temporal, memória de traços, oclusão em tempo real, interface, CAD,
simulador.

Classes: 0=background, 1=paper, 2=hand, 3=ink.
Máscaras podem usar 255 somente para pixels ignorados pela loss e pelas métricas.
Regra de rótulo: o que a câmera vê. Mão sobre traço = hand, nunca ink.
ink_full_mask (opcional) NÃO é classe nem entrada: é só para avaliar
a memória temporal futura.

Contrato com draw/neural.py (atual, fase 2):
  - entrada: RGB 512x512, normalização ImageNet (mean 0.485/0.456/0.406,
    std 0.229/0.224/0.225), input ONNX chamado "image";
  - saída antiga: mask_logits de 2 canais + presence_logit;
  - saída nova: logits de 4 canais, output ONNX chamado "logits".
  - compatibilidade (decisão documentada, integração pelo Codex):
    paper_mask = (argmax == 1); hand_mask = (argmax == 2);
    ink_mask = (argmax == 3); paper_presence é DERIVADO da máscara
    (fração de pixels paper acima do limiar de integração), não uma
    cabeça auxiliar. O modelo não foi simplificado para caber no
    contrato antigo de 2 canais.

Uso:
  # Kaggle (GPU, treino completo):
  python training/train.py --data-dir /kaggle/input/<dataset>/dataset/synthetic \
      --config training/config.json --out-dir /kaggle/working/out
  # Smoke test (sem GPU, sem treino completo; ver função smoke()):
  python training/train.py --config training/config.json --smoke \
      --smoke-dir /tmp/draw-smoke

Imports pesados (torch, transformers, albumentations, onnxruntime) são
lazy, dentro das funções que os usam. O topo deste arquivo só usa
stdlib + numpy + cv2, para que config, pareamento, pesos e
métricas possam ser testados sem GPU e sem torch instalado.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

import cv2
import numpy as np

CLASSES = {0: "background", 1: "paper", 2: "hand", 3: "ink"}
NUM_CLASSES = 4
IGNORE_INDEX = 255
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
# Mesma normalização de draw/neural.py (_MEAN/_STD).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- config
def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    tr = cfg["training"]
    assert cfg["model"]["num_classes"] == NUM_CLASSES, "config deve ter 4 classes"
    assert len(cfg["loss"]["cross_entropy_class_weights"]) == NUM_CLASSES
    return cfg


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_exists(name):
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def hash_input_tree(root):
    """Hash every file in a mounted dataset and return a stable inventory."""
    root = Path(root)
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def validate_source_splits(manifest_path, expected_counts):
    """Require a generator manifest and disjoint original sources per split."""
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError("split_manifest.json é obrigatório para treino Kaggle rastreável")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("split_manifest.json precisa conter scenes")
    scene_counts = {split: sum(x.get("split") == split for x in scenes) for split in ("train", "val", "test")}
    if scene_counts != expected_counts:
        raise ValueError(f"contagem do split_manifest diverge dos arquivos: {scene_counts} != {expected_counts}")
    source_splits = manifest.get("splits")
    if not isinstance(source_splits, dict) or not source_splits:
        raise ValueError("split_manifest.json precisa identificar fontes originais por split")
    for source_kind, splits in source_splits.items():
        if not all(isinstance(splits.get(split), list) for split in ("train", "val", "test")):
            raise ValueError(f"fontes incompletas no split_manifest: {source_kind}")
        sets = {split: set(splits[split]) for split in ("train", "val", "test")}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"vazamento de fonte {source_kind} entre {left}/{right}: {sorted(overlap)[:3]}")
    scene_source_fields = {
        "backgrounds": "background_source",
        "hands": "hand_source",
        "papers": "paper_source",
    }
    for scene in scenes:
        split = scene.get("split")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split inválido em scene {scene.get('id')}: {split}")
        for kind, field in scene_source_fields.items():
            source = scene.get(field)
            if source is not None and kind in source_splits and source not in source_splits[kind][split]:
                raise ValueError(f"fonte {field} da scene {scene.get('id')} diverge do split_manifest")
    return manifest


# ---------------------------------------------------------------- dataset
def pair_images_masks(split_dir):
    """Pareia images/<nome> com masks/<mesmo nome base>.png.

    Aceita também o layout plano do gerador (tools/gen_synthetic.py):
    <split>/scene_*.png + <split>/scene_*_mask.png +
    <split>/scene_*_ink_full.png (opcional). Sem essa aceitação, o
    treino falhava com "split sem images/ ou masks/" sobre a saída
    do gerador — divergência encontrada no QA e corrigida aqui.

    Retorna lista de (caminho_imagem, caminho_mascara, caminho_ink_full|None).
    ink_full é opcional e nunca entra no treino.
    """
    split_dir = Path(split_dir)
    img_dir, mask_dir = split_dir / "images", split_dir / "masks"
    if img_dir.is_dir() and mask_dir.is_dir():
        pairs = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            mask_path = mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                raise FileNotFoundError(f"máscara ausente para {img_path}")
            ink_full = split_dir / "ink_full" / (img_path.stem + ".png")
            pairs.append((img_path, mask_path, ink_full if ink_full.exists() else None))
    else:
        # Layout plano do gerador.
        pairs = []
        for img_path in sorted(split_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if img_path.stem.endswith("_mask") or img_path.stem.endswith("_ink_full"):
                continue
            mask_path = split_dir / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                continue
            ink_full = split_dir / (img_path.stem + "_ink_full.png")
            pairs.append((img_path, mask_path, ink_full if ink_full.exists() else None))
    if not pairs:
        raise ValueError(f"nenhum par imagem/máscara em {split_dir}")
    return pairs


def validate_mask(mask):
    """Máscara válida: classes 0..3 e 255 para pixels deliberadamente ignorados."""
    values = np.unique(mask)
    bad = [int(v) for v in values if v not in (0, 1, 2, 3, IGNORE_INDEX)]
    if bad:
        raise ValueError(f"máscara com valores inválidos (permitidos: 0..3 e 255=ignorado): {bad}")
    return True


def load_mask(mask_path):
    """Lê máscara em grayscale; 255 significa pixel sem supervisão."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"máscara ilegível: {mask_path}")
    validate_mask(mask)
    return mask.astype(np.int64)


def load_image_rgb(image_path):
    """Lê imagem com cv2 e converte BGR -> RGB."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"imagem ilegível: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pixel_counts(pairs, max_files=None):
    """Conta pixels supervisionados por classe; pixels 255 não entram."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, mask_path, _ in pairs[:max_files] if max_files else pairs:
        mask = load_mask(mask_path)
        for c in range(NUM_CLASSES):
            counts[c] += int((mask == c).sum())
    return counts


def compute_class_weights(counts, cap=10.0):
    """Peso ~ inverso da frequência, com teto (evita ink dominar tudo)."""
    total = counts.sum()
    if total == 0:
        return [1.0] * NUM_CLASSES
    freq = counts / total
    weights = 1.0 / np.maximum(freq, 1e-6)
    weights = weights / weights.min()
    return [float(min(w, cap)) for w in weights]


# ---------------------------------------------------------------- métricas
def per_class_metrics(pred, target):
    """IoU, Dice, precision, recall por classe (numpy, sem torch).

    pred/target: arrays 2D de inteiros 0..3/255, mesmo shape. Pixels 255
    no target são excluídos de todos os denominadores e suportes.
    """
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("pred e target precisam ter o mesmo formato")
    validate_mask(pred)
    validate_mask(target)
    if np.any(np.asarray(pred) == IGNORE_INDEX):
        raise ValueError("predição não pode conter pixels ignorados (255)")
    valid = target != IGNORE_INDEX
    out = {}
    valid = target != IGNORE_INDEX
    for c in range(NUM_CLASSES):
        p = (pred == c) & valid
        t = (target == c) & valid
        tp = int((p & t).sum())
        fp = int((p & ~t).sum())
        fn = int((~p & t).sum())
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        pred_support = tp + fp
        target_support = tp + fn
        out[CLASSES[c]] = {
            "iou": tp / denom_iou if denom_iou else None,
            "dice": 2 * tp / denom_dice if denom_dice else None,
            "precision": tp / pred_support if pred_support else (0.0 if target_support else None),
            "recall": tp / target_support if target_support else None,
            "support_px": target_support,
        }
    return out


def ink_tolerance_metrics(pred, target, radii=(2, 3)):
    """Precision/recall de ink com tolerância espacial.

    Dilata predição e ground truth de ink pelo raio dado: um traço fino
    deslocado 1-2 px é visualmente correto, mas a IoU estrita o pune.
    A IoU estrita continua sendo reportada ao lado; esta métrica não
    a substitui, apenas a complementa.
    """
    out = {}
    valid = target != IGNORE_INDEX
    pred_ink = ((pred == 3) & valid).astype(np.uint8)
    target_ink = ((target == 3) & valid).astype(np.uint8)
    for r in radii:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        p_dil = (cv2.dilate(pred_ink, kernel) > 0) & valid
        t_dil = (cv2.dilate(target_ink, kernel) > 0) & valid
        # precision tolerante: predito que cai perto do traço real;
        # recall tolerante: traço real coberto por predição próxima.
        pred_valid = (pred_ink > 0) & valid
        target_valid = (target_ink > 0) & valid
        pred_n = int(pred_valid.sum())
        target_n = int(target_valid.sum())
        matched_pred = int((pred_valid & t_dil).sum())
        matched_target = int((target_valid & p_dil).sum())
        out[f"radius_{r}px"] = {
            "precision": matched_pred / pred_n if pred_n else (0.0 if target_n else None),
            "recall": matched_target / target_n if target_n else None,
        }
    return out


def dice_loss_numpy(prob, target, smooth=1e-6):
    """Dice loss multiclasse em numpy (espelho da versão torch).

    prob: (C, H, W) probabilidades; target: (H, W) inteiros 0..3/255.
    Pixels ignorados são removidos da probabilidade e do alvo.
    """
    valid = target != IGNORE_INDEX
    if not np.any(valid):
        return 0.0
    losses = []
    for c in range(NUM_CLASSES):
        p = prob[c][valid].ravel()
        t = (target[valid] == c).astype(np.float32).ravel()
        inter = float((p * t).sum())
        losses.append(1.0 - (2 * inter + smooth) / (p.sum() + t.sum() + smooth))
    return float(np.mean(losses))


# ---------------------------------------------------------------- torch
def build_model(cfg):
    """SegFormer-B0 pré-treinado, cabeça nova de 4 classes (lazy torch)."""
    from transformers import SegformerForSemanticSegmentation

    model_cfg = cfg["model"]
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_cfg["backbone"],
        num_labels=model_cfg["num_classes"],
        ignore_mismatched_sizes=True,  # cabeça nova: 4 classes, não as do ImageNet
    )
    return model


def build_dataset(data_dir, split, cfg, augment=False, extra_pairs=None):
    """Dataset torch. Import lazy: sem torch, esta função não existe."""
    import torch
    from torch.utils.data import Dataset

    pairs = pair_images_masks(Path(data_dir) / split)
    if extra_pairs:
        pairs.extend(extra_pairs)
    size = cfg["model"]["input"]["height"]
    aug = None
    if augment:
        try:
            import albumentations as A

            aug = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.2),
                    A.Blur(blur_limit=3, p=0.2),
                ]
            )
        except ImportError:
            print("AVISO: albumentations ausente; treino sem augmentation.", file=sys.stderr)

    class SegDataset(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            img_path, mask_path, _ = pairs[i]
            image = load_image_rgb(img_path)
            mask = load_mask(mask_path)
            if aug is not None:
                out = aug(image=image, mask=mask)
                image, mask = out["image"], out["mask"]
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            image = image.astype(np.float32) / 255.0
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
            image = np.transpose(image, (2, 0, 1))
            return (
                torch.from_numpy(image),
                torch.from_numpy(mask.astype(np.int64)),
            )

    return SegDataset()


def combined_loss(logits, target, class_weights, ce_weight=0.5, dice_weight=0.5):
    """CE ponderada + Dice multiclasse; target=255 não supervisiona o pixel."""
    import torch
    import torch.nn.functional as F

    valid = target != IGNORE_INDEX
    if not torch.any(valid):
        # Mantém a loss ligada ao grafo para que backward seja válido, mas
        # produz gradiente zero quando o lote inteiro foi deliberadamente ignorado.
        return logits.sum() * 0.0

    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=IGNORE_INDEX)
    prob = torch.softmax(logits, dim=1)
    valid_flat = valid.reshape(target.shape[0], -1)
    sample_is_valid = valid_flat.any(dim=1)
    dice = torch.zeros(target.shape[0], dtype=logits.dtype, device=logits.device)
    for c in range(NUM_CLASSES):
        p = prob[:, c].reshape(prob.shape[0], -1)
        p = p * valid_flat
        t = ((target == c) & valid).float().reshape(target.shape[0], -1)
        inter = (p * t).sum(dim=1)
        dice = dice + (1.0 - (2 * inter + 1e-6) / (p.sum(dim=1) + t.sum(dim=1) + 1e-6))
    dice = dice / NUM_CLASSES
    # Correção do reteste no Kaggle: sem esta média, dice tem shape [batch]
    # e loss.backward() falha com "grad can be implicitly created only
    # for scalar outputs". A loss tem que ser escalar.
    dice = dice[sample_is_valid].mean()
    return ce_weight * ce + dice_weight * dice


def train_one_epoch(model, loader, optimizer, class_weights, device, scaler=None):
    import torch

    model.train()
    total, n = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        if not (masks != IGNORE_INDEX).any():
            continue
        optimizer.zero_grad()
        logits = model(pixel_values=images).logits
        # SegFormer pode devolver logits em resolução menor: interpola para a máscara.
        if logits.shape[-2:] != masks.shape[-2:]:
            import torch.nn.functional as F

            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = combined_loss(logits, masks, class_weights)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    if n == 0:
        raise ValueError(
            "época sem nenhum lote supervisionado: todas as máscaras contêm apenas 255"
        )
    return total / n


def evaluate(model, loader, device):
    """Avalia um loader inteiro; devolve métricas agregadas por classe."""
    import torch

    model.eval()
    agg = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(NUM_CLASSES)}
    ink_pred_list, ink_target_list = [], []
    supervised_px, ignored_px = 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(pixel_values=images).logits
            if logits.shape[-2:] != masks.shape[-2:]:
                import torch.nn.functional as F

                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1).cpu().numpy()
            target = masks.numpy()
            valid = target != IGNORE_INDEX
            supervised_px += int(valid.sum())
            ignored_px += int((~valid).sum())
            for c in range(NUM_CLASSES):
                p, t = (pred == c) & valid, (target == c) & valid
                agg[c]["tp"] += int((p & t).sum())
                agg[c]["fp"] += int((p & ~t).sum())
                agg[c]["fn"] += int((~p & t).sum())
            ink_pred_list.append(pred)
            ink_target_list.append(target)
    if supervised_px == 0:
        raise ValueError("avaliação sem pixels supervisionados: todas as máscaras são 255")
    metrics = {}
    for c in range(NUM_CLASSES):
        tp, fp, fn = agg[c]["tp"], agg[c]["fp"], agg[c]["fn"]
        metrics[CLASSES[c]] = {
            "iou": tp / (tp + fp + fn) if (tp + fp + fn) else None,
            "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
            "precision": tp / (tp + fp) if (tp + fp) else (0.0 if (tp + fn) else None),
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "support_px": int(agg[c]["tp"] + agg[c]["fn"]),
        }
    metrics["supervised_px"] = supervised_px
    # Keep the metric key consumed by the existing training-zone gate.
    metrics["valid_px"] = supervised_px
    metrics["ignored_px"] = ignored_px
    pred_all = np.concatenate([p.ravel() for p in ink_pred_list]) if ink_pred_list else np.array([])
    target_all = np.concatenate([t.ravel() for t in ink_target_list]) if ink_target_list else np.array([])
    # Métrica de ink com tolerância precisa de imagem 2D; usa o primeiro
    # lote como amostra representativa (o relatório completo usa o
    # avaliador de cenários fixos, abaixo).
    metrics["ink_tolerance_sample"] = {}
    if ink_pred_list:
        metrics["ink_tolerance_sample"] = ink_tolerance_metrics(ink_pred_list[0][0], ink_target_list[0][0])
    supported_ious = [
        metrics[CLASSES[c]]["iou"]
        for c in range(NUM_CLASSES)
        if metrics[CLASSES[c]]["iou"] is not None
    ]
    metrics["mean_iou"] = float(np.mean(supported_ious)) if supported_ious else None
    return metrics


def require_valid_evaluation(metrics, split_name):
    """Fail closed when an evaluation has missing support or invalid metrics."""
    if not isinstance(metrics, dict) or metrics.get("supervised_px", 0) <= 0:
        raise ValueError(f"split {split_name} sem pixels supervisionados")
    if metrics.get("mean_iou") is None or not math.isfinite(metrics["mean_iou"]):
        raise ValueError(f"split {split_name} sem mean_iou válido")
    for class_name in CLASSES.values():
        class_metrics = metrics.get(class_name)
        if not isinstance(class_metrics, dict) or class_metrics.get("iou") is None:
            raise ValueError(f"split {split_name} sem suporte IoU para classe {class_name}")
        for metric_name in ("iou", "dice", "precision", "recall"):
            value = class_metrics.get(metric_name)
            if value is None or not math.isfinite(value):
                raise ValueError(f"split {split_name} com {metric_name} inválida para {class_name}")
    return metrics


# ---------------------------------------------------------------- ONNX
def _make_onnx_dummy(model, size):
    """Create the export sample on the same device as the model parameters."""
    import torch

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    return torch.randn(1, 3, size, size, device=device)


def export_onnx(model, out_path, cfg):
    """Exporta para ONNX com o contrato do Draw: input 'image', output 'logits'."""
    import torch

    model.eval()
    size = cfg["model"]["input"]["height"]
    dummy = _make_onnx_dummy(model, size)
    # dynamo=False: exportador legado. O exportador dynamo padrão do torch
    # 2.11 exige onnxscript, que o ambiente do Kaggle não tem — sem isso,
    # o treino inteiro falhava na exportação, depois das épocas, e perdia
    # até o metrics.json. O exportador legado usa só o pacote onnx.
    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": [0], "logits": [0]},
        opset_version=cfg["export"]["opset_version"],
        do_constant_folding=True,
        dynamo=False,
    )
    return out_path


def validate_onnx(
    model, onnx_path, loader, device,
    max_diff_tolerance=1e-3, min_argmax_iou_tolerance=0.99,
):
    """Compare PyTorch and ONNX logits and argmax masks on the same batch."""
    try:
        import onnxruntime as ort
        import torch
    except ImportError:
        print("AVISO: onnxruntime ausente; validação ONNX pulada.", file=sys.stderr)
        return {"status": "skipped", "reason": "onnxruntime ausente"}
    model.eval()
    images, _ = next(iter(loader))
    with torch.no_grad():
        torch_logits = model(pixel_values=images.to(device)).logits.detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    image_np = images.numpy()
    # Normalização já feita no dataset; o ONNX recebe o tensor normalizado.
    onnx_logits = session.run(["logits"], {"image": image_np})[0]
    if torch_logits.shape != onnx_logits.shape:
        return {
            "status": "failed",
            "reason": f"shape divergente: torch {torch_logits.shape} vs onnx {onnx_logits.shape}",
        }
    if not np.isfinite(torch_logits).all():
        return {"status": "failed", "reason": "logits PyTorch contêm NaN ou infinito"}
    if not np.isfinite(onnx_logits).all():
        return {"status": "failed", "reason": "logits ONNX contêm NaN ou infinito"}
    max_abs_diff = float(np.abs(torch_logits - onnx_logits).max())
    torch_mask = np.argmax(torch_logits, axis=1)
    onnx_mask = np.argmax(onnx_logits, axis=1)
    argmax_iou = {}
    for class_id, class_name in CLASSES.items():
        pred = onnx_mask == class_id
        target = torch_mask == class_id
        union = int(np.count_nonzero(pred | target))
        argmax_iou[class_name] = (
            float(np.count_nonzero(pred & target) / union) if union else None
        )
    supported = [value for value in argmax_iou.values() if value is not None]
    min_iou = min(supported) if supported else None
    degraded = (
        max_abs_diff > max_diff_tolerance
        or min_iou is None
        or min_iou < min_argmax_iou_tolerance
    )
    return {
        "status": "degraded" if degraded else "ok",
        "max_abs_diff": max_abs_diff,
        "max_diff_tolerance": max_diff_tolerance,
        "argmax_agreement": float(np.mean(torch_mask == onnx_mask)),
        "argmax_iou": argmax_iou,
        "min_argmax_iou": min_iou,
        "min_argmax_iou_tolerance": min_argmax_iou_tolerance,
    }


# ---------------------------------------------------------------- smoke
def smoke(smoke_dir):
    """Smoke test sem GPU e sem treino completo.

    Cria 2 pares sintéticos (imagem + máscara 0..3/255), testa pareamento,
    validação de máscara, contagem de pixels, pesos, métricas numpy,
    pixel ignorado, métrica de ink com tolerância e o contrato ONNX (nomes/shapes).
    Não treina modelo; usa PyTorch CPU se disponível para verificar a loss.
    Retorna 0 se tudo passar.
    """
    import shutil

    root = Path(smoke_dir)
    if root.exists():
        shutil.rmtree(root)
    img_dir, mask_dir = root / "train" / "images", root / "train" / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    # Imagem 1: fundo + retângulo de papel + linha de tinta.
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    mask[20, 12:48] = 3  # traço fino, 1 px
    mask[30:40, 30:40] = 2  # mão sobre o papel
    mask[54:60, 54:60] = IGNORE_INDEX  # área sem rótulo confirmado
    cv2.imwrite(str(img_dir / "scene_0001.png"), img)
    cv2.imwrite(str(mask_dir / "scene_0001.png"), mask)
    # Imagem 2: só fundo (negativo).
    cv2.imwrite(str(img_dir / "scene_0002.png"), np.full((64, 64, 3), 90, dtype=np.uint8))
    cv2.imwrite(str(mask_dir / "scene_0002.png"), np.zeros((64, 64), dtype=np.uint8))

    pairs = pair_images_masks(root / "train")
    assert len(pairs) == 2, pairs
    counts = pixel_counts(pairs)
    ignored = int((load_mask(mask_dir / "scene_0001.png") == IGNORE_INDEX).sum())
    assert ignored == 36, ignored
    assert counts.sum() == 2 * 64 * 64 - ignored, counts
    weights = compute_class_weights(counts)
    assert len(weights) == NUM_CLASSES and all(w > 0 for w in weights), weights
    pred = load_mask(mask_dir / "scene_0001.png")
    pred[pred == IGNORE_INDEX] = 0  # predições são classes 0..3, nunca ignore
    # Predição perfeita exceto 1 px de tinta deslocado: IoU estrita cai,
    # tolerância de 2-3 px deve recuperar.
    pred_shifted = pred.copy()
    pred_shifted[20, 12:48] = 0
    pred_shifted[21, 12:48] = 3
    m = per_class_metrics(pred_shifted, pred)
    assert m["ink"]["iou"] < 1.0, "IoU estrita deveria cair com o deslocamento"
    tol = ink_tolerance_metrics(pred_shifted, pred)
    assert tol["radius_2px"]["recall"] > m["ink"]["recall"], (tol, m["ink"])

    # Predições arbitrárias em pixels 255 não podem alterar métricas nem
    # denominadores das classes supervisionadas.
    ignored_target = pred.copy()
    ignored_target[54:60, 54:60] = IGNORE_INDEX
    ignored_pred = ignored_target.copy()
    ignored_pred[54:60, 54:60] = 3
    ignored_metrics = per_class_metrics(ignored_pred, ignored_target)
    assert ignored_metrics["ink"]["iou"] == 1.0, ignored_metrics["ink"]
    assert ignored_metrics["ink"]["support_px"] == 36, ignored_metrics["ink"]
    ignored_tol = ink_tolerance_metrics(ignored_pred, ignored_target)
    assert ignored_tol["radius_2px"]["precision"] == 1.0, ignored_tol
    uniform_prob = np.full((NUM_CLASSES, 64, 64), 0.25, dtype=np.float32)
    changed_ignored_prob = uniform_prob.copy()
    changed_ignored_prob[0:3, 54:60, 54:60] = 0.0
    changed_ignored_prob[3, 54:60, 54:60] = 1.0
    assert dice_loss_numpy(uniform_prob, ignored_target) == dice_loss_numpy(
        changed_ignored_prob, ignored_target
    ), "probabilidades em pixels ignorados não devem alterar Dice numpy"

    try:
        import torch
    except ImportError:
        print("AVISO: torch ausente; smoke de loss torch com pixel ignorado pulado.", file=sys.stderr)
    else:
        # ONNX export receives a CUDA model after Kaggle training. The dummy
        # input must follow that model's device; CPU is still the fallback for
        # parameterless modules and the local smoke path.
        for test_device in ("cpu", "meta"):
            probe = torch.nn.Linear(1, 1).to(test_device)
            export_input = _make_onnx_dummy(probe, 8)
            assert export_input.device == next(probe.parameters()).device, (
                test_device, export_input.device, next(probe.parameters()).device
            )
        empty_probe = torch.nn.Identity()
        assert _make_onnx_dummy(empty_probe, 8).device.type == "cpu"
        target_t = torch.from_numpy(pred[None].astype(np.int64))
        logits = torch.zeros((1, NUM_CLASSES, 64, 64), dtype=torch.float32, requires_grad=True)
        weights_t = torch.tensor(weights, dtype=torch.float32)
        loss = combined_loss(logits, target_t, weights_t)
        assert torch.isfinite(loss), loss
        loss.backward()
        assert torch.isfinite(logits.grad).all(), "gradiente não finito com pixels 255"
        assert torch.count_nonzero(logits.grad[:, :, 54:60, 54:60]) == 0, "pixel 255 deve ter gradiente zero"
        all_ignored = torch.full_like(target_t, IGNORE_INDEX)
        ignored_logits = torch.zeros_like(logits.detach(), requires_grad=True)
        ignored_loss = combined_loss(ignored_logits, all_ignored, weights_t)
        assert torch.isfinite(ignored_loss) and ignored_loss.item() == 0.0, ignored_loss
        ignored_loss.backward()
        assert torch.count_nonzero(ignored_logits.grad) == 0, "lote ignorado deve gerar gradiente zero"

        predicted_ids = pred.copy()
        predicted_ids[54:60, 54:60] = 3  # falso ink apenas na região ignorada
        eval_logits = torch.full((1, NUM_CLASSES, 64, 64), -10.0)
        for class_id in range(NUM_CLASSES):
            eval_logits[0, class_id][predicted_ids == class_id] = 10.0

        class _Output:
            def __init__(self, values):
                self.logits = values

        class _FixedModel:
            def eval(self):
                return self

            def __call__(self, pixel_values):
                return _Output(eval_logits.to(pixel_values.device))

        images_t = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        eval_metrics = require_valid_evaluation(
            evaluate(_FixedModel(), [(images_t, target_t)], "cpu"), "smoke"
        )
        assert eval_metrics["ink"]["iou"] == 1.0, eval_metrics["ink"]
        assert eval_metrics["ink"]["support_px"] == 36, eval_metrics["ink"]
        assert eval_metrics["supervised_px"] == 64 * 64 - ignored, eval_metrics
        assert eval_metrics["ignored_px"] == ignored, eval_metrics
        try:
            evaluate(_FixedModel(), [(images_t, all_ignored)], "cpu")
        except ValueError as exc:
            assert "sem pixels supervisionados" in str(exc), exc
        else:
            raise AssertionError("avaliação totalmente ignorada deveria falhar claramente")
    # Contrato ONNX: nomes e shapes esperados, sem rodar modelo.
    contract = {"input_name": "image", "output_name": "logits", "input_shape": [1, 3, 512, 512], "output_shape": [1, 4, 512, 512]}
    assert contract["output_shape"][1] == NUM_CLASSES
    print(f"SMOKE OK: {len(pairs)} pares, pixels supervisionados {counts.tolist()}, ignorados {ignored}, pesos {[round(w, 2) for w in weights]}")
    print(f"SMOKE OK: ink IoU estrita {m['ink']['iou']:.3f}, recall com tolerância 2px {tol['radius_2px']['recall']:.3f}")
    print("SMOKE OK: pixels 255 excluídos de loss, métricas e contagens; lote totalmente ignorado produz loss/gradiente zero")
    print("SMOKE OK: amostra ONNX acompanha o dispositivo do modelo (CPU/meta; CUDA se disponível no runtime)")
    print(f"SMOKE OK: contrato ONNX {contract}")
    return 0


# ---------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description="Treino SegFormer-B0 do Draw (4 classes).")
    parser.add_argument("--config", default="training/config.json")
    parser.add_argument("--data-dir", default="dataset/synthetic")
    parser.add_argument(
        "--real-train-dir",
        default=None,
        help="Diretório opcional com images/ e masks/ para adicionar ao treino sintético; val/test continuam em --data-dir.",
    )
    parser.add_argument("--out-dir", default="out")
    parser.add_argument(
        "--dataset-provenance-json",
        default=None,
        help="JSON com slugs/versões/privacidade dos inputs Kaggle usados.",
    )
    parser.add_argument("--baseline-metrics-json", default=None, help="Métricas congeladas do baseline, somente para comparação na validação.")
    parser.add_argument("--require-cuda", action="store_true", help="Falha antes do treino se CUDA/GPU estiver ausente.")
    parser.add_argument("--smoke", action="store_true", help="Smoke test sem GPU, sem treino.")
    parser.add_argument("--smoke-dir", default="/tmp/draw-smoke")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.smoke:
        return smoke(args.smoke_dir)

    import torch
    from torch.utils.data import DataLoader

    tr = cfg["training"]
    epochs = args.epochs or tr["epochs"]
    batch_size = args.batch_size or tr["batch_size"]
    torch.manual_seed(tr["seed"])
    random.seed(tr["seed"])
    np.random.seed(tr["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tr["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("GPU CUDA requerida para este kernel; treino não iniciado")
    if device == "cpu":
        print("AVISO: sem GPU; o treino completo em CPU é inviável. Use o Kaggle com GPU.", file=sys.stderr)

    config_path = Path(args.config)
    data_root = Path(args.data_dir)
    manifest_path = data_root / "split_manifest.json"
    input_fingerprint = hash_input_tree(data_root)

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    real_pairs = []
    if args.real_train_dir:
        real_root = Path(args.real_train_dir)
        if (real_root / "train" / "images").is_dir():
            real_root = real_root / "train"
        real_pairs = pair_images_masks(real_root)
    synthetic_train_pairs = pair_images_masks(data_root / "train")
    synthetic_val_pairs = pair_images_masks(data_root / "val")
    synthetic_test_pairs = pair_images_masks(data_root / "test")
    split_counts = {
        "train": len(synthetic_train_pairs),
        "val": len(synthetic_val_pairs),
        "test": len(synthetic_test_pairs),
    }
    source_manifest = validate_source_splits(manifest_path, split_counts)
    split_pixel_counts = {
        split: pixel_counts(pairs).tolist()
        for split, pairs in (("train", synthetic_train_pairs), ("val", synthetic_val_pairs), ("test", synthetic_test_pairs))
    }
    for split, counts_for_split in split_pixel_counts.items():
        if any(count <= 0 for count in counts_for_split):
            raise ValueError(f"split {split} não tem pixel válido para cada uma das quatro classes: {counts_for_split}")
    train_ds = build_dataset(
        args.data_dir, "train", cfg, augment=True, extra_pairs=real_pairs
    )
    val_ds = build_dataset(args.data_dir, "val", cfg, augment=False)
    test_ds = build_dataset(args.data_dir, "test", cfg, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=tr["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=tr["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=tr["num_workers"])
    train_pairs = synthetic_train_pairs + real_pairs
    counts = pixel_counts(train_pairs)
    print(
        f"pares de treino: {len(synthetic_train_pairs)} sintéticos + "
        f"{len(real_pairs)} reais = {len(train_pairs)}"
    )
    print(f"pixels de treino por classe: {counts.tolist()}")
    print(f"pesos da config: {cfg['loss']['cross_entropy_class_weights']}")
    print(f"pesos calculados (referência): {[round(w, 2) for w in compute_class_weights(counts)]}")

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    total_steps = epochs * max(len(train_loader), 1)
    warmup = int(total_steps * tr["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup, 1))
    class_weights = torch.tensor(cfg["loss"]["cross_entropy_class_weights"], dtype=torch.float32, device=device)

    best_iou, bad_epochs = -1.0, 0
    history = []
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, class_weights, device)
        metrics = require_valid_evaluation(evaluate(model, val_loader, device), "val")
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": loss, "val": metrics})
        print(f"época {epoch}: loss {loss:.4f}, val_mean_iou {metrics['mean_iou']:.4f}")
        torch.save(model.state_dict(), out_dir / "checkpoints" / "last.pt")
        if metrics["mean_iou"] > best_iou:
            best_iou = metrics["mean_iou"]
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= tr["early_stopping"]["patience_epochs"]:
                print(f"early stopping na época {epoch} (val_mean_iou sem melhora há {bad_epochs} épocas).")
                break

    # Exportação ONNX a partir do melhor checkpoint.
    # Não-fatal: se a exportação falhar, as métricas e os demais artefatos
    # ainda são escritos, e a falha fica registrada em onnx_validation —
    # antes, uma falha aqui perdia o metrics.json inteiro após o treino.
    model.load_state_dict(torch.load(out_dir / "checkpoints" / "best.pt", map_location=device))
    onnx_path = out_dir / cfg["export"]["onnx"]
    try:
        export_onnx(model, str(onnx_path), cfg)
        onnx_check = validate_onnx(
            model,
            onnx_path,
            val_loader,
            device,
            max_diff_tolerance=cfg["export"].get("max_abs_diff_tolerance", 1e-3),
            min_argmax_iou_tolerance=cfg["export"].get("min_argmax_iou", 0.99),
        )
    except Exception as exc:  # noqa: BLE001 - registrar, não perder o treino
        print(f"AVISO: exportação ONNX falhou ({exc}); métricas serão escritas sem ONNX.", file=sys.stderr)
        onnx_check = {"status": "failed", "error": str(exc)}
    final_metrics = require_valid_evaluation(evaluate(model, val_loader, device), "val")
    # O teste só é aberto uma vez, após seleção do checkpoint por val_mean_iou.
    final_metrics["test"] = require_valid_evaluation(evaluate(model, test_loader, device), "test")
    final_metrics["onnx_validation"] = onnx_check
    final_metrics["history"] = history
    final_metrics["train_pixel_counts"] = counts.tolist()
    final_metrics["split_pixel_counts"] = split_pixel_counts
    if args.baseline_metrics_json:
        baseline = json.loads(Path(args.baseline_metrics_json).read_text(encoding="utf-8"))
        baseline_mean = baseline.get("mean_iou")
        baseline_ink = baseline.get("ink", {}).get("iou")
        candidate_mean = final_metrics.get("mean_iou")
        candidate_ink = final_metrics.get("ink", {}).get("iou")
        if None in (baseline_mean, baseline_ink, candidate_mean, candidate_ink):
            raise ValueError("baseline/candidato sem mean_iou ou ink IoU para comparação")
        delta_mean = candidate_mean - baseline_mean
        delta_ink = candidate_ink - baseline_ink
        final_metrics["baseline_comparison"] = {
            "scope": "synthetic validation only; test was not used for selection",
            "baseline_mean_iou": baseline_mean,
            "candidate_mean_iou": candidate_mean,
            "delta_mean_iou": delta_mean,
            "baseline_ink_iou": baseline_ink,
            "candidate_ink_iou": candidate_ink,
            "delta_ink_iou": delta_ink,
            "criteria": {"mean_iou_min": baseline_mean - 0.005, "ink_iou_min": baseline_ink - 0.01},
            "gate_pass": candidate_mean >= baseline_mean - 0.005 and candidate_ink >= baseline_ink - 0.01,
        }
    final_metrics["train_source_counts"] = {
        "synthetic": len(synthetic_train_pairs),
        "real": len(real_pairs),
    }
    cfg["data_sources"] = {
        "synthetic_root": str(data_root),
        "real_train_root": str(Path(args.real_train_dir)) if args.real_train_dir else None,
        "train_pair_counts": final_metrics["train_source_counts"],
        "validation_source": "synthetic --data-dir val only",
        "test_source": "synthetic --data-dir test, evaluated once after checkpoint selection",
    }
    if args.dataset_provenance_json:
        cfg["dataset_provenance"] = json.loads(Path(args.dataset_provenance_json).read_text(encoding="utf-8"))
    cfg["reproducibility"] = {
        "seed": tr["seed"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: (importlib.metadata.version(name) if _package_exists(name) else None)
            for name in ("torch", "transformers", "numpy", "opencv-python", "onnx", "onnxruntime")
        },
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "input_fingerprint": input_fingerprint,
        "split_manifest_sha256": sha256_file(manifest_path),
        "split_counts": split_counts,
    }

    (out_dir / "classes.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    # split_manifest: cópia validada, se o gerador a forneceu.
    manifest_src = manifest_path
    if manifest_src.exists():
        split_manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
    else:
        split_manifest = {
            "note": "split_manifest.json não fornecido pelo gerador; splits lidos das pastas train/val/test.",
            "splits": ["train", "val", "test"],
        }
    split_manifest["training_sources"] = cfg["data_sources"]
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    artifact_paths = [
        out_dir / "classes.json", out_dir / "metrics.json", out_dir / "training_config.json",
        out_dir / "split_manifest.json", out_dir / "checkpoints" / "best.pt",
        out_dir / "checkpoints" / "last.pt", onnx_path,
    ]
    artifact_hashes = {
        p.relative_to(out_dir).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in artifact_paths if p.is_file()
    }
    (out_dir / "artifact_hashes.json").write_text(json.dumps(artifact_hashes, indent=2), encoding="utf-8")
    print(f"ONNX: {onnx_path} | métricas: {out_dir / 'metrics.json'} | val_mean_iou: {final_metrics['mean_iou']:.4f}")
    if onnx_check.get("status") != "ok":
        print(
            f"ERRO: exportação não validada (status={onnx_check.get('status')}); "
            "artefatos foram salvos para diagnóstico, mas a execução não passa o gate.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
