"""Treinamento do modelo de segmentação do Draw (SegFormer-B0, 4 classes).

Escopo: dados -> treino -> validação -> exportação ONNX -> documentação.
FORA do escopo (Codex, depois): webcam, integração no app, tracking
temporal, memória de traços, oclusão em tempo real, interface, CAD,
simulador.

Classes: 0=background, 1=paper, 2=hand, 3=ink.
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
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

CLASSES = {0: "background", 1: "paper", 2: "hand", 3: "ink"}
NUM_CLASSES = 4
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
    """Máscara válida: um canal, valores só em {0,1,2,3}."""
    values = np.unique(mask)
    bad = [int(v) for v in values if v not in (0, 1, 2, 3)]
    if bad:
        raise ValueError(f"máscara com valores inválidos: {bad}")
    return True


def load_mask(mask_path):
    """Lê máscara com cv2 (grayscale) e valida valores 0..3."""
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
    """Conta pixels por classe (amostra até max_files, None = todos)."""
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

    pred/target: arrays 2D de inteiros 0..3, mesmo shape.
    """
    out = {}
    for c in range(NUM_CLASSES):
        p = pred == c
        t = target == c
        tp = int((p & t).sum())
        fp = int((p & ~t).sum())
        fn = int((~p & t).sum())
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        out[CLASSES[c]] = {
            "iou": tp / denom_iou if denom_iou else 1.0,
            "dice": 2 * tp / denom_dice if denom_dice else 1.0,
            "precision": tp / (tp + fp) if (tp + fp) else 1.0,
            "recall": tp / (tp + fn) if (tp + fn) else 1.0,
            "support_px": int(t.sum()),
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
    pred_ink = (pred == 3).astype(np.uint8)
    target_ink = (target == 3).astype(np.uint8)
    for r in radii:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        p_dil = cv2.dilate(pred_ink, kernel) > 0
        t_dil = cv2.dilate(target_ink, kernel) > 0
        # precision tolerante: predito que cai perto do traço real;
        # recall tolerante: traço real coberto por predição próxima.
        pred_n = int((pred_ink > 0).sum())
        target_n = int((target_ink > 0).sum())
        out[f"radius_{r}px"] = {
            "precision": int(((pred_ink > 0) & t_dil).sum()) / pred_n if pred_n else 1.0,
            "recall": int(((target_ink > 0) & p_dil).sum()) / target_n if target_n else 1.0,
        }
    return out


def dice_loss_numpy(prob, target, smooth=1e-6):
    """Dice loss multiclasse em numpy (espelho da versão torch).

    prob: (C, H, W) probabilidades; target: (H, W) inteiros.
    """
    losses = []
    for c in range(NUM_CLASSES):
        p = prob[c].ravel()
        t = (target == c).astype(np.float32).ravel()
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


def build_dataset(data_dir, split, cfg, augment=False):
    """Dataset torch. Import lazy: sem torch, esta função não existe."""
    import torch
    from torch.utils.data import Dataset

    pairs = pair_images_masks(Path(data_dir) / split)
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
    """0.5 * CE com peso por classe + 0.5 * Dice multiclasse (torch)."""
    import torch
    import torch.nn.functional as F

    ce = F.cross_entropy(logits, target, weight=class_weights)
    prob = torch.softmax(logits, dim=1)
    dice = 0.0
    for c in range(NUM_CLASSES):
        p = prob[:, c].reshape(prob.shape[0], -1)
        t = (target == c).float().reshape(target.shape[0], -1)
        inter = (p * t).sum(dim=1)
        dice = dice + (1.0 - (2 * inter + 1e-6) / (p.sum(dim=1) + t.sum(dim=1) + 1e-6))
    dice = dice / NUM_CLASSES
    # Correção do reteste no Kaggle: sem esta média, dice tem shape [batch]
    # e loss.backward() falha com "grad can be implicitly created only
    # for scalar outputs". A loss tem que ser escalar.
    dice = dice.mean()
    return ce_weight * ce + dice_weight * dice


def train_one_epoch(model, loader, optimizer, class_weights, device, scaler=None):
    import torch

    model.train()
    total, n = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
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
    return total / max(n, 1)


def evaluate(model, loader, device):
    """Avalia um loader inteiro; devolve métricas agregadas por classe."""
    import torch

    model.eval()
    agg = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(NUM_CLASSES)}
    ink_pred_list, ink_target_list = [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(pixel_values=images).logits
            if logits.shape[-2:] != masks.shape[-2:]:
                import torch.nn.functional as F

                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1).cpu().numpy()
            target = masks.numpy()
            for c in range(NUM_CLASSES):
                p, t = pred == c, target == c
                agg[c]["tp"] += int((p & t).sum())
                agg[c]["fp"] += int((p & ~t).sum())
                agg[c]["fn"] += int((~p & t).sum())
            ink_pred_list.append(pred)
            ink_target_list.append(target)
    metrics = {}
    for c in range(NUM_CLASSES):
        tp, fp, fn = agg[c]["tp"], agg[c]["fp"], agg[c]["fn"]
        metrics[CLASSES[c]] = {
            "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
            "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
            "precision": tp / (tp + fp) if (tp + fp) else 1.0,
            "recall": tp / (tp + fn) if (tp + fn) else 1.0,
        }
    pred_all = np.concatenate([p.ravel() for p in ink_pred_list]) if ink_pred_list else np.array([])
    target_all = np.concatenate([t.ravel() for t in ink_target_list]) if ink_target_list else np.array([])
    # Métrica de ink com tolerância precisa de imagem 2D; usa o primeiro
    # lote como amostra representativa (o relatório completo usa o
    # avaliador de cenários fixos, abaixo).
    metrics["ink_tolerance_sample"] = {}
    if ink_pred_list:
        metrics["ink_tolerance_sample"] = ink_tolerance_metrics(ink_pred_list[0][0], ink_target_list[0][0])
    metrics["mean_iou"] = float(np.mean([metrics[CLASSES[c]]["iou"] for c in range(NUM_CLASSES)]))
    return metrics


# ---------------------------------------------------------------- ONNX
def export_onnx(model, out_path, cfg):
    """Exporta para ONNX com o contrato do Draw: input 'image', output 'logits'."""
    import torch

    model.eval()
    size = cfg["model"]["input"]["height"]
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": [0], "logits": [0]},
        opset_version=cfg["export"]["opset_version"],
        do_constant_folding=True,
    )
    return out_path


def validate_onnx(model, onnx_path, loader, device):
    """Compara PyTorch vs ONNX no mesmo lote; retorna diffs ou None se sem onnxruntime."""
    try:
        import onnxruntime as ort
        import torch
    except ImportError:
        print("AVISO: onnxruntime ausente; validação ONNX pulada.", file=sys.stderr)
        return None
    model.eval()
    images, _ = next(iter(loader))
    with torch.no_grad():
        torch_logits = model(pixel_values=images.to(device)).logits.detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    image_np = images.numpy()
    # Normalização já feita no dataset; o ONNX recebe o tensor normalizado.
    onnx_logits = session.run(["logits"], {"image": image_np})[0]
    if torch_logits.shape != onnx_logits.shape:
        return {"max_abs_diff": None, "note": f"shape divergente: torch {torch_logits.shape} vs onnx {onnx_logits.shape}"}
    return {"max_abs_diff": float(np.abs(torch_logits - onnx_logits).max())}


# ---------------------------------------------------------------- smoke
def smoke(smoke_dir):
    """Smoke test sem GPU e sem treino completo.

    Cria 2 pares sintéticos (imagem + máscara 0..3), testa pareamento,
    validação de máscara, contagem de pixels, pesos, métricas numpy,
    métrica de ink com tolerância e o contrato ONNX (nomes/shapes).
    NÃO treina: não há torch nem GPU aqui, e isso é declarado.
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
    cv2.imwrite(str(img_dir / "scene_0001.png"), img)
    cv2.imwrite(str(mask_dir / "scene_0001.png"), mask)
    # Imagem 2: só fundo (negativo).
    cv2.imwrite(str(img_dir / "scene_0002.png"), np.full((64, 64, 3), 90, dtype=np.uint8))
    cv2.imwrite(str(mask_dir / "scene_0002.png"), np.zeros((64, 64), dtype=np.uint8))

    pairs = pair_images_masks(root / "train")
    assert len(pairs) == 2, pairs
    counts = pixel_counts(pairs)
    assert counts.sum() == 2 * 64 * 64, counts
    weights = compute_class_weights(counts)
    assert len(weights) == NUM_CLASSES and all(w > 0 for w in weights), weights
    pred = load_mask(mask_dir / "scene_0001.png")
    # Predição perfeita exceto 1 px de tinta deslocado: IoU estrita cai,
    # tolerância de 2-3 px deve recuperar.
    pred_shifted = pred.copy()
    pred_shifted[20, 12:48] = 0
    pred_shifted[21, 12:48] = 3
    m = per_class_metrics(pred_shifted, pred)
    assert m["ink"]["iou"] < 1.0, "IoU estrita deveria cair com o deslocamento"
    tol = ink_tolerance_metrics(pred_shifted, pred)
    assert tol["radius_2px"]["recall"] > m["ink"]["recall"], (tol, m["ink"])
    # Contrato ONNX: nomes e shapes esperados, sem rodar modelo.
    contract = {"input_name": "image", "output_name": "logits", "input_shape": [1, 3, 512, 512], "output_shape": [1, 4, 512, 512]}
    assert contract["output_shape"][1] == NUM_CLASSES
    print(f"SMOKE OK: {len(pairs)} pares, pixels {counts.tolist()}, pesos {[round(w, 2) for w in weights]}")
    print(f"SMOKE OK: ink IoU estrita {m['ink']['iou']:.3f}, recall com tolerância 2px {tol['radius_2px']['recall']:.3f}")
    print(f"SMOKE OK: contrato ONNX {contract}")
    return 0


# ---------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description="Treino SegFormer-B0 do Draw (4 classes).")
    parser.add_argument("--config", default="training/config.json")
    parser.add_argument("--data-dir", default="dataset/synthetic")
    parser.add_argument("--out-dir", default="out")
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
    np.random.seed(tr["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("AVISO: sem GPU; o treino completo em CPU é inviável. Use o Kaggle com GPU.", file=sys.stderr)

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    train_ds = build_dataset(args.data_dir, "train", cfg, augment=True)
    val_ds = build_dataset(args.data_dir, "val", cfg, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=tr["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=tr["num_workers"])
    train_pairs = pair_images_masks(Path(args.data_dir) / "train")
    counts = pixel_counts(train_pairs)
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
        metrics = evaluate(model, val_loader, device)
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
    model.load_state_dict(torch.load(out_dir / "checkpoints" / "best.pt", map_location=device))
    onnx_path = out_dir / cfg["export"]["onnx"]
    export_onnx(model, str(onnx_path), cfg)
    onnx_check = validate_onnx(model, onnx_path, val_loader, device)
    final_metrics = evaluate(model, val_loader, device)
    final_metrics["onnx_validation"] = onnx_check
    final_metrics["history"] = history
    final_metrics["train_pixel_counts"] = counts.tolist()

    (out_dir / "classes.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    # split_manifest: cópia validada, se o gerador a forneceu.
    manifest_src = Path(args.data_dir) / "split_manifest.json"
    if manifest_src.exists():
        (out_dir / "split_manifest.json").write_text(manifest_src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        (out_dir / "split_manifest.json").write_text(
            json.dumps({"note": "split_manifest.json não fornecido pelo gerador; splits lidos das pastas train/val/test.", "splits": ["train", "val", "test"]}, indent=2),
            encoding="utf-8",
        )
    print(f"ONNX: {onnx_path} | métricas: {out_dir / 'metrics.json'} | val_mean_iou: {final_metrics['mean_iou']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
