"""Métricas de segmentação semântica para o modelo neural do Draw.

Classes (ordem fixa, igual ao contrato de entrega do modelo):

    0 = background
    1 = paper
    2 = hand
    3 = ink

Este módulo é puro NumPy: não depende de PyTorch, ONNX Runtime nem de
webcam, para poder rodar no CI e ser reutilizado no notebook do Kaggle
sem instalar nada além do que o Draw já usa.

Regras de rótulo que as métricas pressupõem (definidas pelo projeto):

- A máscara representa o que a câmera realmente vê. Se a mão cobre o
  traço, o pixel visível é ``hand``, nunca ``ink``.
- A máscara ``ink_full`` (desenho completo antes da oclusão) NÃO é
  classe de treinamento e NÃO entra nestas métricas. Ela serve só para
  avaliar, no futuro, o sistema de memória do Draw.

Métricas obrigatórias (nunca só accuracy global):

- IoU por classe (background, paper, hand, ink);
- Dice por classe (paper, hand, ink; background também é calculado);
- Precision e Recall por classe;
- Accuracy global, reportada apenas como contexto — um modelo que
  acerta todo o fundo e erra toda a tinta pode ter accuracy alta e
  ser inútil, por isso ela nunca é o critério de aceite.

Pixels ground-truth com valor ``255`` são desconhecidos e excluídos de
todas as contagens, inclusive accuracy e tolerância de tinta. Predições
com ``255`` são inválidas. Uma classe sem suporte nos pixels válidos
recebe ``None`` (NA), não uma pontuação perfeita. ``mean_iou`` e
``mean_dice`` usam classes com união não vazia; assim, uma classe só
prevista incorretamente continua penalizada, enquanto uma classe ausente
dos dois lados não melhora a média. ``mean_iou_present`` é a média das
classes presentes no ground truth, incluindo falsos negativos.

Métrica específica de ink com tolerância espacial
--------------------------------------------------
Traços finos são poucos pixels. Um traço previsto 1 ou 2 pixels ao
lado do traço real está visualmente correto, mas a IoU exata cai
muito, porque a interseção de linhas finas deslocadas é quase zero.
Por isso, além da IoU exata, medimos Precision e Recall de ink com
tolerância espacial (padrão: 3 px, dentro da faixa 2-3 px do projeto):

- ``tolerance_precision``: fração dos pixels de ink previstos que
  estão a no máximo ``tolerance_px`` de algum pixel de ink real.
  Implementação: dilata-se a máscara de ink real por um disco de
  raio ``tolerance_px`` e conta-se quantos pixels previstos caem
  dentro dela.
- ``tolerance_recall``: fração dos pixels de ink reais que estão a
  no máximo ``tolerance_px`` de algum pixel de ink previsto.
  Implementação simétrica: dilata-se a máscara prevista e conta-se
  quantos pixels reais caem dentro dela.

A dilatação usa um disco euclidiano (deslocamentos com
``dx**2 + dy**2 <= tolerance_px**2``), implementada em NumPy puro,
sem depender de SciPy. Ela mede proximidade espacial direta, não
forma: duas linhas paralelas a 2 px contam como acerto com
tolerância, mesmo sem sobreposição exata. Alternativas consideradas
e não usadas como métrica principal: comparação de esqueletos
(sensível a pequenas ramificações do thinning) e distance transform
(equivalente à dilatação aqui, mas exigiria SciPy ou OpenCV no
núcleo das métricas). A IoU exata continua sendo reportada ao lado,
para que a tolerância nunca esconda um deslocamento sistemático:
um modelo bom tem IoU de ink razoável E tolerância alta; um modelo
só com tolerância alta e IoU próxima de zero está deslocando tudo.

Bateria fixa de validação
--------------------------
``SCENARIOS`` tem exatamente os 15 cenários fixos do projeto. A
função ``save_predictions`` grava, por cenário, a predição
visualizada em PNG e um manifesto JSON, para que predições de
versões diferentes do modelo sejam comparáveis caso a caso, sem
depender de re-rodar o modelo antigo.
"""

import json
from pathlib import Path

import numpy as np

# Ordem fixa das classes. Não reordenar: ela é o contrato do ONNX.
CLASSES = ("background", "paper", "hand", "ink")
BACKGROUND, PAPER, HAND, INK = 0, 1, 2, 3
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255

# Paleta de visualização (BGR, para cv2.imwrite): fundo preto, folha
# cinza-clara, mão avermelhada, tinta azul. Cores distintas por
# classe, para a PNG ser legível sem legenda.
PALETTE_BGR = {
    BACKGROUND: (0, 0, 0),
    PAPER: (200, 200, 200),
    HAND: (80, 80, 220),
    INK: (220, 120, 40),
}

# Bateria fixa de 15 cenários de validação, na ordem do projeto.
# "id" é estável e vira nome de arquivo; "name" é legível.
SCENARIOS = (
    {"id": "01_clean_sheet", "name": "folha limpa"},
    {"id": "02_sheet_with_drawing", "name": "folha com desenho"},
    {"id": "03_heavily_drawn_sheet", "name": "folha muito desenhada"},
    {"id": "04_hand_covering_corner", "name": "mão cobrindo canto"},
    {"id": "05_hand_covering_center", "name": "mão cobrindo centro"},
    {"id": "06_hand_covering_stroke", "name": "mão cobrindo traço"},
    {"id": "07_hand_partly_outside", "name": "mão parcialmente fora da imagem"},
    {"id": "08_light_background", "name": "fundo claro"},
    {"id": "09_dark_background", "name": "fundo escuro"},
    {"id": "10_strong_shadow", "name": "sombra forte"},
    {"id": "11_sheet_partly_outside", "name": "folha parcialmente fora da imagem"},
    {"id": "12_hand_without_sheet", "name": "mão sem folha"},
    {"id": "13_white_object_no_sheet", "name": "objeto branco sem folha"},
    {"id": "14_thin_pencil_stroke", "name": "traço fino de lápis"},
    {"id": "15_thick_pen_stroke", "name": "traço grosso de caneta"},
)
SCENARIO_IDS = tuple(s["id"] for s in SCENARIOS)


def _validate_masks(pred, gt):
    """Valida máscaras e devolve também a região supervisionada.

    ``255`` é aceito apenas no ground truth e representa pixel ignorado.
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred shape {pred.shape} != gt shape {gt.shape}; "
            "predição e ground truth precisam ter o mesmo formato"
        )
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("máscaras devem ser 2D (H, W) com valores de classe")
    if pred.size == 0:
        raise ValueError("máscaras não podem ser vazias")
    if pred.min() < 0 or pred.max() >= NUM_CLASSES:
        raise ValueError(
            f"pred contém classe fora de 0..{NUM_CLASSES - 1}; "
            f"classes válidas: {list(CLASSES)}"
        )
    invalid_gt = (gt < 0) | ((gt >= NUM_CLASSES) & (gt != 255))
    if np.any(invalid_gt):
        invalid = np.unique(gt[invalid_gt]).tolist()
        raise ValueError(
            f"gt contém valores inválidos {invalid}; classes válidas: "
            f"0..{NUM_CLASSES - 1} e 255 (ignorado)"
        )
    return pred, gt, gt != 255


def _tp_fp_fn(pred, gt, valid):
    """Conta TP/FP/FN por classe, ignorando pixels fora de ``valid``."""
    tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fn = np.zeros(NUM_CLASSES, dtype=np.int64)
    for cls in range(NUM_CLASSES):
        p = (pred == cls) & valid
        g = gt == cls
        tp[cls] = int(np.count_nonzero(p & g))
        fp[cls] = int(np.count_nonzero(p & ~g & valid))
        fn[cls] = int(np.count_nonzero(~p & g))
    return tp, fp, fn


def _safe_div(num, den):
    """Divisão; denominador zero significa métrica indefinida (NA)."""
    return None if den == 0 else float(num) / float(den)


def _per_class_counts(pred, gt, valid):
    tp, fp, fn = _tp_fp_fn(pred, gt, valid)
    return {
        CLASSES[cls]: {
            "tp": int(tp[cls]),
            "fp": int(fp[cls]),
            "fn": int(fn[cls]),
            "gt_pixels": int(tp[cls] + fn[cls]),
            "pred_pixels": int(tp[cls] + fp[cls]),
            "union_pixels": int(tp[cls] + fp[cls] + fn[cls]),
        }
        for cls in range(NUM_CLASSES)
    }


def iou_per_class(pred, gt):
    """IoU por classe; ``None`` indica que a classe não tem suporte válido."""
    pred, gt, valid = _validate_masks(pred, gt)
    tp, fp, fn = _tp_fp_fn(pred, gt, valid)
    return {
        CLASSES[cls]: _safe_div(tp[cls], tp[cls] + fp[cls] + fn[cls])
        for cls in range(NUM_CLASSES)
    }


def dice_per_class(pred, gt):
    """Dice (F1) por classe; ``None`` indica classe sem suporte válido."""
    pred, gt, valid = _validate_masks(pred, gt)
    tp, fp, fn = _tp_fp_fn(pred, gt, valid)
    return {
        CLASSES[cls]: _safe_div(2 * tp[cls], 2 * tp[cls] + fp[cls] + fn[cls])
        for cls in range(NUM_CLASSES)
    }


def precision_recall_per_class(pred, gt):
    """Precision/recall por classe; métricas sem suporte retornam ``None``."""
    pred, gt, valid = _validate_masks(pred, gt)
    tp, fp, fn = _tp_fp_fn(pred, gt, valid)
    precision, recall = {}, {}
    for cls in range(NUM_CLASSES):
        predicted = int(tp[cls] + fp[cls])
        actual = int(tp[cls] + fn[cls])
        # Uma classe presente no GT mas nunca prevista tem precision 0,
        # seguindo a convenção histórica do projeto; se ambos os lados
        # não têm suporte, a métrica é NA.
        precision[CLASSES[cls]] = (
            float(tp[cls]) / predicted if predicted else (0.0 if actual else None)
        )
        recall[CLASSES[cls]] = float(tp[cls]) / actual if actual else None
    return {"precision": precision, "recall": recall}


def evaluate(pred, gt):
    """Avaliação completa; pixels ``255`` no GT são excluídos.

    Retorna métricas por classe, contagens TP/FP/FN, suporte, matriz de
    confusão e agregados. As médias usam classes com união não vazia;
    classes sem suporte têm métricas ``None``.
    """
    pred, gt, valid = _validate_masks(pred, gt)
    counts = _per_class_counts(pred, gt, valid)
    iou = iou_per_class(pred, gt)
    dice = dice_per_class(pred, gt)
    pr = precision_recall_per_class(pred, gt)
    supported = [name for name in CLASSES if counts[name]["union_pixels"] > 0]
    present = [name for name in CLASSES if counts[name]["gt_pixels"] > 0]
    mean_iou = float(np.mean([iou[c] for c in supported])) if supported else None
    mean_dice = float(np.mean([dice[c] for c in supported])) if supported else None
    mean_iou_present = (
        float(np.mean([iou[c] for c in present])) if present else None
    )
    valid_pixels = int(np.count_nonzero(valid))
    accuracy = (
        float(np.count_nonzero((pred == gt) & valid) / valid_pixels)
        if valid_pixels else None
    )
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    if valid_pixels:
        np.add.at(confusion, (gt[valid], pred[valid]), 1)
    return {
        "iou": iou,
        "dice": dice,
        "precision": pr["precision"],
        "recall": pr["recall"],
        "counts": counts,
        "confusion_matrix": confusion.tolist(),
        "valid_pixels": valid_pixels,
        "ignored_pixels": int(gt.size - valid_pixels),
        # Preserve keys consumed by the existing training-zone smoke checks.
        "valid_px": valid_pixels,
        "ignored_px": int(gt.size - valid_pixels),
        "class_support_px": {
            name: counts[name]["gt_pixels"] for name in CLASSES
        },
        "supported_classes": supported,
        "mean_iou": mean_iou,
        "mean_iou_present": mean_iou_present,
        "mean_dice": mean_dice,
        "accuracy": accuracy,
    }


def _dilate_disk(mask, radius):
    """Dilata uma máscara binária por um disco euclidiano de raio dado.

    Implementação NumPy pura (sem SciPy): para cada deslocamento
    (dx, dy) com dx**2 + dy**2 <= radius**2, desloca a máscara e
    acumula com OR. Raio 0 devolve a máscara original.
    """
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    dilated = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            shifted = np.zeros_like(mask)
            y0, y1 = max(0, -dy), min(mask.shape[0], mask.shape[0] - dy)
            x0, x1 = max(0, -dx), min(mask.shape[1], mask.shape[1] - dx)
            if y0 >= y1 or x0 >= x1:
                continue
            y_src, y_dst = slice(y0, y1), slice(y0 + dy, y1 + dy)
            x_src, x_dst = slice(x0, x1), slice(x0 + dx, x1 + dx)
            shifted[y_dst, x_dst] = mask[y_src, x_src]
            dilated |= shifted
    return dilated


def ink_tolerance_metrics(pred, gt, tolerance_px=3):
    """Precision/Recall de tinta com tolerância, ignorando GT=255.

    Métricas sem denominador ou sem qualquer pixel válido retornam
    ``None``; quando só um lado tem tinta, F1 vale 0.
    """
    pred, gt, valid = _validate_masks(pred, gt)
    pred_ink = (pred == INK) & valid
    gt_ink = gt == INK
    n_pred = int(np.count_nonzero(pred_ink))
    n_gt = int(np.count_nonzero(gt_ink))
    if not np.any(valid):
        return {
            "tolerance_px": tolerance_px,
            "precision": None,
            "recall": None,
            "f1": None,
            "pred_ink": 0,
            "gt_ink": 0,
            "matched": 0,
        }
    if n_pred == 0 and n_gt == 0:
        return {
            "tolerance_px": tolerance_px,
            "precision": None,
            "recall": None,
            "f1": None,
            "pred_ink": 0,
            "gt_ink": 0,
            "matched": 0,
        }
    matched_pred = int(np.count_nonzero(pred_ink & _dilate_disk(gt_ink, tolerance_px)))
    matched_gt = int(np.count_nonzero(gt_ink & _dilate_disk(pred_ink, tolerance_px)))
    precision = matched_pred / n_pred if n_pred else (0.0 if n_gt else None)
    recall = matched_gt / n_gt if n_gt else None
    if n_pred == 0 or n_gt == 0:
        f1 = 0.0
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tolerance_px": tolerance_px,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_ink": n_pred,
        "gt_ink": n_gt,
        "matched": matched_pred,
    }


def visualize_mask(mask):
    """Converte uma máscara de classes em imagem BGR (H, W, 3) colorida."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("máscara para visualizar deve ser 2D (H, W)")
    if mask.min() < 0 or mask.max() >= NUM_CLASSES:
        raise ValueError(f"classe fora de 0..{NUM_CLASSES - 1}")
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        out[mask == cls] = PALETTE_BGR[cls]
    return out


def evaluate_scenarios(predictions, ground_truths):
    """Avalia predições da bateria fixa, por cenário e agregada.

    ``predictions`` e ``ground_truths`` são dicts {scenario_id: máscara}.
    Cenários sem ground truth são ignorados no agregado e reportados
    com ``metrics: None``. Retorna {"per_scenario": {...}, "aggregate":
    {"mean_iou": ..., "mean_ink_tolerance_f1": ...}}.
    """
    per_scenario = {}
    ious, ink_f1s = [], []
    for scenario in SCENARIOS:
        sid = scenario["id"]
        if sid not in predictions:
            continue
        entry = {"name": scenario["name"]}
        if sid in ground_truths:
            metrics = evaluate(predictions[sid], ground_truths[sid])
            tol = ink_tolerance_metrics(predictions[sid], ground_truths[sid])
            entry["metrics"] = metrics
            entry["ink_tolerance"] = tol
            if metrics["mean_iou"] is not None:
                ious.append(metrics["mean_iou"])
            if tol["f1"] is not None:
                ink_f1s.append(tol["f1"])
        else:
            entry["metrics"] = None
            entry["ink_tolerance"] = None
        per_scenario[sid] = entry
    return {
        "per_scenario": per_scenario,
        "aggregate": {
            "mean_iou": float(np.mean(ious)) if ious else None,
            "mean_ink_tolerance_f1": float(np.mean(ink_f1s)) if ink_f1s else None,
            "n_evaluated": len(ious),
            "n_ink_evaluated": len(ink_f1s),
        },
    }


def save_predictions(output_dir, predictions, model_name=""):
    """Salva predições da bateria fixa para comparação entre versões.

    Para cada cenário presente em ``predictions``, grava
    ``<scenario_id>_pred.png`` (máscara visualizada com a paleta do
    módulo) e um manifesto ``predictions.json`` com modelo, cenários,
    formatos e, se o manifesto for gerado junto com ground truths, as
    métricas — neste módulo, o manifesto registra o que foi salvo;
    as métricas vivem em ``evaluate_scenarios`` e devem ser salvas
    pelo chamador junto ao manifesto quando houver ground truth.

    Retorna o caminho do manifesto. Exige OpenCV apenas para gravar
    PNG (dependência que o Draw já tem); sem cv2, levanta
    RuntimeError com mensagem clara, em vez de gravar arquivo mudo.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - ambiente sem cv2
        raise RuntimeError(
            "save_predictions precisa de OpenCV para gravar PNG; "
            "instale opencv-contrib-python (já é dependência do Draw)"
        ) from exc
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for scenario in SCENARIOS:
        sid = scenario["id"]
        if sid not in predictions:
            continue
        mask = np.asarray(predictions[sid])
        if mask.ndim != 2:
            raise ValueError(f"predição do cenário {sid} deve ser 2D (H, W)")
        filename = f"{sid}_pred.png"
        ok = cv2.imwrite(str(out_dir / filename), visualize_mask(mask))
        if not ok:
            raise RuntimeError(f"falha ao gravar {out_dir / filename}")
        files[sid] = {"name": scenario["name"], "file": filename, "shape": list(mask.shape)}
    manifest = {"model_name": model_name, "scenarios": files}
    manifest_path = out_dir / "predictions.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path
