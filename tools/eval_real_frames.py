"""Harness de avaliação nos frames reais: o portão dos modelos neurais.

Roda um detector sobre o conjunto fixo de frames reais do usuário
(frames-dataset do laboratório: 43 positivos, 6 negativos, 11 máscaras
anotadas) e sobre o frame_real.png, e reporta sempre no mesmo formato:
presença por classe, falsos positivos nos negativos, IoU de máscara nos
frames anotados e o veredito por frame.

O detector padrão é o clássico atual (detect_sheet_with_confidence).
Um modelo neural entra por --model com um ONNX: ele precisa entregar o
mesmo trio (cantos, veredito, máscara) pela interface Detector abaixo,
e a nota dele sai comparável à do clássico, corrida contra corrida.
Nenhum modelo entra no app sem passar por este relatório.

Uso (a partir da raiz do repositório):
  python tools/eval_real_frames.py
  python tools/eval_real_frames.py --model modelo.onnx   (futuro)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw.paper import detect_sheet_with_confidence  # noqa: E402

_LAB = Path.home() / "workspace" / "draw-lab"
_DEFAULT_DATASET = _LAB / "frames-dataset"
_DEFAULT_FRAME = _LAB / "frame_real.png"

# Quadrilátero de referência do frame_real.png, medido à mão na
# auditoria de textura e confirmado pela auditoria de codebase: é o
# gabarito contra o qual o erro de canto desse frame é calculado.
_REFERENCE_QUAD = np.array(
    [(369, 51), (1014, 12), (818, 593), (259, 606)], dtype=np.float32
)


class ClassicalDetector:
    """Adaptador do detector clássico para a interface do harness.

    A máscara prevista é o quadrilátero detectado preenchido: é a
    região que o pipeline clássico afirma ser papel.
    """

    name = "classico (detect_sheet_with_confidence)"

    def __call__(self, frame):
        corners, verdict = detect_sheet_with_confidence(frame)
        mask = None
        if corners is not None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
        return corners, verdict, mask


class OnnxDetector:
    """Detector neural por ONNX. Estruturado, ainda não implementado.

    O encaixe é este construtor: carregar a sessão do ONNX e, em
    __call__, devolver (cantos, veredito, máscara) no mesmo contrato
    do clássico. Cantos em float32[4,2] TL/TR/BR/BL nos pixels do
    frame; veredito em {"paper", "blank_maybe", None} a partir do
    canal de presença do modelo; máscara uint8 255=papel ou None.
    """

    name = "onnx"

    def __init__(self, model_path):
        raise NotImplementedError(
            f"detector ONNX ainda nao implementado (modelo: {model_path}). "
            "A interface esta pronta em OnnxDetector; falta carregar a "
            "sessao e mapear as saidas do modelo para o contrato."
        )

    def __call__(self, frame):
        raise NotImplementedError


def _mask_iou(predicted, truth):
    inter = int(((predicted > 0) & (truth > 0)).sum())
    union = int(((predicted > 0) | (truth > 0)).sum())
    return inter / union if union else 1.0


def _presence_label(expected, verdict):
    if expected == "positivo":
        if verdict == "paper":
            return "OK"
        if verdict == "blank_maybe":
            return "OK (baixa confianca)"
        return "FALHOU"
    if verdict == "paper":
        return "FALSO POSITIVO"
    if verdict == "blank_maybe":
        return "FALSO POSITIVO (fraco)"
    return "OK"


def evaluate_dataset(detector, dataset_dir):
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    entries = manifest["frames"]
    counts = {
        "positivo": {"paper": 0, "blank_maybe": 0, None: 0},
        "negativo": {"paper": 0, "blank_maybe": 0, None: 0},
    }
    ious = []
    lines = []
    for entry in entries:
        frame = cv2.imread(str(dataset_dir / "frames" / entry["file"]))
        _, verdict, mask = detector(frame)
        expected = entry["class"]
        counts[expected][verdict] += 1
        label = _presence_label(expected, verdict)
        line = (
            f"[{entry['index']:2d}] {entry['file']}  "
            f"esperado={expected:8s}  veredito={str(verdict):12s}  {label}"
        )
        if entry["has_mask"]:
            truth = cv2.imread(
                str(dataset_dir / "masks" / entry["file"]),
                cv2.IMREAD_GRAYSCALE,
            )
            iou = _mask_iou(mask, truth) if mask is not None else 0.0
            ious.append(iou)
            line += f"  IoU={iou:.3f}"
        lines.append(line)
    return counts, ious, lines


def evaluate_reference_frame(detector, frame_path):
    frame = cv2.imread(str(frame_path))
    corners, verdict, _ = detector(frame)
    lines = [f"veredito: {verdict}"]
    if corners is not None:
        errors = np.linalg.norm(corners - _REFERENCE_QUAD, axis=1)
        lines.append(f"cantos: {np.round(corners, 1).tolist()}")
        lines.append(
            f"erro de canto vs referencia: maximo {errors.max():.1f}px, "
            f"medio {errors.mean():.1f}px"
        )
    else:
        lines.append("cantos: nenhum (folha nao detectada)")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Avalia um detector nos frames reais do usuário"
    )
    parser.add_argument(
        "--dataset",
        default=str(_DEFAULT_DATASET),
        help="pasta do frames-dataset (padrao: o do laboratorio)",
    )
    parser.add_argument(
        "--frame",
        default=str(_DEFAULT_FRAME),
        help="frame real avulso com gabarito de cantos "
        "(padrao: frame_real.png do laboratorio)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="ONNX de um detector neural (futuro; hoje só estruturado)",
    )
    args = parser.parse_args()

    if args.model:
        try:
            detector = OnnxDetector(args.model)
        except NotImplementedError as error:
            print(f"ERRO: {error}")
            return 2
    else:
        detector = ClassicalDetector()

    dataset_dir = Path(args.dataset)
    if not (dataset_dir / "manifest.json").exists():
        print(f"ERRO: manifest.json nao encontrado em {dataset_dir}")
        return 2

    print("=" * 72)
    print(f"AVALIACAO EM FRAMES REAIS - detector: {detector.name}")
    print("=" * 72)

    counts, ious, lines = evaluate_dataset(detector, dataset_dir)
    print("\n-- frames-dataset, frame a frame --")
    for line in lines:
        print(line)

    pos = counts["positivo"]
    neg = counts["negativo"]
    print("\n-- resumo de presenca --")
    print(
        f"positivos ({sum(pos.values())}): paper={pos['paper']}, "
        f"blank_maybe={pos['blank_maybe']}, nenhum={pos[None]}"
    )
    print(
        f"negativos ({sum(neg.values())}): nenhum={neg[None]}, "
        f"blank_maybe={neg['blank_maybe']}, "
        f"paper={neg['paper']} (falsos positivos)"
    )
    if ious:
        print(
            f"mascaras anotadas ({len(ious)}): "
            f"IoU medio={np.mean(ious):.3f}, minimo={np.min(ious):.3f}"
        )

    frame_path = Path(args.frame)
    if frame_path.exists():
        print(f"\n-- {frame_path.name} (frame real com gabarito) --")
        for line in evaluate_reference_frame(detector, frame_path):
            print(line)

    print("\n" + "=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
