"""Segundo estágio de tinta (ink-v2), simples e direto.

Recebe a máscara de 4 classes do SegFormer (0=background, 1=paper,
2=hand, 3=ink) e refina SÓ a classe ink com o especialista UNetLite
do kernel guzzbr/draw-ink-train-v2 (3 classes, ink = classe 2).

Regras, de propósito curtas:
- Desligado por padrão: sem flag, devolve a máscara igual.
- Nunca toca em hand (2) nem em background (0).
- A tinta refinada só pode aparecer onde o SegFormer já via paper (1)
  ou ink (3): o especialista não inventa tinta fora da folha.
- Onde o SegFormer via ink e o especialista não vê tinta, o pixel
  volta a ser paper (1), não background.
- Não muda o contrato do SegFormer nem o draw/neural.py: isto é um
  pós-processamento opcional sobre a máscara pronta.

Checkpoint: models/draw_ink_v2.onnx, baixado do kernel
guzzbr/draw-ink-train-v2 e conferido por sha256 (ver
docs/ink-v2.md). O caminho pode ser trocado pela variável de
ambiente DRAW_INK_V2_MODEL; DRAW_INK_V2_ENABLED=1 liga o estágio
via from_env().
"""

import os

import numpy as np

BACKGROUND = 0
PAPER = 1
HAND = 2
INK = 3

# UNetLite do ink-v2: 3 classes, tinta é a classe 2.
_INK_CLASS = 2

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "draw_ink_v2.onnx",
)


def _resize_nearest(mask, shape):
    """Redimensiona máscara booleana para shape, vizinho mais próximo."""
    if mask.shape == shape:
        return mask
    rows = np.linspace(0, mask.shape[0] - 1, shape[0]).round().astype(int)
    cols = np.linspace(0, mask.shape[1] - 1, shape[1]).round().astype(int)
    return mask[rows][:, cols]


class InkStage2:
    """Refinador de tinta, desligado por padrão."""

    def __init__(self, model_path=None, enabled=False):
        self.model_path = model_path or os.environ.get(
            "DRAW_INK_V2_MODEL", _DEFAULT_MODEL
        )
        self.enabled = enabled
        self._session = None

    @classmethod
    def from_env(cls):
        """Liga só se DRAW_INK_V2_ENABLED=1; caminho por DRAW_INK_V2_MODEL."""
        return cls(enabled=os.environ.get("DRAW_INK_V2_ENABLED") == "1")

    def is_available(self):
        """True se o estágio está ligado e o arquivo do modelo existe."""
        return self.enabled and os.path.isfile(self.model_path)

    def predict_ink(self, image_rgb):
        """Devolve máscara booleana de tinta do especialista.

        image_rgb: array HxWx3, uint8 ou float. Exige onnxruntime;
        sem ele, levanta RuntimeError em vez de fingir resultado.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise RuntimeError("onnxruntime não instalado") from exc
        if self._session is None:
            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
        img = np.asarray(image_rgb)
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        # ONNX do ink-v2: input "image", output "logits", RGB 0-1.
        inp = self._session.get_inputs()[0].name
        out = self._session.get_outputs()[0].name
        logits = self._session.run(
            [out], {inp: img.transpose(2, 0, 1)[None]}
        )[0][0]
        return logits.argmax(axis=0) == _INK_CLASS

    def refine(self, seg_mask, image_rgb=None, specialist_mask=None):
        """Refina só a tinta de seg_mask; sem estágio, devolve cópia.

        Passe image_rgb (roda o especialista) OU specialist_mask
        (máscara booleana já pronta, útil para testes e para quem já
        rodou o modelo fora daqui). Sem nenhum dos dois, não há o que
        refinar e a máscara volta igual.
        """
        out = np.array(seg_mask, copy=True)
        if not self.enabled:
            return out
        if specialist_mask is None:
            if image_rgb is None or not os.path.isfile(self.model_path):
                return out
            specialist_mask = self.predict_ink(image_rgb)
        ink = _resize_nearest(np.asarray(specialist_mask, dtype=bool), out.shape)
        paper_or_ink = (out == PAPER) | (out == INK)
        new_ink = ink & paper_or_ink
        out[(out == INK) & ~new_ink] = PAPER
        out[new_ink] = INK
        return out
