"""Tinta por modelo aprendido página→limpa (U-Net, ONNX).

O modelo recebe a página retificada (BGR uint8, qualquer tamanho) e
devolve, por pixel, a probabilidade de ser tinta manuscrita — pauta,
impresso, mão, caneta e sombra ficam fora. É a formulação do reinício
de 2026-10-31: em vez de diferenciar duas fotos sujas (referência,
tecla C, realinhamento), o modelo classifica a página direto e o app
só acumula a união das máscaras. Régua do treino (Kaggle, kernel
guzzbr/draw-page-clean-train-v0): IoU 0,80 com limiar 0,7 sobre
48 páginas held-out sintéticas, contra 0,18 do limiar de luminância.

Inferência por mosaico de 256×256 sem sobreposição, o mesmo
protocolo da avaliação do treino: a página é cortada em blocos de
256, a borda incompleta é preenchida com zeros, e os logits voltam
para a grade da página antes da sigmoide. A alternativa óbvia —
rodar a página em meia resolução e ampliar a máscara — foi medida e
reprovada nesta VM: IoU 0,85 → 0,22, porque o modelo aprendeu traço
fino na resolução cheia e a redução apaga o traço antes da rede. O
mosaico custa ~1,36 s por página de 864×1200 nesta VM (mediana de 20
páginas, 4 threads); no app ele roda nos quadros de detecção, não em
todos os quadros, e o canvas é união, então página já lida não
precisa ser relida.

A sessão é preguiçosa e segue a convenção de draw.neural: threads
intra-op por DRAW_ORT_THREADS, onnxruntime importado no primeiro
uso, erro explícito sem a dependência ou sem o arquivo.
"""

import os
from pathlib import Path

import numpy as np

_PATCH = 256
_ENV_MODEL_VAR = "DRAW_PAGE_CLEAN_MODEL"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "page_clean_unet.onnx"
)
DEFAULT_THRESHOLD = 0.7


def _load_session(model_path):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime nao esta instalado; o modelo pagina->limpa "
            "precisa dele. Instale com: pip install onnxruntime"
        ) from error
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"modelo ONNX nao encontrado: {path}. Passe o caminho do "
            f"arquivo .onnx ou defina a variavel {_ENV_MODEL_VAR}."
        )
    # Mesma convencao de threads do detector de folha (draw.neural):
    # 4 intra-op por padrao, DRAW_ORT_THREADS sobrescreve.
    from .neural import _ort_intra_threads

    options = ort.SessionOptions()
    options.intra_op_num_threads = _ort_intra_threads()
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class PageCleanInk:
    """Máscara de tinta da página retificada, por mosaico de 256."""

    def __init__(self, model_path=None, threshold=DEFAULT_THRESHOLD):
        self.model_path = str(
            model_path
            or os.environ.get(_ENV_MODEL_VAR)
            or DEFAULT_MODEL_PATH
        )
        self.threshold = float(threshold)
        self._session = None
        self._input_name = None

    def load(self):
        """Carrega a sessão ONNX. Retorna True; erro sobe explícito."""
        self._session = _load_session(self.model_path)
        self._input_name = self._session.get_inputs()[0].name
        return True

    def ink_mask(self, warped_bgr):
        """Máscara uint8 0/255 da tinta, no tamanho do warp de entrada."""
        if self._session is None:
            self.load()
        h, w = warped_bgr.shape[:2]
        logits = np.zeros((h, w), dtype=np.float32)
        for y in range(0, h, _PATCH):
            for x in range(0, w, _PATCH):
                tile = warped_bgr[y:y + _PATCH, x:x + _PATCH]
                pad_h = _PATCH - tile.shape[0]
                pad_w = _PATCH - tile.shape[1]
                if pad_h or pad_w:
                    tile = np.pad(tile, [(0, pad_h), (0, pad_w), (0, 0)])
                tensor = (
                    tile.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
                )
                out = self._session.run(
                    None, {self._input_name: tensor}
                )[0]
                hh = min(_PATCH, h - y)
                ww = min(_PATCH, w - x)
                logits[y:y + hh, x:x + ww] = out[0, 0, :hh, :ww]
        prob = 1.0 / (1.0 + np.exp(-logits))
        return np.where(prob > self.threshold, 255, 0).astype(np.uint8)
