"""Coleta (U1): grava exemplos de sessao no formato de dataset.

Cada registro vira um arquivo em input/ (o warp BGR), um em prob/
(o mapa de probabilidade em uint8) e uma linha em manifest.jsonl
com o sha256 dos bytes da imagem gravada, sessao, instante, modelo
e origem. A linhagem por linha e o que permite expurgar depois tudo
que uma origem ruim tocou. Privado por padrao: isto escreve em disco
local, nunca publica nada.
"""

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np


class ExampleRecorder:
    """Gravador de exemplos. Uma chamada de record() por passada."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.input_dir = self.out_dir / "input"
        self.prob_dir = self.out_dir / "prob"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.prob_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.jsonl"
        self._count = 0

    def record(self, session_id, warped_bgr, prob, corners,
               model_sha, origin):
        """Grava um exemplo e retorna a linha do manifesto (dict)."""
        stamp = time.time()
        name = f"{session_id}_{int(stamp * 1000)}_{self._count:06d}"
        self._count += 1

        ok, buf = cv2.imencode(".png", warped_bgr)
        if not ok:
            raise RuntimeError("falha ao codificar o warp em PNG")
        image_bytes = buf.tobytes()
        (self.input_dir / f"{name}.png").write_bytes(image_bytes)

        prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(self.prob_dir / f"{name}.png"), prob_u8):
            raise RuntimeError("falha ao gravar o mapa de probabilidade")

        row = {
            "file": name,
            "session_id": session_id,
            "ts": stamp,
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "model_sha": model_sha,
            "origin": origin,
            "corners": np.asarray(corners, dtype=float).tolist(),
        }
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row
