"""Pareamento imagem/máscara do treino, nos dois layouts aceitos.

Layout images/masks (config/notebook) e layout plano do gerador
(scene_*.png + scene_*_mask.png + scene_*_ink_full.png).
Regressão da divergência encontrada no QA do neural-seg-v1:
o treino rejeitava a saída do gerador.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from train import pair_images_masks


def _write_img(path, value=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8, 3), value, dtype=np.uint8))


def _write_mask(path, value=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8), value, dtype=np.uint8))


def test_images_masks_layout(tmp_path):
    split = tmp_path / "train"
    _write_img(split / "images" / "a.png")
    _write_mask(split / "masks" / "a.png")
    pairs = pair_images_masks(split)
    assert len(pairs) == 1
    assert pairs[0][2] is None


def test_flat_generator_layout(tmp_path):
    split = tmp_path / "train"
    _write_img(split / "scene_train_0000.png")
    _write_mask(split / "scene_train_0000_mask.png")
    _write_mask(split / "scene_train_0000_ink_full.png", 255)
    pairs = pair_images_masks(split)
    assert len(pairs) == 1
    assert pairs[0][1].name == "scene_train_0000_mask.png"
    assert pairs[0][2] is not None and pairs[0][2].name == "scene_train_0000_ink_full.png"
