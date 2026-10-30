# Segundo estágio de tinta (ink-v2)

Módulo: `draw/ink_stage2.py`. Classe: `InkStage2`.

## O que é

Pós-processamento opcional sobre a máscara de 4 classes do SegFormer
(0=background, 1=paper, 2=hand, 3=ink). O especialista é o UNetLite
do kernel `guzzbr/draw-ink-train-v2` (3 classes, tinta = classe 2,
input ONNX `image`, output `logits`).

Regra simples: a tinta refinada só aparece onde o SegFormer já via
paper ou ink; mão e fundo nunca são tocados; tinta sem confirmação
do especialista volta a ser paper.

## Ligado / desligado

Desligado por padrão. Para ligar:

```python
from draw.ink_stage2 import InkStage2
stage = InkStage2(enabled=True)  # ou InkStage2.from_env()
refined = stage.refine(seg_mask, image_rgb=frame_rgb)
```

Ou por ambiente: `DRAW_INK_V2_ENABLED=1` e, opcionalmente,
`DRAW_INK_V2_MODEL=/caminho/draw_ink_v2.onnx`.

Sem modelo no caminho, ou sem `image_rgb` e sem máscara pronta,
`refine()` devolve a máscara igual, sem erro.

## Checkpoint

Arquivo: `models/draw_ink_v2.onnx` (1.968.292 bytes), baixado do
kernel `guzzbr/draw-ink-train-v2` e conferido contra o log da corrida:

- sha256: `5160cc4649ca5c477dafde0cb2c642ff87f52c463bf1717ffa712c8a1c400314`
- corrida: melhor snapshot época 5, IoU subset 0,7994, avaliação
  completa IoU médio 0,8072 (números do log do kernel, não
  revalidados aqui; a métrica final do ink-v2 em `metrics.json`
  ainda não foi baixada por rate limit do Kaggle em outras rotas).

## O que NÃO muda

- Contrato do SegFormer (4 classes, `logits`): intacto.
- `draw/neural.py` da main: não tocado. Este módulo vive na
  training-zone e só opera sobre a máscara já pronta.
- Sem merge em main nem em training-zone: esta branch é só o
  segundo estágio, para revisão.
