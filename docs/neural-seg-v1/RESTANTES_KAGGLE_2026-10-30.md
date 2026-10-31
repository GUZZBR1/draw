# Execução das pendências Kaggle

Data: 2026-10-30
Plano: [.omx/plans/complete-kaggle-validation.md](../../.omx/plans/complete-kaggle-validation.md)

## O que foi concluído nesta rodada

- Corrigido o treinamento para marcar classes sem suporte como `null`/NA e calcular `mean_iou` somente nas classes suportadas; antes, uma classe ausente recebia IoU 1 e podia inflar a seleção de checkpoint.
- Corrigidas as métricas tolerantes de tinta: quando não há tinta em nenhum dos lados, retornam NA; pixels `255` continuam excluídos.
- `tools/validate_onnx.py` e o caminho de validação de `training/train.py` agora rejeitam logits NaN/infinitos.
- O treino salva seus artefatos e métricas de diagnóstico, mas retorna código 2 quando o ONNX fica `skipped`, `failed` ou `degraded`. A notebook Kaggle já tinha um gate equivalente.
- O construtor de splits agora exige sessões de treino e uma sessão de holdout explícitas. Recusa as sessões `session_20261029_130013_453419` e `session_20261029_151126_089557` como holdout, pois ambas já foram usadas em treino. O teste de regressão confirmou a recusa antes de escrever qualquer manifesto.
- Inspecionados os pares Kaggle v3 e v4; snapshots de código estão em `captures/triage/kaggle_kernel_remote/`. v3 confirma que `n=213` conta pares de eval com união não vazia de tinta. v3 falhou após exportar por ausência de `onnxruntime`.
- Revalidados os outputs do piloto SegFormer com 19 máscaras de consenso: ONNX status `ok` nos dois braços, diferença máxima de logits `5,72e-05` e `3,81e-05`, acordo argmax `1,0`. No conjunto sintético comum, `mIoU` mudou de `0,9177035` para `0,9182410` (`+0,0005375`), enquanto IoU de tinta caiu levemente de `0,7433430` para `0,7431751`. Não há evidência de ganho prático ou real.

## Evidência e validação

- `pytest -q tests/test_seg_metrics.py tests/test_build_expert_reviewed_split.py`: **21 passed**.
- `python3 training/train.py --smoke ...`: **SMOKE OK** para pareamento, pixels ignorados, métricas e contrato de saída. A parte da loss Torch foi pulada porque `torch` não está instalado neste Python.
- `py_compile` dos módulos modificados: passou.
- `pytest -q`: **94 passed, 5 failed**. As cinco falhas estão fora dos módulos alterados e vêm de incompatibilidade local em `cv2.aruco`: faltam `ArucoDetector` e `generateImageMarker`, usados por `draw/paper.py` e `template/generate_template.py`.
- O runtime local ainda não tem `torch`, `onnx` nem `onnxruntime`; portanto, não executei paridade local. Os gates ONNX do piloto foram executados no Kaggle e estão registrados nos respectivos `metrics.json`.

## Estado Kaggle mais recente observado

| Kernel | Estado | Conclusão permitida |
|---|---|---|
| `guzzbr/draw-segformer-v1-4class` | COMPLETE | Execução antiga; ONNX foi exportado, mas seu runtime validation foi pulado. |
| `guzzbr/draw-segformer-expert-consensus-pilot-v1` | COMPLETE | Ambos ONNX passaram paridade no Kaggle; resultado comparativo é apenas sintético. |
| `guzzbr/draw-ink-pairs-train-v3` | ERROR | IoU2 `0,8507`, `n=213`; exportou ONNX e falhou sem `onnxruntime`. |
| `guzzbr/draw-ink-pairs-train-v4` | RUNNING na última consulta | Não iniciar ou substituir outro kernel enquanto estiver ativo. O código v4 captura erro de runtime e registra `onnx_check`; isso não garante que o runtime exista nem que paridade passe. |

Hashes dos snapshots baixados: v3 `4aa37aa2a4aab80a26c14cae7d7e85a1ebc9984f1d096f0b3d60954a79d84a03`; v4 `7f2bf421894630bc10b7305ac64015ecb1dc5438d2b3f394e35ad8399cb41dc3`.

## Bloqueio que requer nova captura e revisão

As capturas locais só contêm as duas sessões já atribuídas ao treino. A fila com 90 quadros da sessão de 1280×720 tem 89 máscaras parciais sem rótulo de tinta e uma máscara completa só de fundo. Os 20 candidatos da sessão de 640×480 não passaram revisão. Dividir frames dessas sessões criaria leakage, e chamar a fila parcial de ground truth não seria válido.

Para fechar qualidade real, ainda é necessária uma sessão nova mantida inteira fora do treino, com amostra representativa de pelo menos 60–100 frames e rótulos nativos revisados para fundo, papel, mão e tinta. Pixels incertos ficam como `255`. O fluxo de captura/anotação e o construtor de split estão prontos para gerar o manifesto depois que essa sessão e seus rótulos existirem; a avaliação não pode ser completada sem eles.

Depois de capturar e adicionar a sessão ao CSV de candidatos, congele o split antes da revisão de máscaras, declarando as sessões de treino e a sessão nova de holdout explicitamente:

```bash
python3 tools/build_expert_reviewed_split.py \
  --samples captures/triage/CANDIDATO_NOVO/samples.csv \
  --train-session session_20261029_130013_453419 \
  --train-session session_20261029_151126_089557 \
  --holdout-session ID_DA_SESSAO_NOVA \
  --write-freeze-manifest captures/triage/CANDIDATO_NOVO/split-freeze.json
```

Substitua `CANDIDATO_NOVO` e `ID_DA_SESSAO_NOVA` pelos caminhos/ID reais depois da captura. O argumento `--previously-trained-session` é repetível e deve listar qualquer sessão adicional já usada. O construtor rejeita os dois IDs antigos como holdout mesmo se alguém os passar por engano.

**Estado final:** gates de código e métricas locais corrigidos; benchmark Kaggle sintético revalidado; holdout real e paridade do modelo pareado v4 aguardam respectivamente uma nova sessão anotada e o término/output do kernel já em execução.
