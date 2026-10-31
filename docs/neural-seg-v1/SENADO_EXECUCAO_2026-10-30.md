# Execução do plano de validação e melhoria

Data: 2026-10-30
Estado: comparação Kaggle concluída; os dois braços e os gates de paridade ONNX passaram. Avaliação real sem holdout aprovado.

## Resultado executivo

- A revisão especializada liberou **19 máscaras** da sessão 1280×720 para um piloto de treino. As avaliações cegas de A/B e a revisão independente de C aceitaram esses quadros; o usuário instruiu que o consenso dos agentes vale como revisão validada para este fluxo. Isso não transforma as máscaras em ground truth humano.
- Os **20 candidatos** da sessão 640×480 não passaram o gate de overlay. Ninguém aprovou as máscaras sem ajustes; B marcou zero como aceitos também na proposta v3. Essa sessão fica fora da avaliação deste piloto.
- O pacote mantém `255` nos pixels desconhecidos. O trainer exclui esses pixels de CE, Dice e métricas. Foram confirmados suporte não nulo para as quatro classes nos 19 quadros.
- O Kaggle terminou um modelo de tinta sintética de três classes, com IoU médio de tinta `0.9214`; isso não é comparável ao SegFormer de quatro classes nem prova qualidade em webcam.
- Os três ONNX preexistentes passaram checagens estruturais e inferência CPU; esses testes locais, por si sós, não demonstraram paridade PyTorch–ONNX. Para os dois modelos recém-treinados, o gate Kaggle de paridade passou.
- A comparação Kaggle privada terminou, com a mesma configuração e seed: baseline sintético `mIoU=0.9177035`; sintético + 19 máscaras `mIoU=0.9182410`, diferença `+0.0005375` (cerca de `+0.054` ponto percentual). São métricas nos mesmos 300 exemplos sintéticos, com 78.643.200 pixels válidos; não demonstram desempenho real. IoU de tinta ficou praticamente igual e levemente menor no braço piloto (`0.743175` contra `0.743343`). Não há evidência de ganho prático.

## Revisão especializada das máscaras

Os revisores especializados A e B examinaram as 30 fontes de calibração sem rótulos e depois avaliaram as propostas; C fez adjudicação independente dos 19 candidatos finais. O CSV de consenso contém os IDs, hashes e dimensões. O pacote não inventa horários individuais de revisão que não estavam nos relatórios.

Na sessão de treinamento, 19 máscaras ficaram com consenso triplo. Nos 17.510.400 pixels agregados, há 12.749.215 de fundo, 3.325.607 de papel, 820.556 de mão, 107.683 de tinta e 507.339 desconhecidos (`255`). A revisão apontou que as cenas são parecidas e que as bordas finas ainda têm incerteza; por isso os 19 quadros servem para um piloto exploratório, não como conjunto amplo ou holdout.

Na sessão antiga, a revisão de 20 candidatos encontrou desconhecidos espalhados, fronteiras de página incompletas, panfleto impresso confundido com escrita e objetos (caneta/fones) sem classe confiável. Os revisores não liberaram um holdout utilizável. Não calcularemos IoU real sobre essas máscaras.

Pacote local: `captures/triage/training_ready/expert_consensus_20261030_v1/`. Os arquivos fonte e as propostas originais foram preservados. `samples.csv` SHA-256: `87ebfaa949c116ab72bd2ca168a78bd6ace1b7a65ad0e187864bbe564d41150f`; manifesto SHA-256: `53433ef32aa11fdbea3248f460617b15a0d5c4269d5fbb1b8f8e8f47ac8c383b`.

## Treino Kaggle do piloto

Dataset privado: `guzzbr/draw-segformer-expert-pilot-v1` (inclui as 19 imagens/máscaras, manifesto, cópia do trainer/config atual e wheel offline do ONNX Runtime). O kernel novo é `guzzbr/draw-segformer-expert-consensus-pilot-v1`; usa a GPU T4, cache local de `nvidia/mit-b0`, os mesmos chunks sintéticos e seed/configuração nos dois braços.

1. `synthetic_only`: baseline, sem imagens reais.
2. `synthetic_plus_expert19`: adiciona os 19 quadros revisados ao mesmo treino sintético.

Cada braço exporta ONNX e o notebook exige `onnx_validation.status == "ok"`. O resultado deve ser interpretado como comparação exploratória na validação sintética. Não há holdout real e não se deve dizer que o modelo melhorou em webcam com base nesse número.

Resultados baixados e conferidos em `captures/triage/kaggle_expert_pilot_output/`. O gate ONNX do baseline reportou status `ok`, diferença máxima de logits `5.72e-05`, acordo argmax `1.0`; o braço piloto reportou `ok`, `3.81e-05` e acordo `1.0`. Os IoUs argmax por classe foram `1.0` para ambas as exportações. SHA-256 dos ONNX: baseline `378833699a0bf55986ac199c1a1afa23a0e67afcd65d16fe00fa2ee1f6e8a594`; piloto `202360117fa811542b26cc8fc88c4ba6b01346a5d0b130258013bfd066785ea9`.

## Auditoria dos modelos existentes

| Artefato | Evidência | Limite |
|---|---|---|
| SegFormer-B0, 4 classes | Kaggle completo; validação sintética `mIoU=0.9137`, `ink IoU=0.7269`; input ONNX `image [1,3,512,512]`, logits `[1,4,128,128]`; hash `2ca5ea38210f8f1c7b4a6e981de89c0c434437341a18b7317c6733daba11c908`. | `onnx_validation=null`; 719 exemplos reais do run anterior eram pseudo-rótulos. Não prova qualidade em webcam. A saída 128×128 precisa ser ampliada pelo consumidor; a inferência atual da aplicação faz o resize. |
| Ink pareado v3 | Log: `IoU2=0.8507`, precisão `0.8844`, recall `0.9562`, `n=213`; `n` é o número de pares de avaliação com união de tinta não vazia. Input ONNX `pair [1,3,H,W]`, saída de 3 classes. Hash `1ccc83f22aa9f0c3d68b58b47578fba381c0d3923e7bb9ce1b4ab88cbb9353fe`. | Kernel terminou em erro após exportar porque não havia `onnxruntime`. Seu IoU é tinta nova em before/after e não se compara ao IoU absoluto do SegFormer. |
| Ink v1 sintético | Kaggle completo; 18.000 páginas geradas (16.000 treino/2.000 avaliação), 12.000 passos, IoU médio de tinta `0.9214` em 1.585 páginas sintéticas positivas; vazamento para impressão `0.0256`. Input `image [1,3,H,W]`, saída de 3 classes. Hash `57765d87428b71d8c0d881216ae16ecc2bbef8652950cc464518645de671b4ec`. | Não tem classe mão e foi avaliado apenas em páginas geradas. Não substitui o SegFormer nem mede a webcam. |

Os três ONNX passam `onnx.checker` e execução CPU com tensores zero 512×512; as saídas têm shapes válidos e valores finitos. Esta verificação prova que os grafos abrem e executam; **não** prova paridade com PyTorch nem qualidade semântica real.

## Software e dados externos

- Criado `tools/make_mask_review_app.py`: revisão offline em duas passagens (imagem sem rótulo, depois sobreposição), revisão por quadro, histórico separado por avaliador e exportação CSV. Correções de pincel são salvas como máscara derivada; fonte e proposta ficam intactas.
- Criado `tools/package_expert_consensus_pilot.py`: exige IDs congelados, valida hashes, resolução, valores permitidos e suporte das quatro classes; preserva `255` e não inventa holdout.
- `training/train.py` e `tools/seg_metrics.py` agora tratam `255` como ignorado em loss/contagens/métricas. `tools/validate_onnx.py` recusa classes sem suporte; a validação de exportação registra diferença de logits e IoU por classe e fecha o kernel caso o estado não seja `ok`.
- SAM 2 pode acelerar propostas de máscara e propagação por vídeo, mas é segmentação de objetos sem rótulos Draw; as propostas ainda precisam da distinção semântica entre folha, mão, tinta, impressão e objetos. O SA-V não contém rótulos dessas classes. Fontes: [Meta SA-V](https://ai.meta.com/datasets/segment-anything-video), [Meta SAM 2](https://ai.meta.com/research/sam2).
- O [FiftyOne Brain](https://docs.voxel51.com/brain/index.html) pode ordenar diversidade e sugerir quase duplicatas para reduzir revisão redundante; deve apenas ordenar a fila, nunca excluir frames automaticamente.
- O [Handwriting Inpainting Dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-2886) fornece máscaras binárias de escrita em páginas digitalizadas sob CC BY 4.0. Pode ajudar a iniciar um detector de tinta, mas não rotula mãos, papel de webcam, canetas ou oclusão. Não foi misturado ao treino.
- O rastreador SAM 2 documentado pelo CVAT fica em Online/Enterprise/fluxos específicos; o fluxo comunitário local não deve presumir disponibilidade do tracker. Fonte: [CVAT SAM 2 Tracker](https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/).

Decisão: nenhum dataset externo foi somado ao SegFormer. Os dados encontrados são úteis como ferramentas de triagem ou aquecimento parcial de tinta, mas não fornecem as quatro classes do Draw e não corrigem a ausência de holdout real.

## Próximos gates

1. Comparação, proveniência dos dados, hashes e gates ONNX: concluídos e registrados no pacote de saída.
2. Não publicar nem integrar o modelo piloto como “melhor” sem um holdout real aprovado; a diferença sintética observada é mínima.
3. Para avaliação real, revisar máscaras em outra sessão e reservar sessões inteiras por split; não reutilizar as 20 máscaras rejeitadas.
