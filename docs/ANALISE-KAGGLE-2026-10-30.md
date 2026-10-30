# Análise completa dos notebooks e datasets do Draw no Kaggle — conta GUZZ BR

**Data da análise:** 30 de outubro de 2026
**Conta:** GUZZ BR (`guzzbr`) — conta canônica do projeto desde 29 de outubro de 2026. A conta antiga `guzzfort` foi abandonada por decisão do usuário e não faz parte desta análise, exceto por um kernel de teste de fumaça citado explicitamente como antecedente.
**Escopo:** somente leitura. Nenhum kernel foi criado, rodado, alterado ou interrompido para esta análise. Nenhum output em massa foi baixado (o endpoint de outputs do Kaggle estava sob rate limit 429 em verificações repetidas). Publicado na main do fork GUZZBR1/draw em 30/10/2026, após revisão, a partir do rascunho em `/tmp`. Não foi enviado ao upstream santleme/draw.
**Fontes primárias verificadas:** `metrics.json` final do treino principal (lido diretamente do arquivo baixado, não apenas do log do kernel), `training_config.json`, inventário Kaggle por CLI em 30 de outubro de 2026, código dos dois kernels novos baixado com `kernels pull` (somente código), documentação da training-zone e o relatório da noite do upstream (`docs/noite/2026-10-30.md`).
**Regra de honestidade usada em todo o documento:** tudo que foi lido apenas no código, sem métrica de corrida confirmada em output, está marcado explicitamente como **não verificado (só código)**. Nenhuma métrica foi inventada para preencher lacuna.

---

## 1) Inventário categorizado de notebooks (kernels)

Total listado na conta GUZZ BR: **26 kernels** (25 da varredura anterior + 1 novo). Cinco notebooks sem nome retornam 404 em `status` e não são avaliáveis como modelo; seguem listados como infra.

### 1.1 Treino principal — o modelo de referência do projeto

| Kernel | Status | Última execução | Papel |
|---|---|---|---|
| `guzzbr/notebooka37f38640e` | COMPLETE | 29/10/2026 16:43 (-03), GPU T4 x2, 11 inputs | **Treino completo SegFormer-B0 4 classes.** É o modelo de referência. Métricas finais confirmadas em `metrics.json` (seção 3.1). |
| `guzzbr/draw-segformer-v1-4class` | COMPLETE | 30/10/2026 14:23 (-03) | **Duplicata re-executada.** Código é o notebook oficial antigo do SegFormer-B0 4 classes, 30 épocas via `training/config.json`. Não traz modelo novo. Métricas desta re-execução **não verificadas** (não baixadas, regra anti-429). Não deve ser citada como um segundo modelo. |

### 1.2 Especialistas de tinta — a classe fraca, atacada em três gerações

| Kernel | Status | Última execução | Papel |
|---|---|---|---|
| `guzzbr/draw-ink-v1-full` | COMPLETE | 29/10/2026 22:46 (-03) | **Geração 1, superada. Não usar.** UNetLite com tinta escura e saturada demais; cega para tinta real (seção 3.4). |
| `guzzbr/draw-ink-train-v2` | COMPLETE | 30/10/2026 02:09 (-03) | **Geração 2, a integrada.** UNetLite 3 classes treinado no regime de tinta fraca. Base do segundo estágio `draw/ink_stage2.py`, já integrado à training-zone (seção 3.3). |
| `guzzbr/draw-ink-pairs-train-v3` | **RUNNING** | Iniciado 30/10/2026 16:28 (-03) | **Geração 3, em andamento. Sem conclusão possível.** Muda a formulação do problema: tinta nova = diferença entre folha antes e depois (seção 3.5). **Não tocar enquanto estiver rodando.** |

### 1.3 Cantos da folha — complemento, não segmentação

| Kernel | Status | Última execução | Papel |
|---|---|---|---|
| `guzzbr/draw-cornernet-train` | COMPLETE | 28/10/2026 17:47 (-03) | Detector de cantos por heatmap, treinado com oclusões fortes cruzando cantos. Complemento ao quad da folha sob oclusão, ponto fraco do detector clássico. Métrica held-out **não verificada (só código)**. |

### 1.4 Fase 2 — ideias e receita, não modelo pronto

| Kernel | Status | Última execução | Papel |
|---|---|---|---|
| `guzzbr/draw-phase2-train-v4` | COMPLETE | 30/10/2026 01:53 (-03) | Mesmo encoder mit-b0, decoder MLP, 3 saídas (máscara 2 classes, heatmaps de 4 cantos, cabeça de presença aprendida), aumento de oclusão por cápsula (p = 0,30), consistência fotométrica (peso 0,1). **Só 2 classes, sem hand e sem ink: não é modelo usável para o Draw.** Aproveitamento: ideias portáveis. Contradição interna a esclarecer: a docstring diz "PREPARADO, NÃO LANÇADO", mas o status é COMPLETE. Métricas **não verificadas (só código)**. |
| `guzzbr/draw-phase2-train-v3` | COMPLETE | 29/10/2026 17:26 (-03) | SegFormer-B0 2 classes, dados MIDV/SmartDoc/Lucas-holdout/school/COCO-negativos, AMP FP16, batch 16, LP-FT, speed gate. **Só 2 classes: não é modelo usável.** Aproveitamento: receita portável para um futuro retreino 4 classes. Métricas **não verificadas (só código)**. |
| `guzzbr/draw-phase2-train-v2`, `guzzbr/draw-phase2-train` | COMPLETE | 29/10/2026 e 28/10/2026 | **Superados pela v3.** Não usar. |
| `guzzbr/draw-v3-gate` | COMPLETE | 29/10/2026 16:08 (-03) | Subconjunto da v3. Inclui o loader de JPEG em modo draft, já provado que quebra geometria. Não usar como base. |

### 1.5 Prova inicial, benchmarks, infra e outro projeto

| Kernel | Categoria | Observação |
|---|---|---|
| `guzzbr/draw-seg-train` | Prova fase 1b | 2 classes, 3 épocas. Valor histórico apenas. |
| `guzzbr/draw-bench-t4`, `guzzbr/draw-bench-tpu` | Benchmark | Idênticos entre si em código. Medem custo, não qualidade de modelo. O bench TPU está com status CANCEL_ACKNOWLEDGED, ou seja, cancelado, não COMPLETE — não usar como resultado. |
| `guzzbr/draw-v3-coco-probe`, `guzzbr/draw-v2-probe`, `guzzbr/draw-smartdoc-mount-check`, `guzzbr/draw-mount-check`, `guzzbr/draw-smoke-hello` | Infra / probe | Sondas, checagens de montagem e templates. Não são modelos. |
| `guzzbr/notebooke68447697a`, `guzzbr/notebook6bc72f43f1`, `guzzbr/notebook7eed735694`, `guzzbr/notebooke2b45c7185`, `guzzbr/notebookb685fd444d`, `guzzbr/notebookf044614743` | Sem nome | Não avaliáveis como modelo; 5 deles retornam 404 em `status` (prováveis rascunhos/apagados). |
| `guzzbr/arc3-qwen38-duck-d-prime-public-evaluation` | Outro projeto | ARC-AGI-3/Qwen. Nada a ver com o Draw. Excluído de todas as conclusões. |

**Conclusão do inventário de notebooks:** não existe, na conta GUZZ BR, um substituto direto e melhor do SegFormer-B0 4 classes. Não apareceu B1/B2, DeepLab, SAM ou outra segmentação de mão/tinta não utilizada. O que existe de aproveitável é indireto: um especialista de tinta (v2, integrado), um especialista de tinta pareada (v3, rodando), um detector de cantos (complemento, sem métrica confirmada) e ideias/receita da fase 2.

---

## 2) Inventário categorizado de datasets

Total listado na conta GUZZ BR: **18 datasets** na listagem Kaggle de 30/10/2026 (a contagem anterior de 20 estava desatualizada; não há paginação oculta visível). Datasets locais em `~/workspace/draw-data/` são a fonte dos uploads.

### 2.1 Treino principal — sintético, 3.000 cenas

| Dataset | Tamanho | Conteúdo |
|---|---|---|
| `guzzbr/draw-synthetic-v1-part1` | 223 MB | Treino, cenas 0–399, inclui `split_manifest.json` |
| `guzzbr/draw-synthetic-v1-part2..6` | ~223–225 MB cada | Treino, blocos de 400 cenas (400–2399) |
| `guzzbr/draw-synthetic-v1-part7` | 168 MB | Validação, 300 cenas |
| `guzzbr/draw-synthetic-v1-part8` | 167 MB | Teste sintético, 300 cenas |
| **Total** | — | **3.000 cenas 512×512, 9.001 arquivos: treino 2.400 / val 300 / teste 300.** Gerador com seed 42, split por fonte (mãos 960/120/120 disjuntas, fundos distintos por split), 2.500 cenas com uma mão + 500 com duas mãos, **600 negativos (20%)**: mesa vazia 120, mão sem folha 120, objeto branco 120, folha minúscula 120, folha fora do quadro 120. |

**Proporção de pixels no treino (contada nas 2.400 máscaras de treino):** background 58,288%, paper 38,949%, hand 1,508%, ink 1,255%. O desbalanceamento é estrutural: tinta é 1,26% dos pixels, mão é 1,51%. Qualquer loss sem peso esmaga essas duas classes; qualquer métrica global (acurácia) as esconde.

**Limitação central do sintético:** tinta gerada, fundos de mesa limitados, mãos FreiHAND recortadas — sem oclusão real de caneta, sem sombra real, sem papel amassado, sem caneta cruzando a borda (o caso real t17, seção 4). É por isso que um val_mean_iou sintético de 0,9183 não se traduz diretamente para foto real.

### 2.2 Especialista de tinta

| Dataset | Tamanho | Conteúdo e limitação |
|---|---|---|
| `guzzbr/draw-ink-synth-v1` | 2,75 GB | Sintético do especialista ink-v2 (UNetLite 3 classes). É o maior dataset da conta. Tinta cinza de baixa saturação, degradação simulada. Limitação: sintético; a degradação é simulada, não capturada. |
| `guzzbr/draw-ink-real-v1` | 8,3 MB | Warps reais rotulados usados pelo ink-v2. Limitação: pequeno; risco de overfit ao pouco real. |
| `guzzbr/draw-ink-pairs-v1` | **0 bytes na listagem, populado na prática** | **Novo, insumo do ink-pairs-train-v3. Discrepância parcialmente resolvida:** a listagem mostra 0 bytes, mas `datasets files` lista a primeira página (`after/eval_00000.png` … `after/eval_00019.png`, ~0,7–1,3 MB cada), então o 0 é lag de metadado, não dataset vazio. **Não verificado:** contagem total de arquivos, nem se `before/` e os rótulos dos 16.000 pares descritos no código estão completos — só a primeira página foi lida, sem paginação exaustiva para evitar rajada. Licença no metadado: CC0-1.0, descrição vazia. |

### 2.3 Holdout real e piloto real — o ponto mais sensível do projeto

| Dataset | Tamanho | Conteúdo e limitação |
|---|---|---|
| `guzzbr/draw-frames-lucas` | 53 MB | Frames de webcam do Lucas, 28/10/2026: 49 PNG 1280×720, **43 positivos / 6 negativos, 11 máscaras anotadas**. Limitações severas e conhecidas: máscaras com precisão estimada de ±20 px; oclusões de mão/manga **não** recortadas; imagens espelhadas (webcam); áreas superexpostas; **nenhum positivo tem os 4 cantos reais visíveis** — serve para presença (49) e segmentação parcial (11), **nunca** para derivar cantos ou homografia. |
| `guzzbr/draw-real-seg-pilot-v1` | 30 MB | **Novo, 30/10/2026 14:21 (-03). Inspecionado por manifesto, não por imagem individual.** 719 frames reais com máscaras, das sessões antiga e do lote de 90 takes, split só `train` no manifesto. Máscaras {0,1,2,3,255}, 567 frames com tinta, 707 supervisionados, 12 totalmente desconhecidos, 719 revisados em contact sheet, 21 com correções. **Limitação decisiva, dita pelo próprio manifesto: `dataset_kind: real_frame_pseudo_label_candidate` e `semantic_review_status: unverified; not ground truth` — triagem em thumbnail, erros grosseiros corrigidos, bordas finas de tinta não verificadas.** Válido como piloto exploratório de treino real, **não** como holdout de avaliação. Licença no metadado: "other", privado, não para redistribuição. |

### 2.4 Referência externa — calibra limiar, nunca treina as 4 classes diretamente

| Dataset | Tamanho | Uso correto e restrição |
|---|---|---|
| `guzzbr/smartdoc2015` | 1,04 GB | Documentos em fundos variados, com anotações. Não é escrita manual sobre folha. |
| `guzzbr/school-notebooks-ru` | 2,93 GB | 1.857 páginas de caderno russo (treino 1.557 / val 150 / teste 150), polígonos COCO de linhas de texto. **Os rótulos são texto, não as 4 classes do Draw.** |
| `guzzbr/midv2020-photos` | 3,97 GB | Fotos de 10 tipos de documento de identidade, com cantos anotados. **Restrição: documentos de identidade — não usar como treino do produto sem checar licença e privacidade.** |

### 2.5 Infra

| Dataset | Tamanho | Conteúdo |
|---|---|---|
| `guzzbr/draw-training-code` | 31 KB | `train.py` + `config.json`, pesos [1,0, 1,5, 10,0, 10,0]. **Re-versionado em 30/10/2026 14:21 (-03), na mesma hora do piloto real.** Antes de reutilizar, conferir se o conteúdo mudou em relação à versão usada no treino principal — a comparação anterior mostrou que a versão do dataset diferia da então-main apenas no texto de justificativa dos pesos, com hiperparâmetros idênticos, mas essa verificação precisa ser refeita para esta nova versão. |
| `guzzbr/mit-b0-cache` | 27 MB | Pesos `nvidia/mit-b0`, snapshot validado offline, `refs/main` com 40 bytes sem newline (detalhe que já quebrou um upload anterior e deve ser preservado). |

### 2.6 Datasets locais (fonte dos uploads)

| Diretório | Conteúdo |
|---|---|
| `~/workspace/draw-data/synthetic-v1/` | 1,6 GB, 9.003 arquivos: as 3.000 cenas originais que geraram as 8 partes Kaggle. |
| `~/workspace/draw-data/hands-freihand/` | 79 MB, 2.401 arquivos: 1.200 mãos FreiHAND extraídas do bloco 1 (RGB + máscara 224×224, threshold 127), com `provenance.json`. |
| `~/workspace/draw-data/freihand/` | 5,0 GB, 166.767 arquivos: download completo FreiHAND (`danieldelro/freihand`). RGB 130.240 imagens; máscaras reais só 32.560 (treino) — 4 blocos de fundo das mesmas 32.560 mãos-base. **O split deve ser por mão-base, nunca por arquivo.** |
| `~/workspace/draw-data/kaggle-tmp/`, `kaggle-tmp-guzzbr/`, `mit-b0-cache-guzzbr/`, `synthetic-v1-upload/` | Staging de upload e manifestos (`PARTS.md`, `PARTS_GUZZBR.md`). Não são datasets. |

### 2.7 Regras de rótulo e licença que valem para todos os datasets

- **Mão sobre tinta = mão, nunca tinta.** O modelo não deve aprender a enxergar através da mão.
- **Caneta = fundo.** A caneta em si não é tinta nem mão.
- **Texto impresso ≠ tinta.** Exemplo canônico: o folheto "CASA DOS ROLAMENTOS" na sessão antiga de capturas — texto impresso não pode ser rotulado como tinta. É por isso que o ink-v2 tem uma classe separada para impresso (classe 1), que funciona como supressora implícita de impresso.
- **FreiHAND: licença CC BY-NC 4.0, somente pesquisa, uso comercial proibido, citação obrigatória (Zimmermann et al., ICCV 2019).** Se o Draw se tornar comercial, um modelo treinado com FreiHAND tem um problema de licença a resolver antes — registrado como risco, não como bloqueio atual, porque o uso atual é pesquisa.
- **Sessão inteira = uma fonte.** Nunca dividir por frame entre treino e teste: uma sessão vai inteira para holdout ou inteira para treino, nunca metade para cada lado.

---

## 3) Análise por modelo: arquitetura, dados, por que foi bem, por que foi mal

### 3.1 SegFormer-B0 4 classes — treino principal (`guzzbr/notebooka37f38640e`)

**Arquitetura e configuração (verificadas em `training_config.json` e `metrics.json`):**
SegFormer-B0, encoder `nvidia/mit-b0` com transfer learning, entrada 512×512 com normalização ImageNet (média [0,485, 0,456, 0,406], desvio [0,229, 0,224, 0,225]), saída 4 classes (0 = background, 1 = paper, 2 = hand, 3 = ink). Loss = 0,5 × CrossEntropy ponderada + 0,5 × Dice multiclasse, pesos [1,0, 1,5, 10,0, 10,0] — pesos recalibrados a partir da contagem real dos pixels de treino (fundo 58,288%, papel 38,949%, mão 1,508%, tinta 1,255%), com cap em 10, substituindo os pesos antigos [1,0, 2,0, 3,0, 10,0] que subponderavam a mão. Otimizador AdamW, lr 6e-5, cosseno, early stopping com paciência 5 sobre val_mean_iou, seed 42, batch 4 no treino final (batch 8 era o risco de OOM em T4, em fp32 sem autocast). Treino: 30 épocas, GPU T4 x2, 11 inputs, **sem FreiHAND direto** (as mãos FreiHAND reais já estavam embutidas nas 3.000 cenas sintéticas; o dataset FreiHAND não foi anexado ao kernel por decisão do usuário). Contrato ONNX previsto: input `image` [1, 3, 512, 512], output único `logits` [1, 4, 512, 512]; `paper_presence` derivado da máscara de papel, não uma cabeça auxiliar.

**Resultados finais, melhor época 25, lidos de `metrics.json` (não do log):**

| Classe | IoU | Dice | Precision | Recall |
|---|---|---|---|---|
| background | 0,9974 | 0,9987 | 0,9997 | 0,9977 |
| paper | 0,9868 | 0,9934 | 0,9951 | 0,9916 |
| hand | 0,9414 | 0,9698 | 0,9518 | 0,9885 |
| ink | 0,7475 | 0,8555 | 0,7996 | 0,9198 |
| **Média (val_mean_iou)** | **0,9183** | — | — | — |

**Tinta com tolerância espacial (a métrica que muda a leitura do ink):**

| Tolerância | Precision | Recall |
|---|---|---|
| Raio 2 px | 0,9761 | 0,9998 |
| Raio 3 px | 0,9761 | 1,0000 |

**Curva de aprendizado (de `metrics.json`, campo `history`):** época 0: train_loss 0,3391, val_mean_iou 0,8609, ink IoU 0,6222. Época 10: train_loss 0,1210, val_mean_iou 0,9079, ink IoU 0,7233. Época 25 (melhor): val_mean_iou 0,9183, ink IoU 0,7475. Época 29 (final): train_loss 0,0815, val_mean_iou 0,9149, ink IoU 0,7428.

**Por que foi bem:**
- **O modelo já nasceu maduro.** Na época 0, com pesos pré-treinados do mit-b0, o val_mean_iou já era 0,8609. O transfer learning fez o trabalho pesado de contorno e textura; o treino refinou, não construiu do zero.
- **Fundo e papel saturam rápido e ficam.** Background 0,997 e paper 0,987 com precision e recall acima de 0,99 nos dois: áreas grandes, bem delimitadas e com textura de folha distinta da mesa são o problema fácil da segmentação, e o sintético as representa bem.
- **Mão chega a 0,941 porque tem variação real.** As mãos FreiHAND reais, com variação de pose, tom e fundo, dão ao modelo algo que o sintético puro não daria. O recall de 0,9885 mostra que o modelo raramente perde a mão; ele erra mais por excesso (precision 0,9518) do que por falta — o erro barato para o Draw, porque mão a mais vira área ignorada, não tinta inventada, desde que o portão da mão seja respeitado.
- **A tinta é achada, não perdida.** Recall de tinta 0,9198 no IoU estrito e 0,9998 com tolerância de 2 px: o modelo encontra praticamente toda a tinta. O problema não é ausência.

**Por que foi mal — e o ponto é preciso, não genérico:**
- **Tinta estrita é a única classe fraca, e o motivo é deslocamento, não ausência.** IoU 0,7475 com precision 0,7996 e recall 0,9198 significa: o modelo acha a tinta, mas prediz 20% de tinta a mais (ou deslocada) em relação ao rótulo. Com tolerância de 2 px, a precision salta de 0,7996 para 0,9761 e o recall para 0,9998. Conclusão: o erro dominante é de **1–2 px de deslocamento**, fatal para traço fino de 1–3 px, que é 1,26% dos pixels, mas irrelevante para o usuário final se o traço aparecer no lugar certo dentro de 2 px. Quem olhar só o IoU estrito 0,747 vai concluir que o modelo "não vê tinta"; quem olhar a tolerância vai concluir que ele vê tinta com erro sub-pixel. As duas leituras são verdadeiras em réguas diferentes, e a régua honesta para traço fino é a com tolerância, com o IoU estrito como secundário.
- **O modelo não está sub-treinado; está limitado pela representação da tinta.** De época 10 (0,9079) a época 25 (0,9183) foram só ~1 ponto em 15 épocas, com train_loss caindo de 0,1210 para 0,1136 e depois 0,0815 sem ganho equivalente em val. Mais época, na mesma distribuição de dados, não resolve tinta. O gargalo é dado e representação (espessura do rótulo sintético versus traço real, peso por borda, calibração sub-pixel), não época.
- **O peso 10 no CE ajuda, mas não corrige erro sub-pixel.** Sem peso, tinta e mão seriam esmagadas pelo fundo (58%) e pelo papel (39%). Com peso 10, elas aprendem — mas o CE ponderado não distingue "tinta 1 px ao lado" de "tinta inventada", e o Dice ajuda pouco em estruturas finas. Por isso a precision fica em 0,80 mesmo com recall alto.
- **ONNX: o erro mais caro do projeto foi de entrega, não de modelo.** A exportação falhou com o erro exato: "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor". Causa raiz: `export_onnx()` em `training/train.py` criava o dummy input na CPU enquanto o modelo estava na GPU — um bug de uma linha. Como a exportação era não-fatal por desenho (para não perder o treino por causa do ONNX), o treino "terminou bem" sem artefato entregável: `draw_segmentation_v1.onnx` está **ausente**, a validação PyTorch × ONNX **nunca rodou** (`onnx_validation.status = failed`), e o melhor checkpoint (`checkpoints/best.pt`, 14.932.509 bytes, época 25) ficou sem caminho para a aplicação. Consequência prática: o modelo 4 classes **ainda não está ligado em `draw/neural.py`**, que continua com o contrato de 2 canais do upstream (`mask_logits` + `presence_logit`). O modelo está pronto; o artefato não estava.
- **Nenhum holdout real foi avaliado.** A validação é sintética, do mesmo gerador do treino. Todo número desta seção é, rigorosamente, "desempenho em sintético". Isso não invalida o modelo, mas limita qualquer afirmação sobre foto real — e o precedente do ink-v1 (seção 3.4) mostra que sintético bom pode esconder real ruim.

**Artefatos finais confirmados:** `metrics.json`, `classes.json`, `training_config.json`, `split_manifest.json`, `checkpoints/best.pt` (14.932.509 bytes), todos baixados para `hidden_files/kaggle-full-training/`. `draw_segmentation_v1.onnx`: ausente. O download só se completou após o rate limit 429 ceder, com `--page-size 200` — e esse bloqueio atrasou a entrega final em dias, um custo operacional que precisa entrar no planejamento dos próximos notebooks (nunca deixar a leitura do `metrics.json` para o fim, em burst).

### 3.2 Teste de fumaça de 2 épocas / 36 cenas (antecedente, conta guzzfort)

**Arquitetura e dados:** mesma SegFormer-B0, 36 cenas sintéticas com mãos placeholder, 2 épocas, kernel `guzzfort/draw-seg-test-post-onnx-2-epochs-36-scenes`, ~225 s, offline.
**Resultados:** val_mean_iou 0,3714 → 0,4820; background 0,815, paper 0,695, hand 0,356, ink **0,062** (precision 0,093, recall 0,155); ink com tolerância raio 2 px: precision 0,124, recall 0,120. ONNX de 15.181.516 bytes gerado, mas validação PyTorch × ONNX pulada (`onnxruntime` ausente no kernel).
**Por que foi mal — e por que isso não é uma crítica ao modelo:** não é um modelo ruim, é um **teste de fumaça sub-treinado em dados placeholder**. 36 cenas não ensinam tinta fina e 2 épocas não convergem nada. O valor dele é outro: provar o pipeline completo offline e expor dois bugs que depois foram corrigidos — o Dice que retornava forma `[batch]` em vez de escalar (corrigido com `dice.mean()`) e o exportador dynamo do torch 2.11 que exigia o módulo ausente `onnxscript` (corrigido forçando o exportador legado, `dynamo=False`). **Comparar este 0,4820 com o 0,9183 do treino principal seria erro de categoria.** Ele existe nesta análise apenas para que ninguém, no futuro, confunda "teste de pipeline" com "modelo".

### 3.3 Ink-v2 especialista — geração 2, já integrada à training-zone

**Arquitetura e dados:** UNetLite, 3 classes (0 = fundo/outros, 1 = impresso, 2 = tinta), encoder 16→32→64→128 com GroupNorm(4) + GELU, decoder com skip connections, cabeça 1×1. Treino: crops RGB 512×512, **RGB/255 sem normalização ImageNet** (diferente do SegFormer — detalhe que proíbe média de probabilidades entre os dois modelos), batch 16, AdamW lr 3e-4, 6 épocas, loss CE ponderada [1, 2, 4] + Dice de tinta mascarado. Dados: tinta cinza de baixa saturação, delta local ~10–40 níveis (contra ~100 na v1), alfa por página [0,16, 0,48], degradação de captura simulada, negativos de impresso, e warps reais rotulados (`draw-ink-real-v1`, 8,3 MB). Inferência: **página já retificada** 850×1200 (pad 864×1216), não frame cru — o modelo não tem classe de fundo/mesa nem de mão.
**Resultados (do log da corrida, não de `metrics.json` revalidado — o `metrics.json` do ink-v2 nunca foi baixado):** melhor época **5 de 6**, subset IoU 0,7994, IoU médio da avaliação completa 0,8072, vazamento de impresso 1,41%, páginas sem tinta com P(tinta) média 0,00010. **Atenção:** são métricas próprias, em tarefa própria (página retificada, 3 classes, sem mão/fundo, subconjunto próprio). **Não são diretamente comparáveis ao ink IoU 0,747 do SegFormer.** Qualquer afirmação do tipo "ink-v2 é melhor que 0,747" seria falsa por comparação de réguas diferentes.
**Integração (verificada):** `draw/ink_stage2.py`, classe `InkStage2`, na training-zone em `5e71a0c` (branch `feature/ink-v2` criada a partir da training-zone, mergeada nela por fast-forward e depois apagada, com o conteúdo preservado). Regra única, booleana, sem ensemble: desligado por padrão; mão e fundo nunca tocados; tinta refinada só aparece onde o SegFormer já via paper ou ink; tinta sem confirmação do especialista volta a ser paper. Ligação por `DRAW_INK_V2_ENABLED=1` e caminho por `DRAW_INK_V2_MODEL`; sem modelo, `refine()` devolve a máscara igual, sem erro. Testes: **8 passed** (5 da implementação + 3 da etapa de garantias: desligado = SegFormer puro; ligado, só a classe ink muda; sem tinta do especialista, nenhuma tinta nova aparece) + selftest 96,03% na branch. ONNX real: `models/draw_ink_v2.onnx`, 1.968.292 bytes, sha256 `5160cc4649ca5c477dafde0cb2c642ff87f52c463bf1717ffa712c8a1c400314`, conferido contra o log da corrida. Input ONNX `image`, output `logits`, opset 17, eixos dinâmicos em H/W, dummy criado no device correto — ao contrário do bug do SegFormer, aqui o device estava certo.
**Por que foi bem:** foi bem exatamente onde o SegFormer foi mal por construção. O treino principal usou tinta sintética que, na v1, era escura e saturada demais; o especialista foi treinado justamente no regime que faltava (cinza fraco + degradação real + impresso como negativo). O pico na época 5 de 6 sugere conjunto especialista pequeno ou convergência rápida — bom para custo, ruim para generalização sem holdout real.
**Por que ainda não é uma vitória medida:** (a) a inferência ONNX **nunca foi rodada localmente** (sem `onnxruntime` nesta VM; `predict_ink()` levanta `RuntimeError` claro em vez de fingir resultado); (b) **nenhuma avaliação conjunta** SegFormer + estágio foi feita nas mesmas cenas — o ganho real sobre o 0,747 é hipótese, não medida; (c) o portão da mão é obrigatório porque o especialista não distingue mão/caneta de tinta: sem o `hand_mask` do SegFormer, a própria mão viraria tinta. A regra booleana do `ink_stage2.py` é o desenho certo justamente por isso; um ensemble por média de probabilidades seria o desenho errado, porque os dois modelos usam normalizações e resoluções diferentes.
**Primeira tentativa descartada (registro para não repetir):** o commit `d78a69b` implementava semântica incompatível (deixava fundo virar tinta e nunca removia tinta) e ficou órfão; **não deve ser mergeado**. A semântica correta é a da branch: fundo nunca vira tinta, tinta sem confirmação volta a paper.

### 3.4 Ink-v1 — geração 1, superada, e a lição mais importante do projeto

**Arquitetura e dados:** mesmo UNetLite, mas tinta escura e saturada, delta ~100.
**Resultado:** **cego em tinta real: mediana de P(tinta) = 0,0000 em pixels de tinta verdadeira.**
**Por que foi mal:** distribuição sintética errada. A tinta de treino não se parecia com tinta real fraca, então o modelo aprendeu um conceito de "tinta" que não existe na foto real. O val sintético aceitável não detectou isso, porque o val era do mesmo gerador errado.
**Por que esta seção existe num relatório sobre o futuro:** esta é a lição central do projeto inteiro. **Distribuição sintética errada mata o modelo no real mesmo com val sintético aceitável.** É o precedente que justifica o ink-v2, que desconfia de qualquer val sintético sem holdout real, e que deve ser citado toda vez que alguém propuser "mais época" ou "mais dado sintético" como solução sem antes olhar para o real.

### 3.5 Ink-pairs-train-v3 — geração 3, em andamento, sem conclusão

**Arquitetura e dados (lidos do código, não de resultados — kernel RUNNING):** UNetLite base 16, 483k parâmetros, entrada de 3 canais (after, before, delta assinado), rótulo de 3 valores (0 = resto, 1 = conteúdo antigo, 2 = tinta nova). Treino em crops 512×512, batch 16, AMP, AdamW 3e-4, até 8 épocas, avaliação em folha cheia a cada 2 épocas, export ONNX com conferência em `onnxruntime`. Aumento de desalinhamento de 2–3 px em 50% dos crops.
**O problema que ele ataca, e por que a formulação é boa:** as v1/v2 absolutas ("achar tinta na página") não resolveram o vazamento de impresso: v1 tinha 2,56% e v2 tem 1,41%, contra uma régua de 0,5%. A v3 muda o problema para "achar só a tinta nova comparando before/after". Se o impresso já estava na folha antes, ele é conteúdo antigo (classe 1), não tinta nova (classe 2) — o vazamento deixa de ser um erro de limiar e passa a ser uma decisão de comparação. É a primeira formulação do projeto que ataca a causa do vazamento, não o sintoma.
**Ressalvas abertas, que impedem qualquer conclusão:**
- O kernel está **RUNNING** (iniciado 30/10/2026 16:28 -03). Nenhum resultado existe ainda. Qualquer número sobre a v3, neste momento, seria invenção.
- O dataset de entrada, `draw-ink-pairs-v1`, lista **0 bytes** no Kaggle contra 16.000 pares descritos no código. Até verificar o conteúdo real, não se sabe se o treino está lendo os pares certos.
- O código inclui aumento de desalinhamento de 2–3 px em 50% dos crops, mas há uma ressalva registrada de que o dataset v1 só embute desalinhamento de até 1 px em ~30% dos pares. Se o aumento do código e o desalinhamento do dataset divergirem, o modelo pode aprender a tolerar um desalinhamento que o dado não tem, ou vice-versa. Isso precisa ser conferido quando o treino terminar, comparando o aumento efetivo com a distribuição real de desalinhamento dos pares.
- A avaliação em folha cheia a cada 2 épocas é o desenho certo, mas só terá valor se a folha cheia de avaliação incluir impresso real e desalinhamento real, não só pares sintéticos limpos.

### 3.6 Cornernet — detector de cantos, complemento, sem métrica confirmada

**Arquitetura e dados (só código):** detector de cantos por heatmap, menos de 2M de parâmetros, entrada 256, heatmap 128, sigma 4,0, 4.000 cenas sintéticas 1280×720 com oclusões fortes cruzando cantos, 60 épocas, lr 2e-3, refinamento sub-pixel, exporta `cornernet_v1.onnx`.
**Por que poderia ser útil:** o quad da folha sob oclusão é ponto fraco do detector clássico, não da segmentação. Um detector de cantos treinado com oclusão cruzando cantos ataca exatamente o caso em que o contorno clássico falha — e o caso real mais claro do projeto, a caneta cruzando a borda (take t17, erro de ~110–140 px no canto, que a guarda passa porque não é nem saturado nem quadro inteiro), é um caso de canto, não de segmentação.
**Por que não se pode confiar ainda:** o erro held-out **nunca foi verificado** — só o código foi lido; os outputs estavam bloqueados pelo 429. E há um alerta vindo do upstream, não do fork: a cabeça de cantos do **sheet v4 do upstream não aprendeu** (picos 0,09–0,15 contra alvo 1,0; harness de presença 7/43 contra 43/43 do v3 com guards). Heatmap de canto é a arquitetura que mais precisa de harness real antes de qualquer confiança, porque métrica sintética de canto não transfere sozinha. O cornernet do fork deve ser julgado pelo mesmo harness que reprovou o sheet v4, não pela sua validação sintética.

### 3.7 Phase2 v4 e v3 — ideias e receita, não modelo pronto

**v4 (só código):** mesmo encoder mit-b0, decoder MLP customizado, 3 saídas (máscara 2 classes, heatmaps de 4 cantos 128×128 sigma 2,0, cabeça de presença aprendida), aumento de oclusão por cápsula (p = 0,30), consistência fotométrica (peso 0,1), ONNX que mantém `mask_logits` + `presence_logit` derivado.
**v3 (só código):** SegFormer-B0 2 classes, mistura MIDV/SmartDoc/Lucas-holdout {0, 24, 42} ×12/school/COCO-negativos, AMP FP16, batch 16, LP-FT, speed gate.
**Por que não são modelos usáveis:** ambos são **2 classes, sem hand e sem ink**. Usar v4 ou v3 como modelo seria regredir de 4 classes para 2 — perder exatamente as duas classes que o projeto mais precisa (mão e tinta). Não são fracassos; são **categorias erradas** para o problema atual.
**O aproveitamento correto, e é só ele:**
- Da v4: a cabeça de cantos (como ideia, com o alerta do sheet v4), o aumento de oclusão por cápsula (p = 0,30) e a presença aprendida (como alternativa ao `paper_presence` derivado, desde que validada — presença aprendida pode aprender o viés do sintético).
- Da v3: a receita de treino — AMP FP16 (custo), batch 16 (estabilidade), LP-FT (primeiro treinar a cabeça, depois descongelar — reduz catastrophic forgetting do pré-treino) e speed gate (critério de parada por velocidade, não só por métrica).
- A contradição da v4 (docstring "PREPARADO, NÃO LANÇADO" contra status COMPLETE) deve ser esclarecida a partir do output dela antes de qualquer uso, não assumida.

### 3.8 Pipeline clássico — referência, não concorrente neural

**O que é:** detector por contorno + guarda de luz combinada, validação sintética, sem rede neural.
**Números (bateria sintética expandida, 59 casos):** detecção 34/102 (33,3%) em cenas sintéticas; sombra 0/12, luz baixa 0/12, glare 6/12; **0 falsos positivos em 16 negativos**; ink recall 83,75% **quando detecta** (27,92% contando os misses como zero).
**Por que foi mal onde importa:** o gargalo clássico é **detecção**, não extração de tinta. A porta de textura absoluta (V > 180) barra a folha antes da luz ajudar: luz moderada 0,65, gradiente forte, sombra forte e papel cinza-180 matam a detecção. A oclusão de canto falha de forma **insegura** (erro de 131,40 px em vez de recusar) e a guarda de segurança nunca disparou em fotos reais (0 descartes em 91 frames, todo papel branco) — ou seja, a guarda protege contra casos que não aparecem no real e deixa passar o caso que aparece (caneta na borda).
**Por que esta seção está aqui:** para impedir uma comparação inválida. Comparar o IoU do SegFormer (0,9183) com o recall do clássico (83,75% quando detecta) é comparar coisas diferentes: o SegFormer resolve folha/mão/tinta por máscara, o clássico resolve folha por contorno. O clássico continua útil como fallback e como referência de robustez (0 falsos positivos é um padrão que o neural deve manter), mas não é o caminho principal — e foi por isso que o projeto investiu no SegFormer por máscara.

### 3.9 Upstream: sheet v4 e tinta v2 — veredictos da noite, modelos do upstream, não do fork

**Sheet v4 (upstream, não do fork): NÃO TROCAR.** A cabeça de heatmap de cantos não aprendeu: picos 0,09–0,15 contra alvo 1,0; harness de presença 7/43 contra 43/43 do v3 com guards; replay com desvio-padrão de 467,3 px no canto TR.x contra 65,6 px do v3+guards. O v3 com guards Block 6 fica em produção. Qualificação importante: isso **não invalida** o SegFormer por máscara do fork (val_mean_iou 0,9183 em sintético) — são arquiteturas diferentes (heatmap de canto versus máscara) — mas qualquer futura troca no fork precisaria do mesmo tipo de julgamento (harness + replay com guards), não de IoU de validação sozinho.
**Tinta v2 do upstream (não confundir com o ink-v2 do fork): NÃO TROCAR.** Em replay real, melhorou muito (tinta retida 371 → 8.915; cobertura de núcleo 16,8% → 99,3%), mas as bolhas ficaram em 54,4% (régua ≤ 25%) e 5 de 8 caixas foram zeradas, com ressalvas do próprio autor sobre defeito de rótulo e caixas de régua deslocadas. A heurística continua sendo a fonte de tinta do upstream; a tinta v3 do upstream só vem depois de consertar o juiz (recalibrar as caixas). **Lição transferível:** quando a régua tem caixas deslocadas e rótulo defeituoso, nem melhora real nem piora real são legíveis. O fork deve aplicar a mesma desconfiança às suas próprias réguas antes de celebrar qualquer ganho.
**Novo vectorize do upstream (contexto, não modelo neural):** ferramenta offline (canvas PNG → polilinhas → SVG, Zhang-Suen + grafo + Douglas-Peucker). A comparação medida entre ela e o traço ao vivo do fork mostrou: upstream rápido e preciso em traço fino (erro médio 0,425 px a 2 px, mas 34 polilinhas para 9 traços esperados, 38% abaixo de 15 px), fork estável em 9–11 traços em todas as espessuras com erro 0,109 px, mas SVG 8–18× maior por não simplificar. Veredito daquela comparação, mantido aqui: fork ao vivo como caminho principal, vectorize offline como complementar, **não unificar os dois rastreadores** — e a mesma filosofia vale para SegFormer + ink-v2: segundo estágio com portão, nunca fusão por média.

---

## 4) Padrões gerais: por que foram bem, por que foram mal

### 4.1 O que foi bem, em todos os sistemas

1. **Áreas grandes e bem delimitadas saturam rápido.** Fundo, papel e, em menor grau, mão, são aprendidos com poucos dados e mantêm precision e recall altos. O transfer learning do mit-b0 explica boa parte disso: o modelo já chega sabendo contorno e textura.
2. **Variação real ajuda mais que volume sintético.** A mão chega a 0,941 porque as mãos FreiHAND são reais; o ink-v2 melhora porque o dado de tinta fraca é mais parecido com o real; o ink-pairs-v3 é promissor porque compara antes/depois real. Em todos os casos, o ganho veio de aproximar o dado do real, não de aumentar o sintético.
3. **Regras booleanas simples e portões explícitos funcionam.** O `ink_stage2.py` (só ink dentro de paper/ink, mão e fundo intocados), o `paper_presence` derivado (fração de papel ≥ 0,01 e quad fechado) e a guarda de luz combinada são desenhos que falham de forma previsível. O oposto — média de probabilidades entre modelos com normalizações diferentes, ou presença aprendida sem validação — falha de forma imprevisível.

### 4.2 O que foi mal, em todos os sistemas

1. **Tudo que é fino, fraco ou deslocável.** Tinta de 1–3 px, canto sob oclusão, folha em sombra ou luz baixa. O padrão se repete no SegFormer (ink 0,747 estrito), no clássico (sombra 0/12, luz baixa 0/12), no cornernet (risco de heatmap que não aprende) e no sheet v4 do upstream (picos 0,09–0,15). O sintético limpo esconde o erro; o real, a sombra e a oclusão o revelam.
2. **O val sintético mente por omissão.** Ele não contém o que o real contém: caneta cruzando a borda, mão com caneta sobre a tinta, sombra real, papel amassado, texto impresso no meio da folha. O ink-v1 é a prova: val sintético aceitável, real cego (P(tinta) mediana 0,0000). Nenhum número sintético, por melhor que seja, substitui um holdout real.
3. **O gargalo muda de lugar conforme o sistema, e confundir gargalos leva a otimizar o lugar errado.** No clássico, o gargalo é detecção (33,3%), não extração (83,75% quando detecta). No SegFormer, o gargalo é precision/deslocamento da tinta, não recall (0,92) nem época (plateau de época 10 a 25). No ink-v2, o gargalo é avaliação conjunta, não arquitetura. No cornernet, o gargalo é o harness, não o heatmap em si. No projeto como um todo, o gargalo mais caro foi entrega (ONNX ausente, `metrics.json` bloqueado por 429), não modelo.
4. **Réguas defeituosas produzem veredictos defeituosos.** A tinta v2 do upstream melhorou em replay mas foi reprovada por bolhas, com caixas de régua deslocadas e rótulo defeituoso admitidos pelo próprio autor. A guarda de luz do fork nunca disparou no real porque protege contra casos que não aparecem. O `paper_presence` derivado existe justamente porque uma cabeça auxiliar de presença poderia aprender o viés do sintético. Antes de otimizar um modelo para uma régua, é preciso validar a régua — e a régua do Draw, hoje, é o holdout real que ainda não existe.

### 4.3 A curva do treino principal, lida como diagnóstico

Época 0: 0,8609. Época 10: 0,9079. Época 25: 0,9183. Época 29: 0,9149. Isso é uma curva de **saturação**, não de sub-treino: o modelo aprende rápido (transfer learning), melhora devagar e estabiliza. O train_loss continua caindo (0,3391 → 0,0815) enquanto o val estabiliza — sinal clássico de que mais época, no mesmo dado, rende cada vez menos. A conclusão para os próximos notebooks é direta: **não comprar mais época; comprar dado melhor e régua melhor**. Especificamente: tinta com espessura e cor calibradas contra traço real, oclusão real, e um holdout real que puna o que o sintético não pune.

---

## 5) O que NÃO está provado — e deve ser declarado como pendente, não como resolvido

Esta seção é tão importante quanto as métricas, porque é ela que impede decisões baseadas em números que não significam o que parecem significar.

1. **Desempenho em foto real, de nenhum modelo.** Nenhum modelo — SegFormer principal, ink-v2, cornernet, phase2 — foi avaliado em holdout real anotado. Todos os números de IoU, Dice, precision e recall deste relatório, exceto os do replay do upstream e os da bateria clássica, são sintéticos. O `draw-real-seg-pilot-v1` (30 MB, novo) pode mudar isso, mas seu conteúdo **não foi inspecionado** e não pode ser citado como holdout até que seja.
2. **Ganho conjunto SegFormer + ink-v2.** O segundo estágio está integrado e testado em lógica (8 testes), mas nunca foi avaliado, nas mesmas cenas, contra o SegFormer sozinho. Não se sabe se ele melhora, piora ou não altera o ink 0,747 — e, principalmente, não se sabe se ele melhora a tinta **real**, que é o motivo de ele existir.
3. **Qualquer métrica de cornernet e phase2.** Ambos são **só código** nesta análise. Seus outputs nunca foram lidos por causa do 429. Qualquer número atribuído a eles seria invenção.
4. **Resultados do ink-pairs-train-v3.** Kernel RUNNING. Não há resultado, e o dataset de entrada lista 0 bytes contra 16.000 pares descritos. Nada sobre a v3 pode ser concluído até o treino terminar e o dataset ser verificado.
5. **Conteúdo do `draw-ink-pairs-v1` e do `draw-training-code` re-versionado.** Ambos mudaram hoje (30/10/2026) e foram verificados só parcialmente, por metadado e primeira página de arquivos, não por conteúdo completo. O `draw-training-code` mudou de fato: antes `train.py` 24.820 bytes e `config.json` 8.172 bytes, agora `train.py` 32.937 bytes e `config.json` 10.476 bytes (30.662 bytes no total da listagem, duplicados em `payload/training/`). A descrição cita treinador 4 classes, suporte a ignore-index 255, integração de treino real e correção de device na exportação ONNX — exatamente as duas correções pendentes do treino principal. **Não reutilizar sem diff contra a versão anterior; o código novo ainda não foi validado por uma corrida completa.** O `draw-ink-pairs-v1` está populado, mas sua contagem total não foi verificada.
6. **ONNX do SegFormer principal.** Ausente. Não existe, hoje, um `draw_segmentation_v1.onnx` válido e validado contra o PyTorch. Qualquer plano que dependa de "rodar o modelo 4 classes na aplicação" depende primeiro deste artefato.
7. **Integração do modelo 4 classes na aplicação.** `draw/neural.py` continua com o contrato de 2 canais do upstream. A integração, documentada em `docs/neural-seg-v1/INTEGRATION_MAIN.md` (training-zone), é um passo futuro, não um fato presente — e só faz sentido depois de um ONNX válido, de boas métricas e da bateria de 15 cenários.
8. **A bateria de 15 cenários e o holdout real do plano original.** Ambos estavam nos requisitos do projeto (métricas por classe obrigatórias, tinta com tolerância, bateria visual de 15 cenários, holdout real separado do sintético). A tolerância foi entregue; a bateria de 15 cenários e o holdout real, não.

---

## 6) Observação final — melhorias concretas para os próximos notebooks

Observação, na forma de lista priorizada, do mais barato e mais decisivo para o mais caro. Nada aqui foi executado; tudo aqui é recomendação para os próximos notebooks, na ordem em que deve ser feito.

### 6.1 Primeiro, fechar a entrega — custo quase zero, desbloqueia tudo

1. **Exportar ONNX corretamente desde o primeiro teste, e tornar a validação fatal.** Modelo e dummy no mesmo device (o bug de uma linha do SegFormer), exportador legado (`dynamo=False`), opset 14, input `image` / output `logits`, e validação PyTorch × ONNX obrigatória (IoU por classe + diff máximo de logits) como etapa que **falha o treino** se não passar. Nunca mais um treino "completo" sem artefato. O ink-v2 acertou o device e errou por não ter sido executado localmente; o SegFormer errou o device e era não-fatal. Os dois erros se corrigem com a mesma regra: sem ONNX validado, o treino não está entregue.
2. **Ler o `metrics.json` no mesmo dia, sem burst.** O bloqueio 429 atrasou a entrega final do treino principal em dias. Próximos notebooks: uma única tentativa filtrada por dia, fora do horário de pico, e o `metrics.json` como primeiro artefato a ser lido, não o último. Se o 429 persistir, a leitura pela web UI é o caminho, não mais tentativas pela CLI.
3. **Congelar e versionar o código de treino antes de cada corrida.** O `draw-training-code` re-versionado hoje, sem verificação de diff contra o usado no treino principal, é um risco de irreprodutibilidade. Regra: cada corrida referencia um hash de dataset de código, e o `training_config.json` da corrida registra esse hash. Sem isso, comparar duas corridas é comparar dois códigos desconhecidos.

### 6.2 Segundo, consertar a régua — sem isso, todo ganho é suspeito

4. **Tinta com tolerância como métrica primária, IoU estrito como secundária.** Para traço fino de 1–3 px, o IoU estrito pune deslocamento de 1–2 px que não importa para o usuário (o modelo atual: precision 0,80 estrito contra 0,976 com tolerância de 2 px). Todo próximo notebook deve reportar as duas, e o critério de "melhor época" deve considerar a tolerância, não só o val_mean_iou.
5. **Holdout real anotado antes do próximo retreino.** Sessão inteira como holdout, nunca split por frame, com as regras já definidas (mão sobre tinta = mão; caneta = fundo; texto impresso ≠ tinta). Tamanho mínimo viável: 60–100 frames estratificados (~1 frame a cada 2 s + todos os frames de evento), teto 150–200. Ferramenta: LabelMe ou CVAT, máscara PNG de 1 canal com valores {0, 1, 2, 3}. **Prioridade absoluta de inspeção:** abrir o `draw-real-seg-pilot-v1` (30 MB) e verificar se ele já é, ou pode virar, esse holdout. Se for, o próximo retreino já nasce com régua; se não for, o holdout precisa ser construído antes de qualquer retreino.
6. **Bateria de 15 cenários, executada e publicada.** Ela estava nos requisitos originais e nunca foi entregue. Os 15 cenários devem incluir, obrigatoriamente: sombra, luz baixa, glare, papel amassado, mão com caneta sobre a tinta, caneta cruzando a borda (o caso t17), folha fora do quadro, folha minúscula, texto impresso no meio da folha, e os negativos (mesa vazia, mão sem folha, objeto branco). Sem esses cenários, a próxima validação sintética vai repetir o erro do ink-v1: parecer boa e falhar no real.
7. **Validar a régua antes de otimizar para ela.** Recalibrar as caixas de avaliação (lição da tinta v2 do upstream), verificar o desalinhamento real dos pares do ink-pairs-v1 contra o aumento do código da v3, e reproduzir, para o cornernet, o harness que reprovou o sheet v4 (presença + replay com guards). Uma régua deslocada reprova modelo bom e aprova modelo ruim.

### 6.3 Terceiro, atacar a tinta — precision, não recall

8. **Atacar a precision da tinta, não o recall.** O modelo já acha 92% da tinta (recall 0,9198 estrito, 0,9998 com tolerância); o problema são os 20% de falsos positivos e o deslocamento. Candidatos, em ordem de custo: (a) perda com peso por borda ou por distância, que pune deslocamento mais que ausência; (b) rótulo de tinta com espessura calibrada contra traço real, não contra traço sintético idealizado; (c) o segundo estágio ink-v2, **mas só após avaliação conjunta** nas mesmas cenas — sem ela, o estágio é hipótese, não melhoria.
9. **Avaliar o ink-v2 em conjunto, nas mesmas cenas, antes de qualquer decisão sobre ele.** Protocolo mínimo: mesmas cenas sintéticas de validação + o holdout real (quando existir), SegFormer sozinho versus SegFormer + estágio, reportando ink estrito, ink com tolerância de 2 px e vazamento de impresso. Se o estágio melhorar a tolerância sem piorar o vazamento, ele fica ligado por padrão; se piorar o vazamento, ele fica como opção; se não alterar nada, ele é removido para não carregar complexidade morta.
10. **Decidir o ink-pairs-v3 com o mesmo protocolo, quando ele terminar.** A formulação before/after é a mais promissora do projeto para o vazamento de impresso, mas promessa não é resultado. Quando o treino terminar: verificar o conteúdo real do `draw-ink-pairs-v1`, comparar o desalinhamento efetivo com o aumento do código, avaliar em folha cheia com impresso real, e só então comparar o vazamento dele (meta: abaixo de 0,5%, contra 1,41% da v2 e 2,56% da v1) com o da v2 nas mesmas folhas. Se a v3 vencer, ela substitui a v2 como segundo estágio; se empatar, a v2 fica por ser mais simples; se perder, a v3 é arquivada como experimento de formulação, não como modelo.

### 6.4 Quarto, o próximo retreino — um só, com tudo que já se aprendeu

11. **Se houver próximo retreino, que seja um único treino 4 classes, não três modelos separados.** Incorporar, num só treino: a receita da v3 (AMP FP16, batch 16, LP-FT — custo menor, estabilidade maior, menos forgetting do pré-treino), o aumento de oclusão da v4 (cápsula p = 0,30 — robustez ao caso real de oclusão), e a tinta cinza fraca + degradação real do ink-v2 (o regime que faltava ao treino principal). Early stopping sobre val_mean_iou **e** ink com tolerância, não só val_mean_iou — porque um modelo pode melhorar a média piorando a tinta, que é a classe que importa.
12. **Não treinar em dado real antes da decisão de rota.** As rotas possíveis para as sessões reais (holdout inteiro versus treino inteiro após anotação manual) ainda aguardam decisão do usuário. Treinar em real sem essa decisão, ou com anotação parcial, contaminaria o holdout e destruiria a única régua confiável que o projeto pode ter. A decisão de rota é pré-requisito do retreino, não detalhe.
13. **Cornernet só com harness, e só como complemento.** Antes de qualquer uso: reproduzir o harness do sheet v4 (presença + replay com guards) sobre o cornernet; se ele passar, integrá-lo como ajuda ao quad sob oclusão, nunca como substituto da máscara; se falhar, arquivá-lo. Não investir época em heatmap de canto sem harness — o sheet v4 mostrou que heatmap que não aprende não se conserta com mais época, e sim com formulação e rótulo.

### 6.5 Quinto, o que não fazer — tão importante quanto o que fazer

14. **Não unificar rastreadores nem modelos por pressa.** A comparação medida já mostrou que o traço ao vivo do fork e o vectorize offline do upstream têm filosofias opostas e ambas válidas em seu uso; o mesmo vale para SegFormer (máscara) + ink-v2 (especialista de página) e para SegFormer + cornernet (máscara + cantos). Segundo estágio com portão de mão, nunca média de probabilidades entre modelos com normalizações e resoluções diferentes; complemento de cantos, nunca substituto de máscara.
15. **Não abrir PR para o upstream com nada disto.** Por regra do usuário, só itens não-CAD da main, prontos e um por tema, vão para o santleme/draw. Treino, neural, capturas, tinta, cantos e análise de notebooks ficam no fork, na training-zone. Este relatório, quando publicado, vai para a main do fork (nome a ser dado pelo usuário), não para o upstream.
16. **Não confundir as duas "tintas v2".** A tinta v2 do upstream (reprovada por bolhas, com régua defeituosa) e o ink-v2 do fork (especialista integrado à training-zone) são modelos, avaliações e veredictos diferentes. Qualquer documento futuro que cite "tinta v2" sem qualificar qual das duas está criando uma ambiguidade que já causou confusão neste projeto.
17. **Não celebrar número sintético sem holdout real.** Esta é a regra que resume todas as outras. Enquanto não existir um holdout real anotado e avaliado, todo relatório de "melhora" deve vir com a ressalva explícita: "em sintético". O dia em que o holdout real existir, ele vira a régua principal — e é provável que ele reordene todas as conclusões deste relatório. Isso não é pessimismo; é o que o ink-v1 já provou uma vez.

---

## Apêndice: fontes e rastreabilidade

- Treino principal: kernel `guzzbr/notebooka37f38640e`, Version #1, GPU T4 x2, 11 inputs, COMPLETE. Métricas: `hidden_files/kaggle-full-training/metrics.json` (lido diretamente). Checkpoint: `checkpoints/best.pt`, 14.932.509 bytes, época 25. ONNX: ausente, `onnx_validation.status = failed`.
- Ink-v2: kernel `guzzbr/draw-ink-train-v2`, COMPLETE 30/10/2026 02:09 (-03). Integração: training-zone `5e71a0c` (`draw/ink_stage2.py`, `docs/ink-v2.md`, `models/draw_ink_v2.onnx` sha256 `5160cc4649ca5c477dafde0cb2c642ff87f52c463bf1717ffa712c8a1c400314`, 8 testes passed, selftest 96,03%). Branch `feature/ink-v2` apagada após o merge; conteúdo preservado na training-zone.
- Ink-pairs-v3: kernel `guzzbr/draw-ink-pairs-train-v3`, RUNNING desde 30/10/2026 16:28 (-03). Código em `/tmp/analise-recente/draw-ink-pairs-train-v3/`. Sem resultados.
- Cornernet: kernel `guzzbr/draw-cornernet-train`, COMPLETE 28/10/2026 17:47 (-03). Só código lido; métricas não verificadas.
- Phase2 v4/v3: kernels `guzzbr/draw-phase2-train-v4` (COMPLETE 30/10/2026 01:53 -03) e `guzzbr/draw-phase2-train-v3` (COMPLETE 29/10/2026 17:26 -03). Só código lido; métricas não verificadas.
- Clássico: bateria sintética expandida (59 casos) e avaliação sintética de 122 cenas (seed 42), relatórios de 29/10/2026.
- Upstream: `docs/noite/2026-10-30.md` (sheet v4, tinta v2, vectorize) e comparação medida fork ao vivo × vectorize offline (canvas sintético 850×1200, espessuras 1–3 px).
- Datasets: listagem Kaggle GUZZ BR por CLI em 30/10/2026 (18 datasets) + metadados e manifestos pequenos baixados. `draw-real-seg-pilot-v1` inspecionado por manifesto (não por imagem individual): 719 frames reais, pseudo-rótulo não verificado, não é ground truth. `draw-ink-pairs-v1`: primeira página de arquivos confirma que está populado, apesar do 0 bytes na listagem; contagem total não verificada.
- Estado dos branches no momento da análise: main do fork `b4a77b9`, training-zone `5e71a0c` (com ink-v2 integrado). Nada desta análise altera esses branches.
