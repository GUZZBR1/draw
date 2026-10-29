# Treino completo no Kaggle — pronto para rodar (quando o dataset grande estiver pronto)

> Status: **preparação concluída, treino NÃO executado.**
> Este documento deixa o treino completo pronto para apertar o botão.
> Nada aqui foi aplicado ao código: os ajustes recomendados estão listados
> como decisão, não como mudança feita. O único arquivo criado por esta
> preparação é este `TRAINING_READY.md`.
>
> Base verificada: branch `feature/neural-seg-v1`, notebook offline
> `cc59919` (`notebooks/train_segformer_kaggle.ipynb`), `training/train.py`
> e `training/config.json` lidos diretamente (não por memória de relatórios).

## 1. O que já está pronto (verificado contra o código real)

| Item | Estado |
|---|---|
| Notebook offline (15 células) | Sem `pip install`, sem `git clone`. Código vem como input do kernel. |
| Localização do código | Procura `training/` em `/kaggle/input/*/training`, `/kaggle/input/*/draw/training`, `/kaggle/working/draw/training`; erro claro se não achar. |
| Localização do dataset | Procura `*/dataset/synthetic` e `*/synthetic`; **trava** se não achar ou se algum split tiver 0 imagens. Layouts aceitos: `images/masks` e plano do gerador. |
| Checagem do FreiHAND | Verifica `training/mask/` de verdade, limiar 127, cobertura da máscara; **para** se faltar ou se a cobertura for 0. |
| Célula de config/pixels | `load_config` valida 4 classes e 4 pesos; `pair_images_masks` aceita os dois layouts; `pixel_counts` e `compute_class_weights` funcionam. |
| Célula de treino | `EPOCHS = None` usa a config (30 épocas). Para teste, trocar só `EPOCHS = 2`, sem editar o comando. Comando chama `train.py --config … --data-dir … --out-dir /kaggle/working/out`, argumentos que existem no `argparse` do `train.py`. |
| Célula de métricas | Lê `metrics.json` e espera chaves que o `train.py` realmente grava: `mean_iou`, `background/paper/hand/ink` com `iou/dice/precision/recall`, e `onnx_validation` (via `.get`). |
| Dataset de código no Kaggle | `guzzfort/draw-training-code` **v2**, pronto, com `training/train.py` + `training/config.json` em `/kaggle/input/draw-training-code/training/`. Atenção: a v1 subiu com os arquivos soltos na raiz — **usar a v2**. |
| Dataset de teste no Kaggle | `guzzfort/draw-synthetic-test`, pronto, 36 cenas (train 24 / val 6 / test 6), mas com mãos placeholder — serve para validar o fluxo, **não** para o treino final. |
| FreiHAND público | `danieldelro/freihand` acessível como input (3,9 GB, máscara real 224×224 só no treino, limiar 127, CC BY-NC 4.0, só pesquisa). Serve só para a checagem do notebook; o treino usa o dataset sintético, que já incorpora as mãos. |

## 2. Como o treino completo roda (caminho feliz)

1. Kernel com os inputs da seção 5 anexados (dataset grande + código v2 + FreiHAND).
2. Célula de ambiente: imprime versões de `torch/transformers/cv2/numpy` e se há CUDA. `albumentations` ausente = treino sem augmentation (avisado, não fatal). `onnxruntime` ausente = validação ONNX pulada (avisado, não fatal). **`onnx` ausente = exportação ONNX falha** — o notebook o marca como opcional, mas `torch.onnx.export` precisa dele; ver seção 3.
3. Células de localização, dataset e FreiHAND passam nas travas.
4. Célula de config: compara pixels reais do treino com os pesos da config.
5. Célula de treino com `EPOCHS = None`: `train.py` roda 30 épocas, batch 8 (config), AdamW lr 6e-5, early stopping patience 5 em `val_mean_iou`, salva `checkpoints/last.pt` a cada época e `checkpoints/best.pt` no melhor `val_mean_iou`.
6. Ao final, o melhor checkpoint é exportado para `draw_segmentation_v1.onnx` (input `image` `[1,3,512,512]`, output `logits` `[1,4,512,512]`, opset 14) e são gravados `classes.json`, `metrics.json`, `training_config.json` e `split_manifest.json` (cópia do manifesto do gerador, se ele existir no dataset — ver seção 4, item 3).
7. Célula de métricas lê `metrics.json`. Depois disso, **fora do notebook**: rodar a bateria de 15 cenários com `tools/seg_metrics.py` e avaliar o holdout real separadamente (ver seção 7).

## 3. Ajustes recomendados para o treino completo (NÃO aplicados — decidir antes de rodar)

| # | Item | Código real hoje | Recomendação para o treino completo |
|---|---|---|---|
| 1 | **Batch / OOM em T4 (fp32)** | Config: batch 8, `mixed_precision: bf16`. Código: **sem `autocast`**, treino em fp32; `gradient_accumulation_steps` (config) **nunca é lido** e o `optimizer.step()` acontece a cada lote. | **Rodar o treino completo com batch 4** (`--batch-size 4` na célula de treino, sem tocar na config). Batch 4 reduz o batch efetivo (não há acumulação para compensar), mas é a mitigação que funciona hoje. Nunca reduzir a resolução (traço fino depende de 512×512). Se quiser batch efetivo 8 com batch 4, é preciso corrigir o `train.py` para acumular gradiente — fora desta preparação. |
| 2 | **Warmup** | Config: `cosine_with_warmup`, `warmup_ratio 0.1`. Código: `CosineAnnealingLR` com `T_max = total - warmup`, **sem rampa de LR** — o treino começa direto em 6e-5. | Aceitar explicitamente que o treino **não tem warmup de fato**, ou corrigir o scheduler antes do treino completo. Não prometer "com warmup" no relatório final. |
| 3 | **Precisão mista** | Config promete bf16; código não usa `autocast`/`GradScaler`/`half()`. | Aceitar fp32 (mais lento, mais memória) ou corrigir o `train.py` para bf16 antes do treino completo. O risco de OOM da linha 1 vem daqui. |
| 4 | **Pesos da loss** | Config: CE `[1.0, 2.0, 3.0, 10.0]` + 0,5×Dice. Código registra pixels reais e pesos calculados, mas **usa os da config**. | Na célula de config do dataset grande, comparar: se o peso calculado para `hand` divergir muito de 3,0 (no dataset de teste pequeno ele deu 6,11, esperado num dataset minúsculo), recalibrar e registrar a mudança em `training_config.json`. **Nunca** ajustar peso olhando o split de teste. |
| 5 | **Retomada** | `last.pt`/`best.pt` são salvos, mas **não há `--resume` nem leitura de checkpoint** no início. | Verificar o limite de sessão vigente no Kaggle **antes** de rodar. Se a sessão cair no meio das 30 épocas, o treino recomeça do zero. Mitigação sem código: preferir um dataset que caiba com folga no limite; mitigação com código: implementar `--resume` (fora desta preparação). |
| 6 | **Validação ONNX sem limiar** | Config promete que degradação "aborta a entrega"; código só grava `max_abs_diff` (ou `None` sem `onnxruntime`), sem comparar com limiar. | Usar os limiares que já existem em `tools/validate_onnx.py` como critério manual na leitura do `metrics.json`: `max_abs_diff > 1e-3` ou `min_argmax_iou < 0.99` = degradado, entrega **não** validada. Não chamar o ONNX de validado se `onnx_validation` vier `None`. |
| 7 | **Bateria de 15 cenários** | O notebook promete cenários com predições salvas, mas nem o notebook nem o `train.py` a executam; o `train.py` só grava `ink_tolerance_sample` do primeiro lote de validação. | Após o treino, rodar `tools/seg_metrics.py` (cenários fixos) à parte e anexar as predições ao relatório. Sem isso, a entrega está incompleta segundo o escopo. |
| 8 | **`onnx` obrigatório para a entrega** | Notebook marca `onnx` como opcional; exportação falha sem ele. | Antes de rodar, confirmar que o ambiente do kernel traz `onnx` (a célula de ambiente já o lista). Se faltar, a entrega sai sem ONNX — não é "só um aviso". |
| 9 | **Pesos `nvidia/mit-b0`** | Ver seção 6 — é o único ponto que ainda pode exigir internet. | Resolver com a opção A da seção 6 antes de rodar 100% offline. |

## 4. Plano de upload do dataset grande para o Kaggle

Estrutura que o kernel espera (dataset **privado**):

```
<dataset>/
  dataset-metadata.json        # id, title, licenses
  synthetic/
    train/
      images/<cena>.png
      masks/<cena>.png          # 1 canal, valores 0..3
      ink_full/<cena>.png       # opcional, fora do treino
    val/
      images/  masks/  ink_full/
    test/
      images/  masks/  ink_full/
  split_manifest.json           # OBRIGATÓRIO no nível do dataset (ver item 3)
```

Passos, em ordem:

1. **Gerar com mãos FreiHAND reais** (não placeholder), com `tools/gen_synthetic.py`, seed registrada (config usa seed 42). Split **por fonte original** (mão base, folha base, fundo) 80/10/10, nunca por arquivo: o FreiHAND tem 32.560 mãos únicas × 4 variantes de fundo — dividir por arquivo vazaria a mesma mão entre treino e teste.
2. **Converter o layout plano do gerador** (`<split>/scene_*.png`, `scene_*_mask.png`, `scene_*_ink_full.png`) para `images/masks/ink_full` por split. O `train.py` aceita os dois layouts, mas o layout `images/masks` é o que o notebook conta, o que a config documenta e o que evita ambiguidade no upload. (Conversão é só mover/renomear arquivos; máscaras nunca são reamostradas com interpolação linear — `INTER_NEAREST` se houver qualquer resize.)
3. **Copiar `manifest.json` do gerador para `split_manifest.json` na raiz do dataset.** O `train.py` procura `data_dir/split_manifest.json` (não `manifest.json`); sem ele, ele grava um placeholder e a entrega perde a rastreabilidade de fontes/seeds. O manifesto deve conter: seed, IDs das fontes por split, licença do FreiHAND e a citação (Zimmermann et al., ICCV 2019).
4. **Subir com `kaggle datasets create -p <dir> --dir-mode zip`.** Aprendizado das subidas anteriores: o modo zip **descarta um nível de pasta** — a v1 do dataset de código perdeu o prefixo `training/` e precisou de uma v2 com pasta envoltória extra. **Depois de subir, verificar com `kaggle datasets files <slug>`** se os caminhos aparecem como `synthetic/train/images/…` (e não `dataset/synthetic/…` nem arquivos soltos na raiz). Se a estrutura vier errada, subir uma nova versão com a pasta envoltória corrigida — não tentar consertar no kernel.
5. **Confirmar `status: ready`** antes de anexar ao kernel. Dataset em processamento não aparece como input válido.
6. **Tamanho (estimativa, medida local):** ~0,55 MB por cena (RGB + máscara + ink_full, medido num lote de 6 cenas = 3,3 MB). Ou seja: ~2.000 cenas ≈ 1,1 GB; ~5.000 ≈ 2,8 GB; ~8.000 ≈ 4,4 GB; ~10.000 ≈ 5,5 GB. Upload de vários GB via CLI é lento e pode falhar no meio — subir em uma única operação, com conexão estável, e verificar a contagem de arquivos depois (o dataset de teste tem 108 arquivos para 36 cenas; o grande deve ter 3 arquivos por cena + manifesto).

## 5. Kernel do treino completo (o que anexar e como configurar)

| Input | Slug / caminho no kernel | Para quê |
|---|---|---|
| Dataset sintético grande | `<slug-do-dataset-grande>` → `/kaggle/input/<slug>/synthetic/` (ou `dataset/synthetic`, o notebook procura os dois) | Treino/validação/teste |
| Código de treino | `guzzfort/draw-training-code` **v2** → `/kaggle/input/draw-training-code/training/` | `train.py` + `config.json` (o notebook NÃO clona o repo) |
| FreiHAND | `danieldelro/freihand` → `/kaggle/input/freihand/training/mask/` | Só a checagem da célula 7 (o treino não lê o FreiHAND direto) |
| Pesos mit-b0 (opcional, seção 6) | Dataset de cache, se adotada a opção A | Rodar 100% offline |

Configuração do kernel:

- **Notebook:** `notebooks/train_segformer_kaggle.ipynb` da branch `feature/neural-seg-v1` (versão offline, commit `cc59919` ou posterior com as correções da seção 3, se aplicadas).
- **`EPOCHS = None`** na célula de treino (usa as 30 épocas da config). **Não** deixar `EPOCHS = 2` do teste.
- **Batch:** recomendado **4** para o treino completo (item 1 da seção 3). Para usar batch 4, a célula de treino precisa passar `--batch-size 4` — hoje ela só passa `--epochs` quando `EPOCHS` não é `None`; ou seja, **para batch 4 é preciso editar a célula de treino** (adicionar `--batch-size 4` ao comando) ou aceitar batch 8 da config com o risco de OOM documentado. Decidir isso antes de rodar, não no meio.
- **Acelerador:** GPU (T4 ou P100) confirmada **na interface web do Kaggle**, não só no metadata do push via CLI — na rodada de teste anterior, `enable_gpu: true` via CLI não surtiu efeito e o kernel rodou em CPU (`torch 2.11.0+cpu`).
- **Internet:** necessária só se a opção de pesos for a B da seção 6; com a opção A, internet pode ficar desligada.
- **Outputs a baixar ao final:** `/kaggle/working/out/` inteiro, no mínimo `draw_segmentation_v1.onnx`, `classes.json`, `metrics.json`, `training_config.json`, `split_manifest.json`, `checkpoints/best.pt` (e `last.pt` se houver retomada futura). **Não** commitar o `.onnx` no git por padrão — o contrato de `draw/neural.py` já prevê o binário fora do repo via `DRAW_NEURAL_MODEL`.

## 6. Pesos `nvidia/mit-b0` — o ponto pendente, resolvido (3 opções)

Fato: `training/train.py` carrega os pesos com
`SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", …)`,
sem `local_files_only` e sem cache embutido. Na primeira execução sem
cache, ele baixa do HuggingFace. A célula 2 do notebook **só avisa** se o
cache não for encontrado — ela não trava e não configura o cache. Por isso,
"100% offline" exige um dos caminhos abaixo, decidido **antes** de rodar.

### Opção A (recomendada para 100% offline) — cache como input do kernel

1. Numa máquina com internet, baixar o cache com o próprio `transformers`
   (ex.: `python -c "from transformers import SegformerForSemanticSegmentation; SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b0')"`), ou com `huggingface_hub.snapshot_download("nvidia/mit-b0")`.
2. Subir como dataset **privado** no Kaggle o diretório de cache, preservando a estrutura do HuggingFace:
   `hub/models--nvidia--mit-b0/snapshots/<hash>/…` (config, pesos e tokenizer, ~15 MB no total — pequeno perto do dataset).
3. No kernel, anexar esse dataset e, **na célula 2 do notebook, antes de qualquer `from_pretrained`**, definir:
   ```python
   import os
   os.environ["HF_HUB_CACHE"] = "/kaggle/input/<slug-do-cache>/hub"
   os.environ["HF_HUB_OFFLINE"] = "1"
   ```
   (O `train.py` importa `transformers` de forma lazy, dentro de `build_model`, então definir as variáveis na célula 2 funciona — desde que seja antes da célula de treino. Não basta o aviso atual da célula 2; é preciso **definir** as variáveis.)
4. Verificar na célula 2 que o caminho `hub/models--nvidia--mit-b0` existe no input — se não existir, o treino falhará no `from_pretrained` com erro de cache, e não com erro de rede confuso.
5. Atenção: o dataset de cache precisa ser o cache **completo e íntegro** (o `snapshot_download` garante isso); um único arquivo `pytorch_model.bin` solto não basta para o `from_pretrained` sem ajustes.

### Opção B (mais simples, se a web UI permitir) — internet só para os pesos

Confirmar internet ligada **na interface web** do kernel, rodar o treino, e deixar o `from_pretrained` baixar os ~15 MB uma vez. Todo o resto (dependências, código, dataset, checagem) já é offline. Risco: se a internet falhar no meio, o treino falha antes de começar (no `build_model`), sem perda de treino — é um erro barato, mas exige nova rodada.

### Opção C (não recomendada) — primeira corrida para cachear

Rodar um kernel com internet só para gerar o cache e salvar o cache como output/dataset, depois rodar o treino offline com a opção A. É a opção A com um passo extra; só vale se não houver nenhuma máquina com internet para baixar o cache fora do Kaggle.

**Decisão registrada nesta preparação:** a opção A é a recomendada para o treino completo, porque elimina a única dependência de rede restante e torna a rodada reproduzível. A opção B é aceitável se a internet da web UI for confirmada. O que **não** pode acontecer é rodar o treino completo assumindo offline sem ter feito A nem confirmado B — nesse caso, o erro aparecerá só no `build_model`, depois de todas as células de preparação passarem.

## 7. Estimativa de tempo em T4 (com as ressalvas)

Âncora documentada no runbook: **30 épocas, 5–10 mil imagens, batch 8 → 3–6 h em T4**.
A estimativa abaixo escala linearmente a partir dessa âncora (passos por época =
imagens de treino / batch; avaliação de validação a cada época soma ~20–30%).

| Imagens de treino (80% do dataset) | Batch 8 (config) | Batch 4 (recomendado) |
|---|---|---|
| ~1.600 (dataset de 2.000) | ~1–2 h | ~2–3 h |
| ~4.000 (dataset de 5.000) | ~3–4 h | ~5–7 h |
| ~6.400 (dataset de 8.000) | ~4–5 h | ~7–9 h |
| ~8.000 (dataset de 10.000) | ~5–6 h | ~9–12 h |

Ressalvas obrigatórias:

- São **estimativas, não garantias**: dependem da GPU sorteada (T4 vs P100), da fração de imagens com mão (mais pixels de mão não muda o custo por passo, mas o custo é dominado pela resolução 512×512, não pelo conteúdo), e do tempo de avaliação por época.
- Early stopping (patience 5) pode encurtar a corrida se o `val_mean_iou` parar de melhorar — o tempo acima é o teto de 30 épocas.
- **Sem retomada** (item 5 da seção 3): se a sessão cair, o tempo é perdido. Verificar o limite de sessão vigente no Kaggle antes de rodar um dataset de 8–10 mil imagens com batch 4.
- CPU não é opção para o treino completo: o próprio `train.py` avisa que é inviável. O teste pequeno (36 cenas, 2 épocas) roda em CPU; o completo, não.

## 8. Checklist pronto-para-rodar (todos os itens antes de apertar o botão)

- [ ] Dataset grande gerado com **mãos FreiHAND reais** (não placeholder), seed registrada.
- [ ] Split por fonte original 80/10/10, sem vazamento de mão base entre splits; `split_manifest.json` real na raiz do dataset (não placeholder).
- [ ] Layout convertido para `images/masks/ink_full` por split; contagens conferidas (3 arquivos por cena + manifesto).
- [ ] Dataset grande subido como **privado**, `status: ready`, e estrutura conferida com `kaggle datasets files` (cuidado com o nível de pasta do modo zip).
- [ ] Dataset de código **v2** (`guzzfort/draw-training-code`) anexado; caminho `/kaggle/input/draw-training-code/training/` confirmado.
- [ ] FreiHAND (`danieldelro/freihand`) anexado para a checagem.
- [ ] Pesos mit-b0 resolvidos: **opção A** (cache como input + variáveis de ambiente na célula 2) ou **opção B** (internet confirmada na web UI).
- [ ] Notebook na versão offline (`cc59919` ou posterior), com `EPOCHS = None` e decisão de batch tomada (4 com edição da célula, ou 8 da config aceitando o risco de OOM).
- [ ] Pesos da loss comparados na célula de config (item 4 da seção 3) — decidir recalibrar ou manter `[1, 2, 3, 10]`.
- [ ] GPU confirmada **na web UI** (T4/P100); limite de sessão verificado contra a estimativa da seção 7.
- [ ] Critério de validação ONNX definido antes de rodar (item 6 da seção 3): `max_abs_diff > 1e-3` ou `min_argmax_iou < 0.99` = degradado.
- [ ] Plano pós-treino: rodar a bateria de 15 cenários (`tools/seg_metrics.py`) e avaliar o holdout real do Codex **separadamente** — métricas sintéticas e reais nunca misturadas no mesmo número.

## 9. O que falta (e só isso falta) para apertar o botão

1. **O dataset grande pronto** — gerar com FreiHAND real, converter o layout, copiar o manifesto real, subir e verificar a estrutura. É o único artefato grande que ainda não existe.
2. **O cache dos pesos** (opção A) ou a confirmação de internet na web UI (opção B).
3. **A decisão de batch** (4 com edição da célula, ou 8 aceitando o risco) e, se for o caso, a recalibração dos pesos da loss após a contagem de pixels do dataset grande.
4. **A confirmação de GPU e do limite de sessão na web UI.**

Depois disso, a sequência é: anexar os 4 inputs (seção 5), rodar as células em ordem, baixar `/kaggle/working/out/`, conferir `metrics.json` por classe + `onnx_validation`, rodar os 15 cenários, e só então avaliar o holdout real. **O treino não pode ser dado como bem-sucedido só com métricas sintéticas.**

## 10. Riscos que esta preparação não elimina

- OOM em T4 com batch 8 fp32 (mitigado com batch 4, não eliminado sem correção de código).
- Queda de sessão sem retomada (mitigado verificando o limite, não eliminado sem `--resume`).
- `onnx` ou `onnxruntime` ausentes no ambiente do kernel (a célula de ambiente avisa; `onnx` ausente bloqueia a entrega, `onnxruntime` ausente só pula a validação).
- Upload de dataset de vários GB falhando no meio (verificar contagem de arquivos após o upload).
- Licença: o modelo treinado com FreiHAND é **só pesquisa, uso comercial proibido** (CC BY-NC 4.0) — se o Draw virar produto comercial, isso precisa ser resolvido antes, trocando a fonte ou obtendo autorização. Composições com mãos recortadas continuam sendo derivadas, com a mesma licença e atribuição no manifesto.
