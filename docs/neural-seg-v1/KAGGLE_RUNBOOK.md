# Roteiro de execução do treino no Kaggle — segmentação v1 do Draw

> Roteiro de execução, não é treino e não é integração.
> Nenhum treino foi executado para escrever este documento.
> Escopo do Muse aqui: dados, geração sintética, treinamento,
> validação, exportação ONNX e documentação. Fora do escopo:
> webcam, integração no app, tracking temporal, memória dos traços,
> oclusão em tempo real, interface, CAD e simuladores (Codex, depois).

## 0. Estado no momento em que este roteiro foi escrito

- Branch: `feature/neural-seg-v1`, base `e2c866f`.
- Notebook `notebooks/train_segformer_kaggle.ipynb`: **ainda não existe**
  (o diretório `notebooks/` estava vazio). Este roteiro descreve a
  estrutura esperada e o que verificar quando ele aparecer — ver
  seção 8. Não invente que o notebook já está pronto.
- Código de treino já presente (outro agente): `training/train.py`
  e `training/config.json`. O roteiro abaixo está fundamentado neles
  e em `docs/neural-seg-v1/INTEGRATION.md`. Se o notebook divergir
  deles, o notebook está errado, não o roteiro.
- Contrato do modelo (não mudar na hora de rodar):
  SegFormer-B0, transfer learning (`nvidia/mit-b0`), entrada RGB
  512×512, normalização ImageNet (média `[0.485, 0.456, 0.406]`,
  desvio `[0.229, 0.224, 0.225]`), 4 classes fixas
  `0=background, 1=paper, 2=hand, 3=ink`.
  Regra de rótulo, sem exceção: o que a câmera vê; mão por cima
  de traço = `hand`, nunca `ink`. A `ink_full_mask` é artefato
  separado, nunca classe de treino e nunca entrada.

## 1. Preparar o dataset antes de abrir o Kaggle

1. Gere o dataset sintético com `tools/gen_synthetic.py` (outro
   agente), com split **por fonte** já feito pelo gerador:
   fontes originais (mãos, folhas, fundos) divididas antes de gerar
   variações, 80% treino / 10% validação / 10% teste sintético,
   seed 42, registradas em `split_manifest.json`.
2. Estrutura esperada dentro do dataset (uma pasta por split):

```text
dataset/synthetic/
  train/images/*.png
  train/masks/*.png        # mesmo nome base da imagem, 1 canal, valores 0..3
  train/ink_full/*.png     # opcional, fora do treino
  val/images/*.png
  val/masks/*.png
  val/ink_full/*.png       # opcional
  test/images/*.png
  test/masks/*.png
  split_manifest.json
```

3. Valide localmente, antes de subir: nenhuma máscara com valor
   fora de 0..3 (qualquer outro valor aborta o treino, por
   decisão da config); todo par imagem/máscara com o mesmo nome
   base; splits sem vazamento de fonte (a mesma mão/folha/fundo
   não aparece em dois splits).
4. Não suba o FreiHAND inteiro como dataset de treino. O FreiHAND
   (`danieldelro/freihand`, Kaggle) é matéria-prima das mãos no
   gerador, não entrada direta do treino. Se o notebook precisar
   checar o formato do FreiHAND, anexe-o como segundo input
   somente para essa checagem — nunca para gerar variações novas
   dentro do treino, porque isso quebraria o split por fonte.
5. Não inclua frames reais do Draw no dataset de treino. Eles são
   holdout externo, fornecidos pelo Codex, manualmente anotados,
   separados, sem augmentation. Se ainda não foram fornecidos,
   rode o treino sem eles e deixe a avaliação real para depois —
   não improvise holdout com dados sintéticos.

## 2. Subir o dataset para o Kaggle

1. No Kaggle, crie um dataset **privado** (não público: o sintético
   pode conter mãos do FreiHAND e composições do projeto).
2. Suba a pasta `dataset/synthetic/` completa, incluindo
   `split_manifest.json`. Guarde o slug do dataset, por exemplo
   `<usuario>/draw-synthetic-v1` — o slug exato vai aparecer
   no caminho de input da seção 4.
3. Opcional, só para checagem de formato: anexe também o dataset
   público `danieldelro/freihand` ao notebook, sem usá-lo no treino.

## 3. Criar e configurar o notebook

1. No Kaggle, crie um notebook novo e suba o arquivo
   `notebooks/train_segformer_kaggle.ipynb` do branch
   (quando ele existir; ver seção 8). Não reescreva o notebook
   na interface do Kaggle: qualquer correção volta para o branch.
2. Anexe o dataset privado da seção 2 como input do notebook
   (Add Data → seu dataset). Confirme que ele aparece em
   `/kaggle/input/<slug-do-dataset>/`.
3. Acelerador: **GPU T4 ou P100** (config: `GPU T4 ou P100`,
   Kaggle gratuito). Não rode em CPU: SegFormer-B0 em 512×512
   com batch 8 não é viável em CPU dentro de uma sessão.
4. Internet: **ligada**, porque o notebook baixa os pesos
   pré-treinados do SegFormer (`nvidia/mit-b0`, Hugging Face) na
   primeira execução. Se o download falhar, o erro é de rede/
   cache, não do modelo — não troque a arquitetura por isso.
5. Antes de treinar, rode a primeira célula do notebook e
   confirme: listagem de `/kaggle/input/` com o dataset esperado,
   GPU visível (`nvidia-smi` ou equivalente no notebook), e
   `training/config.json` lido sem erro. Config é a única fonte
   de hiperparâmetros; o notebook apenas a lê.

## 4. Caminhos e comando esperado

Caminhos no Kaggle (o comando abaixo é o documentado em
`training/train.py`):

```text
dados de entrada:  /kaggle/input/<slug-do-dataset>/dataset/synthetic
config:            training/config.json   (dentro do notebook/repo)
saída:             /kaggle/working/out
```

Comando esperado dentro do notebook (ou célula equivalente):

```bash
python training/train.py \
  --data-dir /kaggle/input/<slug-do-dataset>/dataset/synthetic \
  --config training/config.json \
  --out-dir /kaggle/working/out
```

Hiperparâmetros da config (não alterar na sessão sem registrar):

```text
modelo:     SegFormer-B0, pré-treinado, cabeça nova com 4 classes
épocas:     30, com early stopping (patience 5, monitor val_mean_iou)
batch:      8, lr 6e-5, AdamW, weight decay 0,01
scheduler:  cosine com warmup de 10%
precisão:   bf16 (fp16 se a GPU não suportar bf16)
loss:       0.5 * CrossEntropy com peso [1.0, 2.0, 3.0, 10.0]
            + 0.5 * Dice multiclasse
seed:       42
checkpoints: checkpoints/best.pt (melhor val_mean_iou)
             checkpoints/last.pt (a cada época)
```

Se houver OOM: baixe para batch 4 com
`gradient_accumulation_steps=2` (batch efetivo igual).
**Não reduza a resolução** para caber na memória: traço fino
depende dos 512×512.

## 5. Tempo estimado (faixa, com ressalva)

Estimativa registrada na config, para dataset sintético de
~5–10 mil imagens 512×512, batch 8, 30 épocas: **3–6 horas em T4**.
Dataset menor (~2 mil imagens) ou 15 épocas: **1–2 horas**.

Isso é faixa, não promessa: o tempo real depende do tamanho do
dataset gerado, da GPU sorteada (T4 vs P100) e do tempo por época
que o próprio notebook registra. Antes de assumir que cabe numa
sessão, verifique o limite de sessão/GPU vigente na conta do
Kaggle no momento da execução e salve checkpoints com frequência
(`last.pt` a cada época existe para isso). Se a sessão cair,
retome a partir de `checkpoints/last.pt` — não recomece do zero
sem necessidade e registre a retomada em `training_config.json`.

## 6. O que o treino deve produzir (artefatos de saída)

Ao final, `/kaggle/working/out` deve conter, no mínimo:

```text
out/
  draw_segmentation_v1.onnx   # modelo exportado, opset 14
  classes.json                # {"0": "background", "1": "paper", "2": "hand", "3": "ink"}
  metrics.json                # métricas por classe + validação ONNX + ink com tolerância
  training_config.json        # config efetiva usada (inclui retomadas e recalibrações)
  split_manifest.json         # cópia do manifest usado, com seeds e IDs por split
  checkpoints/
    best.pt
    last.pt
```

Mais, para comparação entre versões (bateria fixa de 15 cenários,
ver seção 7): imagens de predição/overlay salvas, uma por cenário.

## 7. Baixar os artefatos de volta

1. No fim da sessão, compacte `/kaggle/working/out` (ex.: zip) e
   baixe pelo navegador de arquivos do notebook. Não confie só
   na sessão: arquivo em `/kaggle/working` some quando a sessão
   termina.
2. Recomendado, para reprodutibilidade: salve também os artefatos
   como uma **nova versão privada do dataset** no Kaggle
   (ou como output versionado do notebook), com o mesmo nome e
   versão do treino. Guarde o link/versão junto com o hash dos
   arquivos.
3. **Não commite o `.onnx` no git por padrão.** O contrato atual
   de `draw/neural.py` já prevê o binário fora do repositório
   (caminho por argumento ou `DRAW_NEURAL_MODEL`). Entregue o
   ONNX pelo armazenamento do Kaggle / release, com hash
   registrado, e só versione no repo os JSONs de métricas,
   config e manifest (pequenos, em texto).
4. Depois de baixar, confira localmente: `classes.json` com a
   ordem 0..3 exata, `metrics.json` com métricas por classe
   (não só accuracy), `training_config.json` igual à config
   usada, e `split_manifest.json` igual ao do dataset de entrada.

## 8. Checklist de verificação antes de fechar a sessão

Não feche a sessão com o treino "rodando bem" no olho. Tudo
abaixo tem que estar registrado em arquivo, não só na tela:

- [ ] **Notebook final verificado.** Quando
  `notebooks/train_segformer_kaggle.ipynb` existir, confira que
  ele: lê `training/config.json` (não hiperparâmetros soltos em
  célula), usa os splits do gerador sem reembaralhar fontes,
  treina SegFormer-B0 com transfer learning, exporta ONNX e
  escreve os 5 artefatos da seção 6. Se o notebook divergir de
  `training/train.py`, pare e alinhe os dois antes de treinar.
- [ ] **Nome do output ONNX resolvido (divergência conhecida).**
  `training/train.py` e `training/config.json` exportam o output
  como `logits`; `docs/neural-seg-v1/INTEGRATION.md` documenta
  o output como `mask_logits`. **Não assuma um dos dois.**
  Antes do treino, confirme qual nome o notebook realmente
  exporta e alinhe o outro documento a ele (ou renomeie a
  exportação). Um ONNX com nome de output diferente do
  documentado é bug de integração esperando para acontecer.
- [ ] **Métricas por classe presentes em `metrics.json`:**
  IoU de background/paper/hand/ink, Dice de paper/hand/ink,
  precision e recall. Accuracy global sozinha **não** valida nada
  (fundo dominante infla accuracy).
- [ ] **Métrica de ink com tolerância:** precision/recall de ink
  com tolerância de 2–3 px (dilatação é a implementação de
  referência), reportada **ao lado** da IoU estrita, nunca no
  lugar dela. Traço fino deslocado 1–2 px pode estar visualmente
  certo e mesmo assim derrubar a IoU — a tolerância mede isso.
- [ ] **Bateria fixa de 15 cenários com predições salvas:**
  folha limpa; folha com desenho; folha muito desenhada; mão
  cobrindo canto; mão cobrindo centro; mão cobrindo traço; mão
  parcialmente fora da imagem; fundo claro; fundo escuro; sombra
  forte; folha parcialmente fora da imagem; mão sem folha;
  objeto branco sem folha; traço fino de lápis; traço grosso de
  caneta.
- [ ] **Validação ONNX vs PyTorch:** mesmo lote de validação,
  modelo original e ONNX, com diferença máxima absoluta dos
  logits e IoU por classe registrados em `metrics.json`
  (campo `onnx_validation`). Se `onnxruntime` estiver ausente
  e a validação for pulada, a entrega **não** está validada —
  instale e rode de novo, não entregue sem isso.
- [ ] **Sem vazamento de split:** nenhum peso, limiar ou peso
  de classe foi ajustado olhando o split de teste. Recalibração
  de peso de classe, se houver, usa só contagem do treino e
  fica registrada em `training_config.json`.
- [ ] **Holdout real separado (quando disponível):** métricas
  sintéticas e reais reportadas separadamente, nunca misturadas
  no mesmo número. Queda grande no real é domain gap
  documentado, não falha escondida e não motivo para treinar
  em cima do holdout.
- [ ] **Amostras visuais:** pelo menos uma imagem por cenário da
  bateria, com overlay da predição, para comparação entre
  versões do modelo.
- [ ] **Artefatos baixados e com hash registrado** (seção 7),
  e ONNX **não** commitado no git por padrão.

## 9. Problemas comuns (e o que não fazer)

| Sintoma | Causa provável | Ação |
|---|---|---|
| OOM na primeira época | Batch 8 não cabe na GPU sorteada | Batch 4 + acumulação 2; nunca reduzir resolução |
| Máscara com valor fora de 0..3 | Dataset gerado errado | Abortar, corrigir o gerador, regenerar; nunca "arredondar" máscara no treino |
| `onnxruntime` ausente, validação pulada | Dependência não instalada no notebook | Instalar e validar; entrega sem validação ONNX não vale |
| Pesos pré-treinados não baixam | Internet desligada / cache Hugging Face | Ligar internet no notebook e tentar de novo; não trocar de arquitetura |
| Dataset não aparece em `/kaggle/input/` | Dataset não anexado ao notebook | Anexar (seção 3) e listar `/kaggle/input/` antes de treinar |
| Sessão cai no meio | Limite de sessão/GPU | Retomar de `checkpoints/last.pt`, registrar retomada |
| Métricas altas no sintético, baixas no real | Domain gap (esperado) | Documentar, não treinar no holdout, não misturar os números |
| Output ONNX com nome diferente do documento | Divergência `logits` vs `mask_logits` (seção 8) | Alinhar exportação e documento antes de entregar |

## 10. Itens que dependem do notebook final (pendentes)

1. **O notebook em si** (`notebooks/train_segformer_kaggle.ipynb`):
   estrutura de células, imports, como ele lê a config, como
   salva checkpoints e como exporta o ONNX. Este roteiro assume
   que ele seguirá `training/train.py` e `training/config.json`.
2. **Resolução da divergência do nome do output ONNX**
   (`logits` vs `mask_logits`) — ver seção 8.
3. **Tempo real por época**, que só existe depois da primeira
   execução com o dataset gerado de verdade.
4. **Avaliação no holdout real**, que depende dos frames reais
   fornecidos pelo Codex (fora deste escopo; o Muse não captura
   câmera).
5. **Amostras visuais e métricas finais**, que só existem depois
   do treino completo no Kaggle (não executado nesta máquina,
   sem GPU).
