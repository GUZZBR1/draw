# Ambiente do Kaggle × código de treino — compatibilidade

Documento de compatibilidade, não de correção. Nenhum arquivo de código
foi alterado para escrevê-lo: as divergências abaixo continuam existindo
em `training/train.py` e em `training/config.json` até que alguém as
corrija nos arquivos indicados em cada item.

Contexto: teste de 2026-10-29 no Kaggle (kernel de 2 épocas, 36 cenas)
falhou antes do treino por falta de internet e de GPU, apesar de
`enable_gpu` e `enable_internet` no metadata do push via CLI. O log
daquele teste mostrou `torch 2.11.0+cpu | cuda: False`. Ou seja:
metadata via CLI não é garantia de GPU nem de internet — isso só se
confirma na interface web do Kaggle, na execução real.

## O que o Kaggle normalmente já traz (com ressalva)

Versões típicas, não prometidas: o ambiente do Kaggle é uma imagem
Docker versionada e muda com o tempo. A única confirmação válida é o
próprio kernel, imprimindo versões na primeira célula. Não fixe
versão em doc sem antes ler o log de uma execução.

| Pacote | Uso no projeto | Situação típica no Kaggle |
|---|---|---|
| `torch` (2.x, CUDA) | Modelo, loss, otimizador, ONNX | Pré-instalado. No teste de 2026-10-29, estava presente mesmo sem internet (`torch 2.11.0+cpu`). |
| `transformers` | `SegformerForSemanticSegmentation`, pesos `nvidia/mit-b0` | Costuma estar pré-instalado, mas em versão mais antiga que a mais recente. Baixar os pesos `nvidia/mit-b0` exige internet (ou o peso já em cache / como input). |
| `opencv-python` (`cv2`) | Topo de `training/train.py` (`import cv2`), leitura de imagem/máscara no smoke e no dataset | Costuma estar pré-instalado no ambiente de data science. O notebook não o instala nem o menciona — conta com o default. Se faltar, o treino nem começa. |
| `numpy`, `Pillow` (via `cv2` indiretamente) | Contagem de pixels, métricas numpy | Pré-instalados. |
| `albumentations` | Augmentation em `build_dataset(..., augment=True)` | **Não garantido.** O código já prevê a falta (aviso "treino sem augmentation" e segue), mas sem ele o treino perde a augmentation documentada em `config.json`. |
| `onnx`, `onnxruntime` | Exportação e validação ONNX | **Não garantidos.** Sem `onnxruntime`, `validate_onnx` em `train.py` retorna `None` e a validação é pulada em silêncio no `metrics.json` (só um aviso no stderr). Sem `onnx`, a exportação falha. |
| `git` | Clone do repo na célula de clone do notebook | Existe no ambiente, mas sem internet o clone falha com `Could not resolve host: github.com` — foi exatamente a falha do teste de 2026-10-29. |
| GPU T4/P100 | Treino | **Não é default.** Precisa estar ligada na sessão (web UI) e disponível na cota. O teste de 2026-10-29 rodou em CPU apesar do metadata. |
| Internet | `pip install`, clone, download de pesos | **Não é default em kernel via CLI.** Precisa estar ligada na sessão. Sem ela, nada que baixe funciona. |

Consequência prática: o notebook atual depende de internet em três
pontos independentes — `pip install`, `git clone` e download dos pesos
`nvidia/mit-b0` via `transformers`. Basta um deles falhar para a
execução morrer antes do treino. A rota sem internet (subir `training/`
como input do kernel e usar só o que já está instalado, com os pesos
em cache ou como input) elimina os três de uma vez, mas exige preparar
esses inputs antes — não é só tirar o `pip install`.

## Divergências ambiente × código, item por item

### 1. Warmup prometido, mas sem fase de warmup no código

- Promessa (`training/config.json`): `scheduler: "cosine_with_warmup"`,
  `warmup_ratio: 0.1`.
- Código (`training/train.py`, função `main`): calcula
  `warmup = int(total_steps * warmup_ratio)` e depois cria
  `CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup, 1))`.
  O valor de `warmup` só encurta o `T_max`; não existe rampa de
  learning rate. O treino começa direto no `lr` cheio (6e-5) e decai
  por cosseno.
- Teste pequeno (36 cenas, 2 épocas, CPU/GPU): **não bloqueante.**
  Com 2 épocas o efeito de warmup é irrelevante para validar se o
  notebook roda.
- Treino completo (30 épocas, 5–10 mil imagens): **não bloqueante
  para rodar, bloqueante para reproduzir o documentado.** O modelo
  treina, mas não com o schedule prometido — e qualquer comparação
  com um treino futuro "com warmup de verdade" não é comparável.
- Correção recomendada (sem aplicar aqui): em `training/train.py`,
  trocar o scheduler por um com warmup real (ex.: rampa linear por
  `warmup` passos seguida de decaimento cosseno), ou, se a decisão
  for manter o comportamento atual, mudar `training/config.json`
  para `cosine` sem warmup e documentar que `warmup_ratio` só
  encurta o `T_max`. Não deixar os dois dizendo coisas diferentes.

### 2. bf16 prometido, mas treino em fp32 sem autocast

- Promessa (`training/config.json`): `mixed_precision: "bf16
  (fp16 se a GPU não suportar bf16)"`.
- Código (`training/train.py`): não há `torch.autocast`,
  `GradScaler` nem `half()` em lugar nenhum. `train_one_epoch`
  aceita um parâmetro `scaler=None` que nunca é usado nem passado
  por `main`. Todo o forward, loss e backward rodam em fp32.
- Teste pequeno: **não bloqueante.** Em 36 cenas e 2 épocas, fp32
  só é mais lento; não impede validar o notebook.
- Treino completo: **alto risco, ver seção de OOM.** fp32 dobra
  a memória de ativações e de gradientes em relação a bf16, e é o
  principal motivo de o batch 8 ser arriscado numa T4.
- Correção recomendada (sem aplicar aqui): em `training/train.py`,
  envolver o forward/loss em `torch.autocast(device_type="cuda",
  dtype=torch.bfloat16)` quando houver GPU, e ajustar
  `training/config.json` se a decisão for ficar em fp32 (nesse caso,
  baixar o batch documentado para 4). Não prometer bf16 num arquivo
  e rodar fp32 no outro.

### 3. `gradient_accumulation_steps` ignorado

- Promessa (`training/config.json`): `gradient_accumulation_steps: 1`
  (e o texto diz que, em OOM, deve-se usar batch 4 + acumulação 2
  para manter o batch efetivo).
- Código (`training/train.py`): `train_one_epoch` faz
  `optimizer.step()` a cada lote, sem acumular. `main` nunca lê
  `gradient_accumulation_steps` da config. Ou seja, a mitigação de
  OOM documentada **não existe no código**: baixar para batch 4
  apenas reduz o batch efetivo pela metade, não o mantém.
- Teste pequeno: **não bloqueante.** Batch 4, 2 épocas, 36 cenas
  rodam sem acumulação.
- Treino completo: **bloqueante para a mitigação de OOM, não para
  rodar.** Se der OOM com batch 8, a única saída real hoje é batch
  4 com batch efetivo menor (treino um pouco diferente do
  documentado), ou corrigir o código.
- Correção recomendada (sem aplicar aqui): em `training/train.py`,
  implementar acumulação em `train_one_epoch` (dividir a loss pelo
  número de passos de acumulação, chamar `optimizer.step()` só a
  cada N lotes, ler `gradient_accumulation_steps` da config em
  `main`). Até lá, o runbook deve dizer "batch 4, batch efetivo 4",
  não "batch 4 + acumulação 2".

### 4. Checkpoints salvos, mas sem retomada

- Código (`training/train.py`): salva `checkpoints/last.pt` a cada
  época e `checkpoints/best.pt` quando o `val_mean_iou` melhora.
  Depois do treino, carrega `best.pt` para exportar o ONNX.
- Falta: nenhum argumento `--resume`, nenhuma leitura de
  `last.pt` no início de `main`, nenhum registro de época
  interrompida. Se o kernel cair (sessão do Kaggle tem limite de
  tempo; o treino completo estimado é de 3–6 h em T4), a corrida
  **recomeça do zero**, mesmo com os checkpoints no disco.
- Teste pequeno: **não bloqueante.** 2 épocas acabam antes de
  qualquer limite de sessão razoável.
- Treino completo: **alto risco operacional.** Não impede o
  treino de começar, mas transforma qualquer queda no meio em
  perda total do progresso. Com 30 épocas, isso é o tipo de coisa
  que só aparece na hora errada.
- Correção recomendada (sem aplicar aqui): em `training/train.py`,
  adicionar `--resume` (carregar `last.pt`, restaurar época,
  otimizador e scheduler, e continuar o loop a partir da época
  seguinte) e, no notebook, documentar o caminho de retomada.
  Alternativa sem código: dividir o treino em chamadas menores
  com `--epochs` parciais — o que hoje também não retoma pesos,
  então não é retomada de verdade.

### 5. Validação ONNX sem limiar de aborto

- Promessa (`training/config.json`, `export.validation`):
  "Degradação significativa aborta a entrega."
- Código (`training/train.py`, `validate_onnx`): compara PyTorch
  vs ONNX num lote e grava só `max_abs_diff` em
  `metrics.json` (`onnx_validation`). Não compara contra nenhum
  limiar, não marca degradação, não aborta nada. Se `onnxruntime`
  faltar, grava `None` e segue.
- Referência existente (não usada pelo treino):
  `tools/validate_onnx.py` já tem limiares documentados
  (`max_abs_diff > 1e-3` ou `min_argmax_iou < 0.99` → `degraded:
  True`). O treino não os usa.
- Teste pequeno: **não bloqueante.** O objetivo do teste é ver o
  notebook rodar; o valor de `max_abs_diff` já aparece no
  `metrics.json` para inspeção manual.
- Treino completo / entrega: **bloqueante para chamar a entrega de
  validada.** Um ONNX degradado sairia como artefato normal, com
  `metrics.json` sem nenhum sinal de alerta além de um número
  que ninguém é obrigado a ler.
- Correção recomendada (sem aplicar aqui): em `training/train.py`,
  após `validate_onnx`, comparar `max_abs_diff` e IoU de argmax
  contra os limiares de `tools/validate_onnx.py` e, se degradado,
  sair com erro (ou, no mínimo, gravar `degraded: true` explícito
  em `metrics.json`). Até lá, a célula de métricas do notebook
  deve imprimir e interpretar o valor, não só mostrá-lo.

### 6. Dependências do notebook que o ambiente pode não ter

- O notebook instala `torch transformers albumentations onnx
  onnxruntime` via `pip install` — o que exige internet e tempo.
  Sem internet, falha (foi o que aconteceu no teste de 2026-10-29:
  210 s tentando o PyPI, sem `onnxruntime`).
- `cv2` não é instalado pelo notebook e é importado no topo de
  `training/train.py`. Conta com o default do Kaggle.
- Teste pequeno: **bloqueante se sem internet** (o pip falha antes
  de qualquer célula útil), **não bloqueante se com internet**.
- Treino completo: idem, mais o risco de o `pip install` sem
  versão fixada trocar `transformers`/`torch` por versões
  incompatíveis entre si.
- Correção recomendada (sem aplicar aqui): no notebook, verificar
  versões instaladas antes de instalar (só instalar o que falta,
  com versão fixada), ou, na rota sem internet, subir `training/`
  e os pesos como inputs e remover o `pip install` e o `git clone`.

## Risco de OOM em T4 com batch 8, fp32, 512×512

Estimativa, não medida: SegFormer-B0 tem poucos parâmetros (~3,7 M),
mas a memória dominante não são os pesos — são as ativações. Em
512×512, com batch 8, em fp32, sem autocast:

- Ativações do encoder (várias escalas, a maior em 128×128) +
  ativações do decoder + logits interpolados para 512×512 +
  gradientes (fp32, mesmo tamanho dos pesos, mas sobre ativações
  grandes) + estados do AdamW (2× os pesos) + o lote em si.
- Numa T4 de 16 GB, isso é **plausível, mas sem margem confortável**.
  O ponto mais pesado é o lote de validação e o cálculo de loss
  com logits em 512×512 para 8 imagens ao mesmo tempo, junto com
  o DataLoader e o resto do processo. Qualquer outro consumo
  (outro processo, fragmentação, validação logo após o treino)
  pode empurrar para OOM. Não há medição real ainda — o teste
  pequeno (batch 4, 36 cenas) não prova nada sobre batch 8 em
  5–10 mil imagens.
- Mitigação que funciona hoje: **batch 4**. Reduz as ativações do
  lote aproximadamente pela metade. Custo: batch efetivo menor
  (ver item 3 — sem acumulação, não há como manter o batch
  efetivo 8). Mitigação completa (batch 4 + acumulação 2) exige a
  correção do item 3.
- Mitigação que ajudaria muito, mas exige código: bf16 com
  autocast (item 2) — quase metade da memória de ativações e
  gradientes.
- Recomendação prática para o treino completo, na ordem:
  1. Começar com batch 4 (não 8) até que bf16 ou acumulação
     existam de fato.
  2. Se mesmo batch 4 der OOM, o problema não é mais o batch —
     é preciso medir onde a memória estoura antes de mexer em
     resolução (que nunca deve ser reduzida: traço fino depende
     dela).

## Tabela de decisão

| Item | Teste pequeno (36 cenas, 2 épocas) | Treino completo (30 épocas, 5–10 mil imagens) | Ação antes de… |
|---|---|---|---|
| GPU ligada na sessão | Recomendado, não obrigatório (CPU roda 2 épocas, devagar) | **Obrigatório** — em CPU o treino completo é inviável | …qualquer corrida: confirmar na web UI, não só no metadata |
| Internet ligada | **Obrigatória** na rota atual (pip + clone + pesos) | **Obrigatória** na rota atual | …qualquer corrida: confirmar na web UI, ou migrar para a rota sem internet |
| `torch` / `transformers` / `cv2` pré-instalados | Pode rodar assim mesmo (verificar versões na 1ª célula) | Pode rodar assim mesmo, com versões fixadas no pip se instalar | …nada: só imprimir versões |
| `albumentations` ausente | Pode rodar (treino sem augmentation, com aviso) | Pode rodar, mas perde augmentation documentada | …treino completo: instalar ou aceitar explicitamente a perda |
| `onnxruntime` ausente | Pode rodar, mas validação ONNX é pulada (`None`) | **Não entregar como validado** | …entrega: instalar, ou rodar `tools/validate_onnx.py` à parte |
| Warmup (item 1) | Pode rodar assim mesmo | Pode rodar, mas **não reproduz o documentado** | …treino completo: corrigir o scheduler ou corrigir a config |
| bf16 vs fp32 (item 2) | Pode rodar assim mesmo | **Alto risco de OOM** com batch 8 | …treino completo: usar batch 4, ou implementar autocast |
| Acumulação ignorada (item 3) | Pode rodar assim mesmo | Mitigação de OOM documentada **não funciona** | …depender da mitigação: implementar acumulação |
| Sem retomada (item 4) | Pode rodar assim mesmo | **Alto risco operacional** (queda = recomeço) | …treino longo: implementar `--resume`, ou aceitar o risco explicitamente |
| ONNX sem limiar (item 5) | Pode rodar assim mesmo (inspecionar `max_abs_diff` à mão) | **Não chamar de validado** sem limiar | …entrega: implementar o limiar ou validar à parte com `tools/validate_onnx.py` |
| Batch 8 fp32 em T4 | Não se aplica (teste usa batch 4) | **Arriscado** — começar com batch 4 | …treino completo: batch 4 até bf16/acumulação existirem |

## Veredicto

- **O teste pequeno pode rodar assim mesmo**, desde que GPU e
  internet estejam de fato ligadas na sessão (ou que se migre
  para a rota sem internet). Nenhuma das divergências acima
  impede 2 épocas em 36 cenas.
- **O treino completo não deve começar sem antes decidir,
  explicitamente, sobre:** batch 4 (não 8), ausência de retomada
  e validação ONNX sem limiar. Warmup e bf16 podem ser aceitos
  como "documentação divergente" por uma corrida, mas não devem
  continuar divergentes depois dela — a correção é pequena e
  está mapeada item por item acima.
- **Nada aqui foi corrigido no código.** Este documento é o mapa
  das correções; aplicá-las é o próximo passo, em
  `training/train.py` e `training/config.json`, com testes
  locais (smoke + pareamento) antes de qualquer nova corrida
  no Kaggle.
