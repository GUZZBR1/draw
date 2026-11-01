# Plano de treino e datasets no Kaggle — handoff

**Data:** 2026-10-31  
**Branch-alvo:** `training-zone`  
**Objetivo:** preparar revisões privadas e rastreáveis dos datasets Draw e, quando os gates da etapa de treino abaixo passarem, executar um treino Kaggle privado de segmentação single-frame. A detecção temporal continua sendo uma tarefa e um dataset separados.

## Instrução para o chat executor

Leia este documento inteiro antes de agir. Trabalhe somente no worktree da branch `training-zone`; não mude nem transporte alterações da `main`. Preserve os originais, não publique dados da webcam como públicos e não chame pseudo-rótulos de ground truth. A autorização atual permite iniciar somente kernels privados de treino, e somente após todos os gates da seção 10 passarem. Não publique novos datasets nem altere versões de dataset sem necessidade documentada.

Antes de qualquer upload, faça o preflight local abaixo. Se um artefato falhar, publique somente os pacotes que passaram e registre o bloqueio dos outros. Não remova dados, não substitua uma versão anterior sem manter uma revisão rastreável e não altere datasets públicos.

## 1. Meta e tarefas separadas

O alvo principal é aprender a identificar os traços visíveis na folha capturada pela câmera. Há três problemas diferentes; não misturar suas máscaras ou métricas:

1. **Segmentação em um frame:** classes `0=background`, `1=papel visível`, `2=mão visível`, `3=tinta/desenho adicionado pelo usuário visível`, `255=incerto/ignorar`. Texto e impressão pré-existentes pertencem a papel, conforme o contrato do projeto.
2. **Detecção temporal de tinta nova:** entrada de pares de frames antes/depois e máscara da mudança atribuível a traço novo. Métricas deste modelo não são comparáveis à IoU de tinta do segmentador.
3. **Vetorização/previsão de traço:** fase posterior e dataset próprio, com ordem e tempo dos pontos, início/fim de traço e coordenadas. Máscaras de pixels sozinhas não contêm essa informação.

Rótulos de cena (por exemplo, “mão escrevendo”) servem para filtrar e estratificar exemplos; **não substituem máscaras pixel a pixel**.

## 2. Branch, checkout e preservação

- O alvo remoto observado é `origin/training-zone`, no commit `5fa5a0c7` (`feat(training): publish reviewed segmentation workflow`). Foi criado um worktree local limpo em `.worktrees/training-zone`, branch local `training-zone`, acompanhando esse remoto.
- O checkout principal continua em `main`, 5 commits à frente de `origin/main`, com alterações locais preexistentes. Não trocar a branch principal, não fazer cherry-pick delas e não as limpar.
- Todo código, documentação, configuração, manifestos de pacote e commits desta tarefa ficam em `training-zone`. Capturas privadas volumosas e checkpoints não entram no Git.
- A pasta de capturas atualmente está no checkout principal em `/home/gusta/projetos/coimbra/draw/captures/`; leia-a como fonte. Gere pacote temporário dentro do worktree `training-zone`, mantendo os JPEGs originais intactos.
- Não fazer push de branch/commit sem pedido explícito. A autorização de treino é condicionada à seção 10 e não autoriza ativação do modelo na câmera ao vivo.

## 3. Inventário verificado

### 3.1 Datasets Kaggle existentes

Use `kaggle datasets metadata`, `kaggle datasets files` e os manifestos/README de cada dataset para confirmar a edição antes de empacotar. A lista abaixo é a auditoria observada em 2026-10-31:

| Dataset | Conteúdo / papel | Decisão |
|---|---|---|
| `guzzbr/draw-ink-synth-v1` | 16.000 páginas sintéticas de treino e 2.000 de julgamento, máscaras pixel a pixel de tinta e impressão, manifestos. | Pode ser usado em pré-treino sintético; preservar split e não chamar julgamento sintético de teste real. Não reenviar cópia sem mudança material. |
| `guzzbr/draw-ink-pairs-v1` | Pares sintéticos before/after; usado pela família de modelo temporal. | Manter separado do segmentador single-frame. Não alegar qualidade de câmera. |
| `guzzbr/drawink-corpus` | Edição pública: QuickDraw, EMNIST e 1.000 páginas sintéticas com camadas `mask_total`, `mask_visible`, `ignore`, `clean`. | Complemento de pré-treino; conferir hash/linhagem para não duplicar o `draw-ink-synth-v1`. Catálogo mostrou tamanho zero, mas o endpoint de arquivos listou conteúdo. |
| `guzzbr/draw-real-seg-pilot-v1` | 719 imagens reais com pseudo-máscaras; 707 com pixels supervisionados e 12 sem pixels de treino. | Somente experimento/treino fraco, identificado como pseudo-label. Não serve para avaliação de qualidade nem holdout. |
| `guzzbr/draw-segformer-expert-pilot-v1` | 19 amostras de um piloto de consenso por agentes; um único contexto/sessão, sem holdout. | Exploração apenas; consenso de agentes não é ground truth humano adjudicado. |
| `guzzbr/draw-ink-real-v1` | Frames reais retificados com rótulos heurísticos e região `255` ignorada. | Experimento fraco; manter sua autoridade de rótulo explícita. |
| `guzzbr/draw-realbg-v2` | Fundos reais, negativos sem tinta e imagens de holdout de cenário vazio. | Útil para falsos positivos/negativos; não é fonte de exemplos positivos de tinta. Preservar o holdout sem treino. |
| `guzzbr/draw-drawfmt-v1` | DIBCO/H-DIBCO, sprites de tinta CVL e sondas de falso positivo. | Privado e licenciamento por componente; não copiar para pacote público nem misturar sem conferir `SOURCES.md`. |
| `guzzbr/drawink-corpus-full` / `drawink-corpus-pilot` | Corpus de documentos, incluindo DIBCO, CVL e outras fontes. Algumas fontes têm licença restrita ou sem licença de redistribuição. | Não redistribuir. Uso privado apenas conforme licença/origem documentada. |

Outros datasets de documentos/notebooks (SmartDoc, MIDV, School Notebooks etc.) não fornecem automaticamente máscara da tinta do usuário; só incluir se a tarefa e os rótulos forem inspecionados e compatíveis.

### 3.2 Capturas locais e estado dos rótulos

- `captures/training/session_*`: **8 sessões, 5.086 JPEGs, cerca de 198,4 MiB**. Resoluções: uma sessão 640×480 e sete 1280×720, conforme inventário anterior.
- A categorização cruzada `captures/triage/local_training_capture_categorization_20261031_v1/README.md` relata nesse subconjunto: 286 `pseudo_supervised`, 4.789 sem rótulo e 11 `pseudo_all_unknown`; nenhum frame marcado como aprovado para treino.
- O manifesto canônico `captures/manifests/source_frames_20261031_after_user_review.csv` cobre **61.106 frames**: 60.207 sem rótulo, 688 pseudo-supervisionados, 180 propostas de modelo que precisam correção, 19 de consenso exploratório e 12 totalmente ignorados. Nenhum está `training_approved=True`.
- Há também 216 decisões de cena em `captures/triage/review_batches/part_02/labels_reviewed.csv` (207 manter, 9 excluir como duplicata clara). Isso é triagem por cena, não segmentação pixel a pixel.
- As contagens 286 e 688 têm escopos diferentes; não somar. Cruzar por ID e SHA-256 antes de qualquer pacote.
- O pacote conhecido de 719 frames de duas sessões está designado como treino exploratório. Bordas finas não foram revisadas e essas sessões não podem ser reaproveitadas como holdout.
- **Conclusão atual:** há material para pacotes de revisão e experiências com pseudo-rótulos, mas os 5.086 JPEGs não são ground truth pronto. O estado de aprovação é zero.

## 4. Ferramentas locais já disponíveis

No branch `training-zone`, inspecionar e reaproveitar antes de criar ferramenta nova:

- `tools/build_capture_inventory.py` — inventário/crosswalk entre IDs de frames, manifesto e hashes.
- `tools/build_expert_reviewed_split.py` e `tools/package_expert_consensus_pilot.py` — fluxo de revisão/consenso; conferir regra de sessão congelada, pois a seleção antiga de holdout conflita com a alocação mais recente.
- `tools/build_training_dataset.py` — pacote consumido pelo treinamento.
- `tools/propose_semantic_masks.py`, `tools/evaluate_pixelmask_proposals.py`, `tools/make_mask_review_app.py`, `tools/serve_mask_review_app.py` — propostas e revisão; proposta não equivale a rótulo aprovado.
- `training/config.json`, `training/train.py`, `notebooks/train_segformer_kaggle.ipynb` — contrato e treino Kaggle. A config registra as 719 imagens reais como pseudo-labels em `real_train`, split por sessão e ausência de holdout real.
- `docs/neural-seg-v1/INTEGRATION.md`, `REAL_MASK_ANNOTATION.md`, `REAL_FRAMES_PLAN.md`, `KAGGLE_RUNBOOK.md` e `SENADO_VALIDACAO_KAGGLE_2026-10-30.md` — contratos, limitações e decisões anteriores.

## 5. Fluxo de execução

### Etapa A — reconfirmar estado e proveniência

1. Registrar `git status`, commit, branch, hash de configuração/código e versão exata dos manifestos.
2. Consultar metadados, versões e arquivo-lista dos slugs Kaggle acima. Salvar resposta de inventário no próprio bundle da tarefa, sem credenciais.
3. Cruzar as 8 sessões locais com o manifesto canônico por ID, nome e SHA-256. Resolver discrepâncias antes do pacote. Detectar duplicatas exatas e parentesco de crops/resize/frames vizinhos.
4. Preservar o holdout existente `draw-realbg-v2` e todas as sessões que já foram usadas em treino. Nenhum frame de uma sessão de treino entra em holdout.

### Etapa B — formar pacotes locais

Criar releases locais privados, com identificador imutável e manifestos, sem sobrescrever originais:

1. **Pool de revisão real (criar):** JPEG original, `frames.csv`, ID/sessão/timestamp/resolução/hash, tag de cena já fornecida, estado de rótulo e caminho de máscara se existir. Incluir os 5.086 frames ou justificar uma amostragem por critérios explícitos. Não inventar máscara para os não rotulados.
2. **Candidatos pseudo (se os arquivos e hashes conferirem):** exportar máscaras pseudo em coluna/camada separada, com método, versão, autoridade e pixels por classe/ignore. Marcar `heuristic_pseudo` ou `model_proposal`; excluir por padrão da avaliação e do treino supervisionado forte.
3. **Frames com consenso de agentes:** manter em conjunto/campo separado `agent_consensus_exploratory`; nunca nomear “human ground truth”.
4. **Dataset temporal em revisão (opcional, criar separado):** pares de frames reais com IDs/timestamps alinhados e estado `unlabeled` se não houver máscara adjudicada de tinta nova. Não derivar ground truth automático por diferença/heurística. Só usar para treino supervisionado quando a máscara de delta for revisada.
5. **Máscaras humanas aprovadas:** nenhuma disponível no inventário atual. Quando surgirem, usar quatro classes, `255` para incerteza genuína, resolução nativa e revisão/adjudicação documentadas.

### Etapa C — validação obrigatória do pacote

O preflight deve falhar fechado e gerar relatório JSON/CSV:

- todo sample tem ID estável, sessão/take, original e SHA-256, dimensões, split, fonte, licença e autoridade/estado de anotação;
- imagens decodificam; imagem e máscara correspondem em dimensões; máscara é single-channel e contém somente `0,1,2,3,255`;
- imagem sem máscara aparece como `unlabeled`, nunca com máscara vazia inferida; `255` nunca vira fundo; amostra totalmente `255` não entra no treino;
- nenhuma sobreposição de sessão, take, frame, hash ou linhagem entre train/val/test; frames derivados e pares ficam no split da fonte;
- treinar/selecionar apenas com dados permitidos; não ajustar limiar usando test/holdout;
- licença e privacidade verificadas; imagens de webcam do proprietário ficam em dataset privado;
- tamanho, contagem e manifestos conferem antes/depois do arquivo.

### Etapa D — criar/atualizar datasets Kaggle

1. Usar datasets **privados** para dados de webcam do proprietário e para pseudo-rótulos. Nunca atualizar o corpus público com capturas privadas.
2. Preferir um novo slug privado e versionado para o pool de captura/revisão, por exemplo `draw-real-capture-pool-v1`, caso o formato/tarefa seja diferente dos datasets existentes.
3. Atualizar `draw-real-seg-pilot-v1` somente se o conteúdo mantiver o mesmo contrato e a revisão explicitar que são pseudo-labels exploratórios. Não fundir automaticamente as 719 imagens antigas com 5.086 frames novos; primeiro cruzar hashes e preservar os dois manifestos de origem.
4. Não atualizar o holdout `draw-realbg-v2` com dados de treino. Não incorporar novos pares reais ao `draw-ink-pairs-v1` sem rótulos temporais revisados; criar pacote temporal privado separado para pares sem rótulo/revisão.
5. Não copiar material do `drawink-corpus-full`, CVL, DIBCO ou DocUNet para uma edição pública sem autorização/licença verificada por fonte.
6. Publicação Kaggle só ocorre depois dos gates da etapa C. A instrução original autorizava criar/atualizar datasets privados; a autorização atual permite apenas kernels privados após a seção 10 e não permite publicar dados públicos.
7. Após upload, verificar `kaggle datasets files`, versão, privacidade, tamanhos e hashes por download amostral ou integral conforme viável. Guardar slug, version number, arquivo-lista, hashes e logs no relatório local.

### Etapa E — fechamento local

- Registrar no branch `training-zone` documentação, scripts reutilizáveis, schemas, relatórios e manifestos leves. Não versionar JPEGs, máscaras privadas grandes, tokens ou checkpoints pesados.
- Commit local na branch `training-zone`; não fazer push sem pedido explícito.
- Encerrar a tarefa com relatório: datasets criados/atualizados, versão e privacy, contagens por estado de anotação, hashes, preflight, arquivos não enviados e motivo.
- A regra original de não iniciar kernel foi substituída pela autorização condicional da seção 10, concedida pelo usuário em 2026-11-01. Se qualquer gate falhar, registrar bloqueio e não iniciar treino.

## 6. Protocolo do Senado e critérios de decisão

Revisão de arquitetura, crítica adversarial e verificação são gates independentes. Parecer atual dos revisores: **aprovar o empacotamento/versionamento exploratório, bloquear alegações de qualidade real e holdout**.

### Aceite para dataset de revisão/pseudo-label

- Manifesto completo e reproduzível, fonte e autoridade de cada rótulo claras.
- Privacidade Kaggle = private para conteúdo de câmera.
- Zero colisões sem resolução, zero vazamento entre sessões/splits, originais preservados.
- Pseudo-labels e consenso claramente separados de máscaras humanas.
- File listing e hashes remotos conferem com o release local.

### Aceite para dataset de treino supervisionado real

- Máscaras humanas revisadas pixel a pixel, dimensões e classes válidas, incerteza marcada como `255`.
- Sessões inteiras separadas entre train/val/test; avaliação real final usa sessão nova e não participa de seleção.
- Support/cobertura de tinta, mão, papel e negativos suficiente; protocolo e agregação de métrica congelados antes de abrir holdout.

### Aceite para modelo/ONNX, em tarefa posterior

- Execução CPU, tensores/contrato corretos, logits finitos e paridade PyTorch–ONNX com `max_abs_diff <= 1e-3` e `argmax IoU >= 0.99` por classe, conforme config/validador do branch.
- Gate de artefato ONNX separado do gate de qualidade de imagem.
- Nenhuma métrica sintética, pseudo-label ou IoU temporal é apresentada como precisão real de câmera.

## 7. Riscos e reparos obrigatórios

| Risco | Reparo |
|---|---|
| Tags de cena confundidas com segmentação | Nunca promover `scene_label` a máscara; manter tarefas/campos distintos. |
| Pseudo-label parecer anotação aprovada | Campo obrigatório `annotation_authority`/`label_status`, rótulos separados e aviso no README Kaggle. |
| Vazamento por frames vizinhos, pares ou augmentações | Split por sessão/take e linhagem, não aleatório por arquivo. |
| Reempacotar fonte com licença restrita | Manter privado/omitir e registrar licença individual. |
| Upload substitui material anterior | Novos releases versionados, changelog, hashes e verificação remota. |
| Usar captura nova como holdout sem independência | Holdout só de sessão inédita, não usada em treino ou seleção. |
| “Autoaperfeiçoamento” aceita sozinho modelo pior | Cada rodada gera candidato e relatório; promoção permanece gated por métricas/revisão, nunca automática por completar o ciclo. |

## 8. Fatos e referências locais para confirmar no checkout alvo

- `training/config.json`: classes, `255`, 719 pseudo-labels em treino, duas sessões, nenhum holdout real independente.
- `tools/build_capture_inventory.py`, `tools/build_training_dataset.py`, `tools/build_expert_reviewed_split.py`, `tools/package_expert_consensus_pilot.py`.
- `captures/triage/local_training_capture_categorization_20261031_v1/README.md` e `captures/manifests/source_frames_20261031_after_user_review.csv` (dados locais estão no checkout principal; confirmar acessibilidade, sem movê-los).
- `captures/triage/review_batches/part_02/labels_reviewed.csv`: triagem de cena, não pixel masks.
- `docs/neural-seg-v1/REAL_MASK_ANNOTATION.md`, `REAL_FRAMES_PLAN.md`, `INTEGRATION.md`, `KAGGLE_RUNBOOK.md`, `SENADO_VALIDACAO_KAGGLE_2026-10-30.md`.
- `tools/runner/README.md` e `tools/runner/cycle.py`: ciclo de treino existente; não decide promoção automática do modelo.

## 9. Resultado esperado desta execução

Ao terminar, deve haver evidência local auditável do preflight e, somente se a seção 10 passar, resultado de kernel privado. Se não houver dados compatíveis para qualidade real, nenhum resultado sintético ou pseudo-label deve ser apresentado como validação de câmera.

## 10. Etapa F — treino Kaggle single-frame (autorização condicional)

Esta etapa foi acrescentada em 2026-11-01. Ela autoriza um kernel privado somente quando os dados e o pipeline forem verificados antes da execução. O treino single-frame marca pixels visíveis de tinta (`classe 3`). Pares before/after, máscaras de delta e métricas temporais pertencem a outra tarefa; não os anexar nem comparar seus resultados com IoU de segmentação.

### 10.1 Dados elegíveis e limites

- A entrada elegível para esta rodada exploratória é `guzzbr/draw-synthetic-v1-part1` a `part8`, versões privadas 1. Os seis primeiros pacotes contêm 2.400 cenas de treino; `part7` e `part8` contêm 300 cenas de validação e 300 de teste, respectivamente. Cada cena tem imagem 512×512 e máscara por pixel com classes `0=background`, `1=paper`, `2=hand`, `3=ink`; `ink_full` é auxiliar e nunca é entrada/target do treino.
- O `split_manifest.json` é obrigatório e deve conferir seed 42, IDs por cena e separação das fontes de fundo, mão e papel; não dividir ou embaralhar cenas depois. Falhar se houver sobreposição de fonte, pares ausentes, classe sem pixels ou máscara fora de `{0,1,2,3,255}`.
- Os datasets são sintéticos. Resultado desta rodada é **exploratório/sintético**, não demonstra qualidade em câmera real e não recebe rótulo de avaliação humana.
- `guzzbr/draw-real-capture-pool-v1` (5.086 frames, 297 pseudo-máscaras e 4.789 sem máscara) é somente revisão. Não anexar ao treino. `guzzbr/draw-real-seg-pilot-v1` (719 pseudo-rótulos) também fica fora desta rodada. Não usar cenas descritas, pseudo-máscaras ou consenso de agentes como ground truth.
- `guzzbr/draw-ink-pairs-v1` é temporal e fica fora deste treino. `guzzbr/draw-realbg-v2` preserva seu papel de holdout e não deve ser aberto antes do fechamento das decisões de modelo. Não misturar `draw-ink-synth-v1` (rótulos de tinta/impressão em páginas documentais) com o contrato 4-classes desta rodada.

### 10.2 Auditoria obrigatória pré-kernel

Antes de `kaggle kernels push` ou outra execução, registrar para cada input utilizado: slug, versão atual, status `ready`, privacidade, metadata, listagem integral de nomes/tamanhos e SHA-256 do conteúdo baixado ou hash manifesto equivalente verificável. Para o dataset sintético, comparar todos os arquivos e tamanhos com os ZIPs baixados; conferir hashes por arquivo, integridade dos arquivos, dimensões, classes e contagens de split. Confirmar também a versão privada do cache `guzzbr/mit-b0-cache` e a fonte/código exatos do treino. Não incluir input cuja privacidade, conteúdo ou versão não possa ser confirmada.

### 10.3 Critérios e comparação definidos antes do treino

- Código registra seed 42 (Python, NumPy, PyTorch e CUDA), hash do código/configuração/manifesto, hash por arquivo e fingerprint do dataset, slugs/versões/privacidade dos inputs, ambiente/PyTorch/GPU, contagens de pixels e amostras, checkpoints, ONNX e hashes de saída.
- Seleção de checkpoint usa somente `val_mean_iou`; `test` não participa de treino, early stopping, limiar, pesos ou seleção. Avaliar `test` uma vez, após selecionar o checkpoint, e reportar suas métricas separadamente.
- Reportar em validação e teste sintéticos, por classe: IoU, Dice, precision e recall; reportar também precision/recall de tinta com tolerância de 2 e 3 px ao lado da IoU estrita. Não usar accuracy global como critério.
- Baseline congelado antes desta rodada: execução `synthetic_only`, mesmo manifesto de 3.000 cenas (hash remoto `9ba05ab08ad5bf276a96b527777424b8e143958a31d192e7862d458af105c931`), 2.400 pares de treino; validação `mIoU=0.9177034868`, `IoU_ink=0.7433429512`, ONNX validado (`max_abs_diff=5.72e-05`, mínimo IoU de argmax 1,0). Comparar `novo - baseline` na validação; não usar o teste para decidir hiperparâmetros. Para leitura diagnóstica, avaliar baseline e candidato uma única vez no mesmo teste sintético, sem seleção posterior baseada nesse resultado.
- Aceite técnico de exportação: ONNX Runtime efetivamente executado; `onnx_validation.status=ok`, `max_abs_diff <= 1e-3` e `min_argmax_iou >= 0.99`. Falta de ONNX Runtime, validação pulada, teste ausente ou hash/manifesto ausente significa **não validado**, nunca sucesso.
- Gate comparativo pré-fixado: na validação, aceitar como candidato exploratório apenas se `mIoU >= baseline - 0.005` e `IoU_ink >= baseline - 0.01`, além do gate ONNX. Isso não autoriza integração ao vivo nem alegação real; qualquer falha mantém o baseline e registra o resultado como reprovado.

### 10.4 Execução e artefatos

- Executar em kernel `is_private=true`, GPU confirmada dentro do runtime e apenas com os oito chunks sintéticos e o cache privado de pesos necessários. O pool de revisão e o pacote real pseudo não são inputs. Não criar/atualizar datasets Kaggle para a rodada; não fazer push da branch.
- Salvar `best.pt`, `last.pt`, `.onnx`, `classes.json`, `metrics.json`, `training_config.json`, `split_manifest.json`, `artifact_hashes.json`, inventário de hashes de input e log completo em output privado do kernel. Se CUDA, preflight, classes, hash, exportação ou validação ONNX falhar, parar e marcar execução como bloqueada/não validada.
- Registrar kernel/versão/estado, datasets/versões/privacidade, splits/contagens, métricas por classe e baseline, teste sintético, hashes, limitações e próximo ciclo de anotação. Manter segmentação e detecção temporal em seções e métricas distintas.
