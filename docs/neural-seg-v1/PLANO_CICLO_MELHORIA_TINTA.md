# Plano: ciclo de melhoria do modelo de traços

**Escopo:** segmentação pixel a pixel de tinta/traço para a câmera ao vivo do Draw. Este documento não descreve o classificador de cena (`papel parado`, `mão escrevendo`, `mão interagindo`): são modelos e rótulos diferentes.

**Princípio:** o modelo pode sugerir máscaras e apontar exemplos difíceis. Uma previsão não vira rótulo de treino sem revisão. O modelo não altera os próprios pesos durante a captura.

**Estado:** proposta de planejamento, não autorização de execução. Antes de implementar o ciclo ou iniciar treino/envio ao Kaggle, a arquitetura precisa de revisão do Senado e o usuário precisa pedir explicitamente a execução no Kaggle.

## Resultado pretendido

A cada sessão de câmera, coletar material com proveniência; revisar as máscaras sugeridas; treinar uma versão candidata; medir se ela melhora em dados reais mantidos fora do treino; e só então atualizar o modelo da câmera. A melhoria é contínua entre sessões, mas supervisionada por revisão e gates de qualidade.

## Modelos e tarefas

1. **Segmentação de um frame:** SegFormer com as classes visíveis `background=0`, `paper=1`, `hand=2`, `ink=3`. A mão cobre a tinta quando o traço está ocluído. `255` significa pixel incerto e é ignorado na loss e nas métricas.
2. **Comparação temporal de pares:** os modelos `ink-pairs` comparam frames antes/depois e tentam encontrar tinta nova. É uma tarefa distinta; seus scores não podem ser comparados diretamente com IoU de tinta do SegFormer.
3. **Aplicação ao vivo:** executar o modelo aprovado em cada frame que o app processar, medir latência e gravar a versão do modelo. Nenhum peso muda durante a captura.

A implementação de treino principal está em `training/train.py` e é chamada por `notebooks/train_segformer_kaggle.ipynb`. Ferramentas existentes incluem `tools/propose_semantic_masks.py`, `tools/make_review_sheets.py`, `tools/apply_review_corrections.py` e `tools/build_training_dataset.py`.

## Estado conhecido dos dados e modelos

Inventário auditado em 2026-10-31; os caminhos sob `captures/` são dados locais e podem não estar versionados no Git:

- **19 máscaras de consenso:** 1280×720, de uma sessão, split `train`; consenso de três agentes, aceito pelo usuário para o fluxo. Contêm 107.683 pixels de tinta e 507.339 pixels incertos (`255`). O manifesto as descreve como consenso de agentes, não ground truth humano. Já foram usadas no piloto expert-consensus com dados sintéticos.
- **719 pseudo-máscaras reais:** duas sessões já destinadas ao treino; 707 têm pixels supervisionados e 12 são totalmente ignoradas. Houve triagem em miniaturas e correção de erros grandes, mas as bordas finas não foram revisadas em resolução original. São úteis para treino exploratório, não referência independente.
- **60.187 máscaras propostas automaticamente:** geração concluída, mas todas continuam sem revisão e sem aprovação para treino.
- **180 propostas inspecionadas em resolução completa por agentes:** 0 passaram sem correção; as 180 precisam de correção.
- **Fila de anotação humana:** 20 itens ainda `not_started`; nenhum foi corrigido pelo fluxo humano em resolução completa. Não há holdout real independente aprovado.
- **Kaggle:** o SegFormer base foi treinado com dados sintéticos e 719 pseudo-máscaras; a validação reportada era sintética. O piloto expert-consensus usou dados sintéticos e 19 máscaras de consenso, também avaliado em validação sintética. Esses pilotos demonstram que o treino funciona, não a precisão real independente.
- **Kernel pair v6:** na última consulta read-only de 2026-10-31 aparecia `RUNNING`, sem logs nem outputs disponíveis. O código descrevia EMA/consistência semissupervisionada como projetada, mas não executada por falta de pares reais não rotulados antes/depois. Reconsultar antes de usar esse estado.

## Ciclo operacional

### 1. Capturar

- Iniciar uma sessão nova com ID único; não misturar takes nem sobrescrever sessões anteriores.
- Preservar os JPEGs originais. Manter `frames.csv` e um `take.json` com sessão/take, câmera e resolução, início/fim, configuração de captura, versão do app e observações sobre pausas ou eventos.
- Calcular SHA-256 dos originais e manter a relação entre ID do frame, arquivo e timestamp monotônico. O timestamp ajuda a ordenar frames; não prova que a câmera não perdeu frames.
- Durante a inferência, registrar também versão/hash do modelo, configuração de entrada, latência e caminho/hash da máscara proposta. A pontuação do modelo é uma sugestão, não uma confiança calibrada.
- Não mudar pesos durante a gravação. A captura continua se o modelo estiver desligado ou falhar.

### 2. Formar a fila de revisão

- Gerar propostas de máscara para os frames capturados, preservando imagem e proposta originais.
- Priorizar exemplos com incerteza do modelo, desacordo entre avaliadores, mudanças de cena e condições pouco representadas. Incluir amostra aleatória para medir erros que o seletor não destacou.
- Registrar candidatos sem os chamar de `ground truth`. Não propagar uma máscara para frames vizinhos como se tivesse sido conferida.

### 3. Revisar com agentes e escalar dúvidas

- Pelo menos dois revisores examinam a imagem original em resolução plena e a máscara, independentemente antes de comparar respostas.
- Cada revisão informa classes, áreas incertas, erros encontrados e máscara corrigida quando possível. Não usar só miniaturas para aprovar bordas de tinta.
- Um terceiro revisor adjudica desacordos. Até o processo passar por calibração contra máscaras humanas, consenso de agentes é apenas **candidato revisado**, não aprovação automática para treino.
- Encaminhar ao usuário os casos em que a tinta esteja ambígua, fina, ocluída, misturada com texto impresso, ou em que os revisores discordem. Mesmo após calibração, manter amostra aleatória para auditoria humana; concordância pode estar errada.
- Guardar separadamente rótulos de cena e máscaras pixel a pixel. Uma descrição como “mão escrevendo” não define os pixels de tinta.

### 4. Fechar um snapshot de treino

Antes de treinar, gerar um manifesto versionado que ligue, por frame: ID, take/sessão, caminho e SHA-256 da imagem, dimensões, caminho e SHA-256 da máscara, origem da máscara, revisores, decisão, split e contagens de pixels por classe e `255`.

- Treino, validação e teste são separados por sessão/take e linhagem; frames vizinhos e derivados ficam no mesmo split.
- Pseudo-rótulos, consenso de agentes e máscaras humanas aparecem como grupos distintos nos relatórios.
- Só entram no treino máscaras aprovadas pelo gate de dados definido para aquela rodada. Nunca converter `255` em fundo ou papel.
- Congelar os IDs, hashes, versão da taxonomia, configuração, seed e checkpoint inicial antes de ler o teste.

### 5. Treinar nova versão no Kaggle

- Treinar um candidato SegFormer com o snapshot aprovado e versionado. O plano preferencial é iniciar do último checkpoint PyTorch aprovado e registrar esse hash; se o treinador não suportar resume de forma reproduzível, usar o backbone pré-treinado fixo e documentar que é uma nova execução, não uma continuação.
- Dados sintéticos e reais devem ser relatados separadamente. Não treinar com previsões automáticas sem revisão.
- Não disparar kernels nem publicar datasets automaticamente. O usuário pediu que qualquer execução/envio ao Kaggle espere um pedido explícito.
- Guardar código/configuração, manifesto/splits, hashes dos pesos e ONNX, versões de dependências, logs, métricas por amostra e artefatos do Kaggle.

### 6. Avaliar e promover

Manter um teste real independente, anotado em resolução plena e não usado para selecionar o modelo. As sessões atuais destinadas ao treino não são esse holdout.

Reportar, no mínimo:

- IoU e Dice de tinta estrita; precisão/recall por classe; suporte e pixels ignorados;
- tolerância espacial para bordas de traço, separada da IoU estrita;
- falsos positivos em frames sem tinta, texto impresso, mão cobrindo o traço, tinta fina e diferentes condições de luz;
- resultados por frame e sessão, sem tratar frames vizinhos como sessões independentes;
- latência p50/p95 e taxa de frames processados no computador/câmera de destino.

Validar o ONNX separadamente do desempenho semântico: carregar no runtime CPU, conferir contrato e saídas finitas e comparar PyTorch/ONNX. Só promover se os critérios forem definidos antes de abrir o teste, a qualidade real não regredir e a latência couber no uso ao vivo. Manter o modelo anterior disponível para rollback.

### 7. Fechar a rodada

Publicar um relatório local com decisão `promote`, `revise` ou `inconclusive`, erros e limites. Se promovido, fixar o hash do modelo usado na próxima captura. O próximo ciclo começa com a versão registrada, não com autoatualização invisível.

## Estados recomendados para cada máscara

`proposal_unreviewed` → `agent_reviewed_candidate` → `human_adjudicated` ou `needs_user_review` → `training_approved` / `rejected`.

Só `training_approved` entra no snapshot de treino. Aprovação para treino não torna a máscara parte do teste. Uma decisão de cena não aprova uma máscara; revisão de máscara não aprova o modelo.

## Critérios para chamar o processo de melhoria

- Todo frame e máscara têm ID e hashes rastreáveis.
- Não há mistura de sessões/linhagens entre treino e teste.
- Nenhuma proposta automática não revisada vira rótulo.
- O holdout real é independente, revisado em resolução plena e aberto uma vez por candidato congelado.
- A nova versão supera os limites de qualidade e latência fixados antes da avaliação.
- O modelo anterior permanece disponível até a promoção ser confirmada.

Até esses critérios serem atingidos, chamar o sistema de **ciclo experimental assistido por agentes**, não de autoaperfeiçoamento autônomo nem de modelo validado.
