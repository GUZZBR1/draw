# Motor local de triagem de frames

## Decisão do Senado

Implementar como **motor de priorização de revisão**, não como filtro que apaga frames ou determina automaticamente que um frame é inútil. Os sinais OpenCV, pHash e OpenCLIP são propostas. A amostra cega mede se a fila ajuda a encontrar exemplos úteis sem ocultar eventos importantes.

O Senado aprovou essa arquitetura para implementação de forma condicional. A qualidade real da fila e o tempo total do fluxo ainda precisam passar pelos gates abaixo.

## Fluxo

1. **Inventário imutável:** registrar cada frame por `(session_id, frame_index)`, caminho original, dimensões, SHA-256, tempo/take ou segmento e estado de decodificação.
2. **Pontuação local:** combinar qualidade da imagem, folha candidata, pHash e, opcionalmente, scores semânticos OpenCLIP. Os scores não são probabilidades calibradas.
3. **Fila reversível:** escolher âncoras, mudanças visuais, categorias raras e exemplos de baixa confiança. Dar prioridade menor a candidatos parecidos, mantendo-os no manifesto e no overflow.
4. **Auditoria cega:** selecionar exemplos de alta e baixa prioridade sem mostrar score, categoria prevista, status da fila ou ID de origem aos revisores. Três agentes visionais marcam utilidade e eventos de forma independente.
5. **Máscaras propostas:** gerar rascunhos somente para imagens selecionadas e com tags visuais disponíveis. Pixels ambíguos continuam `255`; uma proposta não é rótulo.
6. **Revisão de máscara e pacote:** somente máscaras aprovadas por consenso de três agentes podem entrar como `agent_consensus`. Elas não são ground truth humano. A validação da fila e a validação das máscaras são evidências separadas.

## Como executar

Execute os comandos a partir da raiz do repositório. O classificador processa um frame por vez e grava apenas artefatos de triagem no diretório de saída:

```bash
python3 tools/classify_frames.py \
  --session captures/training/session_20261029_130013_453419 \
  --session captures/batches/batch_01/training/session_20261029_151126_089557 \
  --output captures/triage/run/classifier.csv \
  --takes-output captures/triage/run/takes.csv

python3 tools/select_review_frames.py \
  --input captures/triage/run/classifier.csv \
  --manifest-output captures/triage/run/selection_manifest.csv \
  --review-output captures/triage/run/review_queue.csv \
  --overflow-output captures/triage/run/overflow.csv \
  --summary-output captures/triage/run/selection_summary.json

python3 tools/frame_microengine.py sample-audit \
  --manifest captures/triage/run/selection_manifest.csv \
  --output-dir captures/triage/run/blind_audit \
  --seed 20261030

python3 tools/frame_microengine.py evaluate-audit \
  --mapping captures/triage/run/blind_audit/audit_mapping.private.csv \
  --reviewer-csv captures/triage/run/blind_audit/reviewer1.csv \
  --reviewer-csv captures/triage/run/blind_audit/reviewer2.csv \
  --reviewer-csv captures/triage/run/blind_audit/reviewer3.csv \
  --output captures/triage/run/blind_audit/audit_report.json
```

O arquivo `blind_packet.csv` e as cópias `images/` são para os revisores. O arquivo `audit_mapping.private.csv` revela quais exemplos são da fila alta ou baixa e deve permanecer separado. O resultado só autoriza priorização exploratória se todos os gates do relatório passarem; o status `fail` ou `inconclusive` não autoriza descarte.

Se houver uma lista local de frames já revisados com colunas `session_id,frame_index`, passe-a como `--reviewed-ids <arquivo.csv>` para evitar reavaliá-los. Essa lista é local e não faz parte do repositório.

O OpenCLIP é opcional: `classify_frames.py --clip` precisa de PyTorch e `open_clip_torch`; sem isso, o passe usa OpenCV/pHash. O `sample-audit` valida que o manifesto contém as duas sessões atuais completas (2.518 e 13.327 frames). Ao substituir ou adicionar uma sessão, atualize primeiro o contrato de inventário e seus testes.

## Prova cega da triagem

Congelar a versão e os scores do motor antes de escolher a amostra. Usar um seed registrado e estratificar por sessão e grupo de revisão:

- **Baixa prioridade:** 60 frames por sessão, escolhidos entre itens fora da fila primária, distribuídos por takes ou segmentos e faixas de score.
- **Alta prioridade:** 50 frames por sessão da fila primária, igualmente distribuídos por grupos.
- **Total inicial:** 220 frames, avaliados independentemente pelos três agentes. Excluir IDs já avaliados nas filas anteriores para evitar reciclar decisões visíveis aos agentes.

O pacote de revisão deve conter somente IDs cegos, cópias de imagem e campos vazios de decisão/tags. A tabela que liga ID cego ao frame original e às pontuações fica separada do pacote dos revisores.

Marcar `useful`, `not_useful` ou `uncertain`; registrar folha, mão, tinta do usuário, caneta, oclusão, transição e qualidade ruim. Para o gate de perda, qualquer voto `useful` ou `uncertain` conta conservadoramente como exemplo possivelmente útil. Desacordos não viram negativos.

### Critérios propostos

- **Integridade:** cada fonte aparece uma vez no manifesto, todos os caminhos existem, todos os hashes conferem e nenhum original é apagado.
- **Perdas da baixa prioridade:** zero casos `useful/uncertain` entre 60 amostras de cada sessão. Reportar o limite de confiança por sessão e a incerteza agrupada por take/segmento; não anunciar o limite binomial combinado de 2,5% para uma amostra estratificada.
- **Enriquecimento da fila alta:** diferença absoluta entre a taxa de exemplos `useful/uncertain` das filas alta e baixa com limite inferior de 95% acima de 20 pontos percentuais, calculado por bootstrap agrupado por take/segmento. Usar diferença, não razão alta/baixa, porque a taxa baixa pode ser zero.
- **Eventos raros:** só calcular recall de tinta, mão sobre tinta/folha ou transição se houver pelo menos 50 positivos adjudicados por categoria. Abaixo disso, declarar “suporte insuficiente”; nunca interpretar ausência na amostra como ausência na gravação.
- **Disposição:** falha em qualquer gate deixa a classificação como exploratória e amplia a fila; não causa exclusão automática de fonte.

## Máscaras e treinamento

- Avaliar as máscaras em amostra separada da amostra usada para desenhar, escolher ou treinar propostas.
- Exigir revisão por frame e consenso unânime dos três agentes para liberar uma máscara; divergência permanece incerta ou vai para adjudicação.
- Preservar `255` como desconhecido. O branch `training-zone` inicialmente rejeita `255`; a implementação deve portar suporte de ignore-index para validação, CE, Dice e métricas antes de empacotar máscaras parciais. Se esse suporte não passar seus testes, só máscaras densas com `{0,1,2,3}` poderão ser treinadas.
- Não alegar melhora em webcam sem uma sessão real independente com máscaras aprovadas. O piloto existente alterou o mIoU sintético de `0.917703` para `0.918241` e não demonstrou melhora prática.

## Tempo e privacidade

A varredura existente processou 15.845 frames em 845 segundos (cerca de 14 minutos), mas isso só prova o passe anterior, não o fluxo final nem a revisão dos agentes. Reexecutar no branch final e registrar duração, memória de pico, frames decodificados, linhas do manifesto e checksums. O objetivo de cinco horas vale para o fluxo completo e só poderá ser declarado se o tempo de revisão e adjudicação também ficar abaixo dele.

Os originais continuam locais e imutáveis. Pacotes enviados ao Kaggle devem permanecer privados; nenhum JPEG bruto entra em Git.

## Revisão adversarial

O risco mais forte é um frame parecer duplicado e conter um traço novo ou uma oclusão breve. pHash e CLIP só ordenam a fila. A amostra baixa e a regra de incerteza procuram expor esse erro. Ainda assim, a prova é limitada às sessões e à amostra: ela não demonstra que todos os frames são úteis, nem que qualquer classe rara foi coberta se não houver suporte suficiente.

O código e os resultados de validação serão registrados após a implementação. Este documento registra o desenho aprovado, não declara o motor validado em produção.
