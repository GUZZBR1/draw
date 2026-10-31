# Resultado da validação do motor de triagem — 2026-10-30

## Resultado

O motor completou o inventário e a auditoria cega, mas **falhou como filtro para descartar frames**. Ele pode ser usado para ordenar uma fila de revisão, desde que todos os originais continuem preservados e frames de baixa prioridade não sejam tratados como negativos.

## Execução e integridade

- O classificador local processou **15.845 frames** em **277,4 s**: 2.518 imagens 640×480 e 13.327 imagens 1280×720.
- A seleção levou cerca de **47 s** e criou uma fila de **644 frames**. Nenhum arquivo fonte foi alterado ou removido.
- A verificação encontrou 15.845 IDs únicos, nenhum caminho ausente e nenhum hash divergente.
- O passe local recalculou os sinais OpenCV/pHash. Como PyTorch/OpenCLIP não estão instalados neste ambiente, os 10 campos semânticos OpenCLIP foram reutilizados do inventário anterior. Os 15.845 IDs e hashes coincidiram integralmente com o inventário que originou esses scores.
- A preparação do pacote de auditoria levou cerca de **4 s**. A execução completa desta rodada, incluindo revisores, terminou abaixo de cinco horas; o tempo total de ponta a ponta não foi gravado como uma métrica única.

## Auditoria cega

Três revisores visuais independentes analisaram o mesmo pacote embaralhado, sem scores, prioridades ou IDs de origem. Cada um reportou que **183/183 frames pareciam úteis** para segmentação.

| Sessão | Alta prioridade | Baixa prioridade |
|---|---:|---:|
| 640×480 | 21/21 úteis | 60/60 úteis |
| 1280×720 | 42/42 úteis | 60/60 úteis |

Na sessão 640×480 havia só 21 grupos de alta prioridade elegíveis; na sessão 1280×720, 42. O motor revisou todos esses grupos. A baixa prioridade foi amostrada em 60 grupos por sessão. O acordo das três avaliações para o rótulo `useful` foi 100%.

## Gates

- **Perdas na baixa prioridade:** falhou. Todos os 120 frames amostrados da fila baixa foram julgados úteis ou potencialmente úteis.
- **Enriquecimento da fila alta:** falhou. A diferença ponderada alta–baixa foi 0 ponto percentual; IC bootstrap de 95% `[0, 0]`, abaixo do mínimo aprovado de 20 pontos.
- **Eventos raros:** sem suporte confirmado por dois revisores por imagem; não há base para declarar recall de tinta, oclusão ou transição.
- **Status final:** `fail` para filtragem automática. Isso não invalida o inventário nem a utilidade do ranking como auxílio exploratório.

## Limites

As decisões são julgamentos visuais de utilidade para segmentação, não máscaras pixel a pixel nem ground truth humano. A amostra é limitada às duas sessões existentes. O resultado indica que o ranking atual não separa frames dispensáveis dos úteis o bastante para justificar descarte ou exclusão do treinamento. Para reduzir o volume com segurança, o classificador e os limiares precisam ser ajustados e passar uma nova auditoria cega.

Os arquivos de imagens cegas e o mapeamento privado permanecem locais em `captures/triage/microengine-run/`; nenhum frame bruto, score individual ou mapping de origem é publicado por este relatório.
