# Integração do modelo de segmentação v1 (4 classes) com o Draw

> Contrato do modelo e do caminho experimental opt-in.
> A implementação está em `draw/segmentation.py` e `draw/app.py`.
> Ela não substitui o detector legado `--neural` e não detecta traços
> novos entre frames.

## 1. O que muda e o que não muda

### Contrato atual (fase 2, em `draw/neural.py`, base `e2c866f`)

- Entrada: frame da câmera, esmagado para **512×512** (sem letterbox),
  convertido de BGR para RGB, dividido por 255 e normalizado com a
  média e o desvio do ImageNet.
- Saída ONNX: dois tensores, `mask_logits` com **2 canais**
  (background, paper) e `presence_logit` (1 valor).
- Decisão: presença positiva (`presence_logit > 0`) **e** quadrilátero
  que fecha. Sem um dos dois, o resultado é `(None, None)`.
- Máscara: probabilidade do canal paper (softmax de 2 canais) com
  limiar **0,5**, redimensionada para o tamanho do frame com
  `INTER_NEAREST`, em uint8 0/255.
- Quadrilátero: maior contorno da máscara, área mínima de 1% do frame,
  escada de epsilon do detector clássico (`draw/paper.py`),
  reajuste de cada aresta como reta sobre os pontos do contorno
  (deslocamento máximo aceito de 100 px), refino subpixel só nos
  cantos dentro do quadro, e exigência de quadrilátero convexo.
  Tudo isso já existe em `neural.py:fit_quad` e **não deve ser
  reescrito** — deve ser reutilizado.
- Sessão ONNX preguiçosa: `onnxruntime` só é importado no primeiro
  uso; sem ele, erro explícito. O caminho do modelo chega por
  argumento ou pela variável `DRAW_NEURAL_MODEL`. O binário fica
  fora do repositório.

### Contrato novo (v1, 4 classes, `--segmentation-model`)

- O modelo passa a segmentar **o que a câmera vê, pixel a pixel**,
  em 4 classes. Nada aqui é "o que está por baixo da mão" — ver
  a regra de rótulo visível na seção 6.
- A única saída do ONNX v1 é a segmentação. **Não há `presence_logit`
  no ONNX v1.**
- O modelo não é usado para localizar a folha nesta integração. O
  detector clássico ou o detector legado `--neural` continua responsável
  pelos cantos; a nova opção fornece a máscara visível de tinta.

## 2. Input do ONNX v1

| Item | Valor |
|---|---|
| Nome do input | `image` (manter o mesmo nome do contrato atual) |
| Formato | RGB, não BGR. O frame vem da câmera em BGR (OpenCV) e **deve** ser convertido com `cv2.COLOR_BGR2RGB` antes de entrar no modelo |
| Redimensionamento | Esmagado para **512×512** com `INTER_LINEAR`, exatamente como hoje. Sem letterbox, sem crop, sem preservar aspecto. Isso distorce a proporção da imagem, e é assim que o modelo é treinado e avaliado — mudar isso na integração quebraria o modelo |
| Escala | Dividir por 255,0 (float32) |
| Normalização | ImageNet, igual à atual: média `[0.485, 0.456, 0.406]`, desvio `[0.229, 0.224, 0.225]` |
| Layout | NCHW, float32, batch 1 |
| Shape final | `[1, 3, 512, 512]` |

Ordem das operações, sem atalho: BGR → RGB → resize 512×512 →
`/255` → `(x - mean) / std` → transpose para CHW → batch.

## 3. Outputs do ONNX v1 e ordem das classes

| Item | Valor |
|---|---|
| Nome do output | `logits` (único output do ONNX v1) |
| Shape do ONNX atual | `[1, 4, 128, 128]`, float32, logits (não probabilidades) |
| Pós-processamento | redimensionar logits bilinearmente para 512×512 (como no avaliador do treino) e depois aplicar `argmax` sobre os canais |

Ordem das classes, fixa e documentada em `classes.json`:

```json
{
  "0": "background",
  "1": "paper",
  "2": "hand",
  "3": "ink"
}
```

**Não reordenar, não renomear, não remover canal.** Qualquer modelo
que troque essa ordem é um modelo diferente e precisa de uma nova
versão do documento, não de um ajuste silencioso na integração.

## 4. Proposta original de `paper_presence` (não implementada)

Esta seção descreve a proposta anterior de também derivar a presença e
os cantos da classe `paper`. Ela foi substituída pela separação de
responsabilidades: este ONNX segmenta pixels; o detector clássico ou o
detector legado `--neural` localiza a folha. A classe `paper` continua
disponível em `predict_masks`, mas não decide presença no caminho atual.

### Decisão documentada

**`paper_presence` no v1 é derivado da máscara de `paper`. Não há
cabeça auxiliar de presença no ONNX v1.**

Definição exata, para o Codex implementar:

```text
paper_prob   = softmax(logits, dim=canais)[:, 1, :, :]   # canal paper
paper_mask   = argmax(logits, dim=canais) == 1            # 512×512, bool
paper_fraction = paper_mask.mean()   # fração de pixels paper, entre 0 e 1
quad         = fit_quad(paper_mask em 0/255, frame)
paper_presence = 1  se paper_fraction >= 0,01 e quad is not None
                 0  caso contrário
```

O limiar de 0,01 (1% do frame) é o mesmo da área mínima de contorno
já usada em `fit_quad` (`_MIN_CONTOUR_AREA_RATIO`). Não é um número
novo: é o mesmo critério, aplicado duas vezes de forma coerente
(fração global e maior contorno).

Além do booleano, a integração deve expor `paper_fraction` como
escore contínuo de confiança da presença (útil para logs, debug e
para o harness de avaliação). O booleano é a decisão; a fração é a
medida.

### Prós e contras considerados

**Derivado da máscara (escolhido):**

- Prós: uma única fonte de verdade. Presença e máscara nunca
  discordam (não existe "o modelo diz que há papel, mas a máscara
  está vazia", nem o inverso). ONNX mais simples, com um único
  output. Comportamento determinístico e reproduzível a partir da
  segmentação. O limiar é interpretável (fração de papel visível).
- Contras: se a máscara falhar, a presença falha junto — não há
  um segundo sinal para resgatar. Uma folha muito pequena ou muito
  oclusa pode ficar abaixo de 1% mesmo estando presente; esse é
  um limite conhecido, e a resposta correta é melhorar a
  segmentação, não inventar um segundo sinal que a contradiga.

**Cabeça auxiliar (rejeitada para o v1):**

- Prós: um sinal independente, que poderia resgatar casos em que a
  máscara falha, e que reproduz o formato do contrato atual
  (`presence_logit`) sem mudar a assinatura.
- Contras: dois sinais que podem divergir, e aí a integração
  precisa de uma regra de desempate arbitrária (qual vale quando
  discordam?). Treino mais complexo (peso da loss auxiliar,
  calibração do logit). Risco de a cabeça aprender um atalho
  ("há algo branco e retangular") que a máscara, mais exigente,
  não aprendeu — exatamente o falso positivo que os negativos
  do dataset existem para evitar.
- Veredito: a compatibilidade com o formato antigo não justifica
  um segundo sinal que pode contradizer o primeiro. **O modelo
  não é simplificado para manter o contrato antigo; o contrato
  é adaptado ao modelo.** Se, com métricas em mãos, o derivado
  se mostrar insuficiente, uma cabeça auxiliar pode ser avaliada
  como v2, como output adicional, sem quebrar a segmentação —
  nunca como substituto silencioso da máscara.

## 5. Thresholds da proposta original de localização da folha

Os thresholds para `paper_presence` abaixo pertencem à proposta
histórica da seção 7a. O caminho experimental atual não deriva cantos
nem presença de folha; ele usa diretamente o `argmax` categórico e não
introduz threshold de confiança.

| Onde | Valor | Por quê |
|---|---|---|
| Máscara de paper (para o quadrilátero) | `argmax == 1` | Em 4 classes, o argmax é a decisão do modelo. É o equivalente direto do `prob > 0,5` do modelo de 2 canais (lá, prob > 0,5 implicava argmax) |
| Máscara confiante (opcional, para debug/visualização) | `argmax == 1` **e** `prob_paper >= 0,5` | Mostra só os pixels em que o modelo está seguro; não usar para o quadrilátero, para não encolher a folha |
| `paper_presence` | `paper_fraction >= 0,01` **e** quadrilátero fecha | Mesmo critério de área mínima já usado em `fit_quad` |
| Máscaras de hand e ink | `argmax == 2` e `argmax == 3` | Sem limiar adicional no v1; se a calibração futura pedir, documentar como nova versão |
| Quadrilátero | regras atuais de `fit_quad`, sem mudança | Maior contorno, área ≥ 1%, escada de epsilon, refit de arestas (≤ 100 px), subpixel só dentro do quadro, convexo |

Não criar threshold novo na integração sem registrar aqui o motivo
e a métrica que o justifica.

## 6. Como extrair cada máscara

O ONNX atual emite logits 128×128. O consumidor redimensiona cada
canal bilinearmente para 512×512 antes do `argmax`, como faz o avaliador
do treino. O mapa categórico resultante é redimensionado para o frame
original com `INTER_NEAREST`, preservando IDs inteiros sem criar classes
intermediárias. `predict_classes` retorna IDs `uint8` 0..3;
`predict_masks` retorna quatro máscaras booleanas nomeadas.

```text
logits_512 = resize_bilinear(logits, 512, 512)     # [1, 4, 512, 512]
seg        = argmax(logits_512, dim=canais)        # [512, 512], valores 0..3
paper_mask = (seg == 1)   # folha visível, bool
hand_mask  = (seg == 2)   # mão visível, bool
ink_mask   = (seg == 3)   # traço visível, bool
# background = (seg == 0), ignorado pelo Draw
# IDs: resize para (frame_w, frame_h), INTER_NEAREST
```

Pontos importantes:

- As classes ficam alinhadas ao frame original antes do warp para
  coordenadas do papel. O app aplica a homografia já fornecida pelo
  detector de folha, usando `INTER_NEAREST`.
- `hand_mask` e `ink_mask` são saídas single-frame. O app usa tinta
  visível como observação e mão/fundo como pixels desconhecidos; a
  memória temporal continua no `InkAccumulator`.
- As máscaras são mutuamente exclusivas por construção (argmax):
  um pixel é exatamente uma classe. Isso é o que implementa a
  regra de rótulo visível (abaixo).

### Regra de rótulo visível (obrigatória, sem exceção)

A segmentação representa **o que a câmera vê**, não o que está
por baixo:

```text
papel visível → paper
traço visível → ink
mão visível   → hand
restante      → background
```

Se a mão cobre um traço, o pixel final é `hand`, **não** `ink`.
O modelo não deve ser treinado para "enxergar através da mão".
A máscara do desenho completo (`ink_full_mask`, com o traço
inclusive onde a mão o esconde) é um artefato separado da geração
sintética, **não** é classe de treinamento, **não** entra no ONNX
e **não** deve ser usada como rótulo. Ela servirá, no futuro,
para avaliar o sistema de memória do Draw — fora deste escopo.

## 7. Integração atual do app

O runtime está em `draw/segmentation.py` e é ativado explicitamente por
`--segmentation-model MODELO.onnx` ou `DRAW_SEGMENTATION_MODEL`. A flag
`--neural` e `draw/neural.py` permanecem no contrato legado de detecção
de folha; não passe o ONNX de quatro classes a essa flag.

No modo experimental, o app infere no frame da câmera e transforma o
mapa de classes para coordenadas do papel com vizinho mais próximo.
Classe 3 (`ink`) vira evidência de tinta; classes 1 (`paper`) e 3 são
pixels visíveis. Classes 2 (`hand`) e 0 (`background`) são desconhecidas
e não podem adicionar tinta nem votar para apagá-la. A observação de um
único frame vai para o `InkAccumulator` existente, que continua sendo a
memória temporal. Sem a flag, o app mantém a extração clássica. Uma falha
durante a inferência é avisada e usa a extração clássica naquele frame;
falha de dependência/contrato interrompe o carregamento explicitamente.

O caminho está desligado por padrão e é exploratório. O smoke sintético
valida execução e contrato, não a qualidade em câmera real ou latência ao
vivo. Não houve ativação da câmera.

Verificação desta integração: suíte completa com OpenCV contrib 4.10.0,
ONNX Runtime 1.20.1: 203 testes passaram e 1 foi ignorado. O smoke ONNX
usou somente imagens sintéticas do split de treino, sem rótulos ou métricas.

## 7a. Proposta original de conversão paper → detector (substituída)

O texto abaixo preserva o plano anterior de reutilizar a saída de paper
para mudar o detector de folha. Ele não descreve o comportamento atual.
Os passos e checklists nesta subseção estão arquivados como histórico,
não são pendências da integração atual. Não execute o harness real/holdout
proposto aqui para validar ou ajustar este modelo sintético.

Passo a passo, reutilizando o que já existe:

1. Rodar o ONNX v1 no frame (seção 2) e obter `logits`
   `[1, 4, 128, 128]` no artefato atual.
2. Redimensionar logits bilinearmente para 512×512 e aplicar `argmax`.
3. Redimensionar `paper_mask` para o tamanho do frame com
   `INTER_NEAREST`, em uint8 0/255.
4. Chamar `fit_quad(máscara, frame)` — o mesmo de hoje, sem
   reescrever: maior contorno, escada de epsilon, refit de
   arestas, subpixel dentro do quadro, convexo.
5. Calcular `paper_fraction` e `paper_presence` pela definição
   da seção 4.
6. Devolver, no contrato atual, `(cantos, "paper")` quando
   `paper_presence == 1`, e `(None, None)` caso contrário —
   exatamente como `detect_sheet_with_confidence` faz hoje.
7. Expor, em adição ao contrato atual (sem quebrá-lo), as
   máscaras novas: um método novo (sugestão: `predict_masks(frame)`)
   devolvendo `paper_mask`, `hand_mask`, `ink_mask` e
   `paper_fraction`, nas coordenadas do frame. O método antigo
   continua existindo e continua devolvendo só `(cantos, veredito)`.

### O que o Codex precisará mudar em `draw/neural.py` (lista, sem implementar aqui)

- [ ] `predict`: ler **um** output (`logits`, 4 canais) em
      vez de dois; remover a leitura de `presence_logit`; trocar
      o softmax de 2 canais por softmax + argmax de 4 canais.
- [ ] Extrair `paper_mask`, `hand_mask` e `ink_mask` do argmax,
      com resize `INTER_NEAREST` para o tamanho do frame.
- [ ] Calcular `paper_presence` pela definição da seção 4
      (fração ≥ 0,01 e quadrilátero fecha), em vez de
      `presence_logit > 0`.
- [ ] Manter intactos, sem tocar: `fit_quad`, `_approx_quad`,
      `_refit_corners`, `_refine_inside`, a sessão preguiçosa,
      o erro explícito sem `onnxruntime`, o caminho por argumento
      ou `DRAW_NEURAL_MODEL`, e o fallback para o detector
      clássico quando o neural não vê folha (isso está em
      `draw/app.py`, não aqui, e continua valendo).
- [ ] Manter o input exatamente como na seção 2 (nome `image`,
      512×512 esmagado, ImageNet). Qualquer divergência aqui é
      bug de integração, não "ajuste".
- [ ] Atualizar o docstring do módulo para o contrato v1 e
      apontar para este documento.
- [ ] Validar com o harness de avaliação (`tools/eval_real_frames.py`)
      antes de considerar a integração pronta, e comparar as
      métricas com o modelo da fase 2 — sem essa comparação, a
      integração não está validada.

### O que o Codex NÃO fará neste passo (fora do escopo dele aqui também)

- Não mudar o app, a webcam, o tracking temporal, a memória dos
  traços, a lógica de oclusão em tempo real, a interface, CAD ou
  simuladores. A integração é só a troca do detector neural pelo
  modelo v1, atrás do mesmo contrato de cantos + veredito, mais
  as máscaras novas expostas.

## 8. Arquivos da entrega v1 (referência para a integração)

Os arquivos abaixo são entregues por outros agentes deste projeto
(treino, dados). A integração deve esperá-los com exatamente estes
nomes e formatos; se algum vier diferente, isso é divergência a
resolver, não detalhe a improvisar na hora:

| Arquivo | Conteúdo |
|---|---|
| `draw_segmentation_v1.onnx` | Modelo v1, input `image` `[1,3,512,512]`, output atual `logits` `[1,4,128,128]` |
| `classes.json` | `{"0": "background", "1": "paper", "2": "hand", "3": "ink"}` |
| `metrics.json` | Métricas por classe (IoU e Dice por classe, precision, recall) e métrica de ink com tolerância espacial |
| `training_config.json` | Arquitetura, entrada, loss, seeds, hiperparâmetros — tudo para reproduzir o treino |
| `split_manifest.json` | Quais fontes originais (mãos, folhas, fundos) foram para train/val/test, com seeds |

Antes de integrar, validar o ONNX: rodar o modelo original e o ONNX
nos mesmos frames e comparar as saídas; registrar a comparação.
Conversão que degrade a segmentação não é entregue.

## 9. Limites conhecidos desta interface

- O modelo vê o frame esmagado em 512×512: traço muito fino pode
  sumir no downscale. A métrica de ink com tolerância (2–3 px)
  existe por isso, e a integração não deve "corrigir" traço fino
  com limiar próprio.
- Folha muito pequena ou muito oclusa pode ficar abaixo de 1% de
  pixels paper e, portanto, com `paper_presence == 0`, mesmo
  estando parcialmente visível. Isso é o comportamento definido,
  não um bug a contornar na integração.
- `hand_mask` e `ink_mask` são do frame, não da folha retificada.
  Quem precisar delas na folha precisa aplicar a homografia do
  quadrilátero — fora do modelo, fora deste documento.
- Este documento descreve a interface. Métricas, amostras e o
  modelo em si são entregues pelos outros agentes e documentados
  no README desta pasta.
