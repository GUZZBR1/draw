# Segmentação neural v1 do Draw — background / paper / hand / ink

> **Contexto atual:** o app tem um caminho experimental opt-in para o ONNX
> single-frame em `draw/segmentation.py`, chamado por
> `--segmentation-model`. `draw/neural.py` e `--neural` continuam sendo o
> detector legado da folha. Segmentação visível e detecção temporal são
> etapas distintas. Consulte [`INDEX.md`](INDEX.md) para navegar os documentos.

Modelo de segmentação semântica com 4 classes, treinado no Kaggle,
para o pipeline do Draw:

```text
frame
  ↓
SegFormer-B0 (transfer learning)
  ↓
background / paper / hand / ink
```

No Draw, quando `--segmentation-model` for solicitado:

```text
paper      → pixel visível que pode votar para limpar tinta
hand       → pixel desconhecido/ocluído; não adiciona nem apaga tinta
ink        → evidência de tinta visível neste frame
background → pixel desconhecido; não vota no acumulador
```

Quando a mão cobrir o desenho, o modelo retorna `hand` — nunca
"enxerga através da mão". A memória temporal do Draw, que preserva
o último estado confiável do traço até a mão sair, é trabalho
futuro, fora daqui.

## Foco deste projeto

Exclusivamente:

```text
câmera
→ identificar a folha
→ identificar a mão
→ identificar o traço
→ manter o desenho digital coerente
```

Não considerar CAD, simuladores ou integrações futuras neste
projeto. O escopo do Muse aqui é só o modelo e a sua documentação.

## Escopo do Muse (só isto)

```text
dados
→ geração sintética
→ treinamento
→ validação
→ exportação ONNX
→ documentação
```

## O que está explicitamente FORA (e quem fará depois)

Os itens abaixo estavam fora do escopo original de treino e validação
registrado neste documento. O estado atual do app e da integração está no
README raiz e no índice desta pasta:

- webcam
- integração do modelo no aplicativo (`draw/neural.py`, `draw/app.py`)
- tracking temporal
- memória dos traços
- lógica de oclusão em tempo real
- interface
- CAD
- simuladores

[`INTEGRATION.md`](INTEGRATION.md) documenta o contrato de saída do modelo
single-frame; ele não descreve o modelo temporal pareado.

## Classes

| Índice | Classe | Significado |
|---|---|---|
| 0 | `background` | Tudo que não é folha, mão ou traço |
| 1 | `paper` | Papel visível |
| 2 | `hand` | Mão visível |
| 3 | `ink` | Traço visível |

Regra de rótulo, sem exceção: a segmentação representa o que a
câmera vê. Mão por cima de traço = `hand`, nunca `ink`. A máscara
do desenho completo (`ink_full_mask`, com o traço inclusive onde
a mão o esconde) é um artefato opcional da geração sintética,
**não** é classe de treinamento e **não** entra no ONNX — servirá
no futuro para avaliar a memória do Draw.

## Entregáveis prioritários (obrigatórios)

1. **Gerador sintético reproduzível** — mãos FreiHAND + folhas +
   fundos/mesas + traços + augmentations, com seeds reproduzíveis,
   e splits feitos **antes** da geração (fontes originais divididas
   entre train/val/test, para que variações da mesma mão, folha ou
   fundo não vazem entre conjuntos).
2. **Notebook Kaggle reproduzível** — SegFormer-B0 com transfer
   learning, entrada RGB 512×512, saída de 4 classes.
3. **Configuração completa do treinamento** (`training_config.json`).
4. **Definição dos splits** (`split_manifest.json`, com seeds e IDs
   das fontes por conjunto; sugestão inicial 80/10/10).
5. **Métricas por classe** (`metrics.json`) — IoU por classe,
   Dice de paper/hand/ink, precision e recall. Nunca só accuracy
   global.
6. **Métrica de ink com tolerância** — precision/recall de traço
   com tolerância espacial de 2–3 px (dilatação, distance transform
   ou esqueleto), porque traço fino deslocado poucos pixels tem
   IoU artificialmente baixa mesmo estando visualmente certo.
7. **Avaliação no holdout real** — pequeno conjunto de frames reais
   do Draw, manualmente anotado, separado, nunca usado no treino
   nem em augmentation. Métricas sintéticas e reais reportadas
   **separadamente**; queda grande no real é domain gap a
   documentar, não a esconder.
8. **Amostras visuais** — bateria fixa de pelo menos 15 cenários
   (folha limpa, folha com desenho, folha muito desenhada, mão
   cobrindo canto/centro/traço, mão parcialmente fora, fundo
   claro/escuro, sombra forte, folha parcialmente fora, mão sem
   folha, objeto branco sem folha, traço fino de lápis, traço
   grosso de caneta), com predições salvas para comparação entre
   versões.
9. **Modelo ONNX** — `draw_segmentation_v1.onnx`, validado contra
   o modelo original (mesmos frames, saídas comparadas, sem
   degradação significativa).
10. **Documentação da interface ONNX** — [`INTEGRATION.md`](INTEGRATION.md):
    input, shape, normalização, outputs, ordem das classes,
    thresholds, extração de cada máscara e `paper_presence`.
11. **Documentação de compatibilidade com `draw/neural.py`** —
    também em [`INTEGRATION.md`](INTEGRATION.md): conversão de
    `paper` para o contrato atual (máscara → maior contorno →
    quadrilátero) e a lista do que o Codex precisará mudar depois.

## Decisão sobre `paper_presence` (resumo)

Documentada em detalhe em `INTEGRATION.md`, seção 4:

**`paper_presence` é derivado da máscara de `paper`** — fração de
pixels paper ≥ 0,01 **e** quadrilátero que fecha. **Não há cabeça
auxiliar de presença no ONNX v1.** Motivo: uma única fonte de
verdade, sem risco de presença e máscara discordarem. A
compatibilidade com o contrato antigo não justifica um segundo
sinal: **o modelo não é simplificado para manter o contrato
antigo; o contrato é adaptado ao modelo.**

## Dataset sintético (regra de tamanho)

Não é obrigatório entregar todo o dataset gerado se isso
ultrapassar limites razoáveis de armazenamento. Nesse caso,
entregar gerador + seeds + configuração + manifest + amostra
representativa, de forma que o dataset completo possa ser
regenerado.

## Dados e licenças (a verificar antes do treino)

- **FreiHAND** (Kaggle, `danieldelro/freihand`): verificar antes
  do pipeline completo se contém imagens RGB utilizáveis e
  máscaras reais de segmentação da mão, formato, quantidade,
  resolução, licença e termos. Se não contiver máscaras adequadas,
  **não inventar rótulos**: documentar o problema e usar fonte
  alternativa apropriada, registrando licença e origem.
- **Frames reais do Draw**: fornecidos pelo Codex, manualmente
  anotados, só para holdout. O Muse não captura câmera.
- Negativos obrigatórios no sintético: mesa vazia, mão sem folha,
  objetos brancos, retângulos claros, livro, caixa, monitor,
  superfícies claras, folha muito pequena, folha praticamente
  fora — para o modelo não aprender que "branco ou retangular
  = paper".

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `README.md` | Este documento: escopo, entregáveis, o que está fora |
| `INTEGRATION.md` | Interface ONNX v1 e compatibilidade com `draw/neural.py` |

Os arquivos do modelo (`draw_segmentation_v1.onnx`,
`classes.json`, `metrics.json`, `training_config.json`,
`split_manifest.json`) são entregues pelos agentes de dados e
treino, não por este documento.

## Estado

Este documento começou como plano de treino e interface. A integração atual
é opt-in e usa `draw/segmentation.py`; ela não substitui o detector legado.
O modelo foi treinado com dados sintéticos, portanto a integração técnica não
comprova qualidade em câmera real. Use [`INDEX.md`](INDEX.md) e os relatórios
para o estado e as limitações atuais.
