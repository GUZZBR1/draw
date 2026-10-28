# Coleção de exemplos reais

Esta pasta é a régua do Draw fora do sintético. O selftest e a
verificação geométrica provam o pipeline em cenas geradas por código;
eles não cobrem iluminação real, desfoque, oclusão parcial nem webcam
de verdade. Esta coleção existe para medir isso, caso a caso, com
resposta esperada — nunca para servir de álbum de fotos.

## Regra número um: foto sem resposta esperada não é teste

Um caso só entra no manifesto se tiver, no mínimo:

- a foto original, sem edição, em `raw/` (nunca editada depois);
- `cases/<id>/truth.json` com os campos obrigatórios do schema
  (ver `manifest.json` e `tools/evaluate_dataset.py`);
- cantos da folha revisados por um humano, no padrão do
  `tools/label_frames.py` (overlay verde para conferência).

Foto sem `truth.json`, sem cantos revisados ou sem consentimento
não entra no manifesto. O avaliador recusa o manifesto nesses casos.

## Splits: dev calibra, test só mede

- `dev`: pode calibrar limiares. É aqui que se aprende com os erros.
- `test` (held-out): só mede, nunca calibra. Se um limiar for
  ajustado olhando um caso de `test`, esse caso deixa de valer e
  precisa ser substituído por outro, ainda não visto pelo ajuste.

Nunca misturar os splits num mesmo número agregado sem dizer qual é
qual: o relatório do avaliador separa `dev` e `test` sempre.

## O que fotografar

Variedade que o sintético não tem: luz natural e artificial, sombra
parcial, folha inclinada, papel branco, amarelado, pautado e
reciclado, caneta e lápis, câmera de celular e webcam, perto e longe.
Cada caso registra papel, iluminação, inclinação aproximada,
ferramenta e fonte da câmera no `truth.json`. Um caso que falha ganha
uma classe de falha no relatório (`detection`, `corner`, `ink`),
para o próximo ciclo saber o que corrigir — não basta saber que
falhou.

## O que NÃO fotografar

- Rostos, pessoas identificáveis, documentos pessoais, texto pessoal
  legível (cartas, cadernos pessoais, telas com dados).
- Mãos com identificadores desnecessários, quando evitável.
- Qualquer coisa de terceiros sem licença escrita registrada no
  manifesto.

Se a foto contém algo disso por acidente, ela não entra: apague o
arquivo e o caso.

## Consentimento, licença e privacidade

- Só fotografar para o dataset com consentimento explícito de quem
  desenhou e de quem aparece na cena.
- Remover EXIF antes de arquivar (GPS, serial do dispositivo, data e
  hora do dispositivo). A foto arquivada em `raw/` já deve estar
  limpa.
- Licença por foto, padrão `internal-only`, até decisão do dono do
  repositório sobre a licença do projeto (backlog item 10, pendente).
  **Nenhuma foto deste dataset é publicada externamente antes dessa
  decisão.**
- Fotos de terceiros só com licença escrita registrada no manifesto;
  sem isso, fora do dataset.

## Fotos fora do git, por padrão

As fotos ficam fora do versionamento (ver `.gitignore`): `raw/` e as
fotos dentro de `cases/` não são commitadas. O que vai para o git é
o manifesto, os `truth.json` (metadados, sem dado pessoal) e este
README. O avaliador roda a partir de qualquer pasta de dataset
local: `python tools/evaluate_dataset.py <manifesto>`.

## Como rodar

Sem webcam, sem janela, com seed fixa:

```
python tools/evaluate_dataset.py dataset/manifest.json --split all
python tools/evaluate_dataset.py dataset/manifest.json --split test --report out/relatorio.md
python tools/evaluate_dataset.py dataset/manifest.json --synthetic
```

`--synthetic` prova o avaliador antes da primeira foto real: ele
avalia cenas sintéticas geradas por código (as mesmas do selftest) e
mostra que o relatório, as métricas e o código de saída funcionam.
Um manifesto sem casos reais é válido: o relatório diz "0 casos" e
o avaliador sai com código 0, sem fingir medição.

## Níveis de verdade (truth.json)

- **A — obrigatório:** cantos da folha revisados por humano (ou
  `expects_sheet: false` para negativos), tipo de papel, iluminação,
  inclinação aproximada, ferramenta, classes presentes.
- **B — quando a forma é conhecida por construção:** `expected_shapes`
  com `{kind, params aproximados, tolerance_px}`. Quem desenhou
  declara; a tolerância é explícita, não um ponto exato falso.
- **C — só quando traçado:** `ink_mask` traçada, em coordenadas da
  folha retificada (850×1200). Nunca usar a saída do próprio
  `extract_ink` como verdade: isso é vazamento de rótulo, o avaliador
  sempre passaria.

## Baseline

Os limiares em `manifest.json` (`baseline`) são os portões atuais,
copiados do selftest: taxa de detecção, erro máximo de canto e
recall mínimo de tinta. Eles só mudam com valores medidos e
justificativa registrada — nunca em silêncio para fazer um caso que
falha passar.
