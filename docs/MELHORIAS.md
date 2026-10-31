# Melhorias possíveis para o Draw — pesquisa e análise

Data: 2026-10-29. Base: `origin/main` em `3d41174` (upstream `9c47e19`,
totalmente sincronizado: 26 commits à frente, 0 atrás).
Trabalho feito nesta branch: só este documento. Nenhum código alterado,
nenhum push, nenhum merge. A branch do Senado (`feature/senado-roadmap`,
7 commits, 101 testes) continua separada e divergente da main em 5 commits;
antes de qualquer merge dela, trazer a main nova para dentro dela.

## Estado atual (o que já existe, para não refazer)

- Pipeline: detecção da folha (ArUco ou sem marcadores, RETR_TREE),
  warp para 850×1200, tinta por limiar adaptativo, canvas ao vivo.
- Fork: traços vetoriais (tecla `v`, SVG+JSON), captura (`draw/capture.py`,
  listagem, backends, erros, reconexão), CI, backlog, verificação
  geométrica (108 sondas, pior erro 3,05 px).
- Senado (fora da main): diagnóstico por fonte, manifesto de dataset,
  régua de medição do papel, SVG em mm, ordem temporal em modo sessão,
  DXF R2000, reconhecimento linha/círculo/retângulo/`unknown`.
- Upstream novo (2026-10-28): `tools/eval_real_frames.py`, harness de
  avaliação em frames reais do laboratório (43 positivos, 6 negativos,
  11 máscaras anotadas, em `~/workspace/draw-lab`, fora do repo), com
  interface para um modelo ONNX futuro. Não duplicar esse harness: o
  manifesto de dataset do Senado deve ser reconciliado com ele.

## Gaps medidos, não achados (ordem de impacto)

1. **Sombra e luz baixa falham no portão de textura, não na geometria.**
   Régua do Senado: detecção 7/10 variantes, 0 falsos positivos; sombra
   derruba a fração de branco de 0,98 para 0,25 e luz baixa para 0,00
   (limiar absoluto V > 180); brilho passa com recall de só 51,56%.
   Próximo passo: normalização de luz atrás de flag (fundo por blur
   largo + divisão, CLAHE no canal L, ou Retinex multi-escala
   log(I) − log(G∗I)), medida relativa de branco, nunca afrouxar o
   portão às cegas. Custo: baixo, sem dependência nova.
2. **Canto oculto é falha insegura.** O detector devolve um quadrilátero
   errado (erro 131 px, recall 3,24%) em vez de recusar. Próximo passo:
   validação de cantos + confirmação temporal antes de aceitar um
   quadrilátero de fallback.
3. **Máscara de marcadores aplicada mesmo sem marcadores.** `extract_ink`
   sempre apaga tinta perto dos 4 centros do template, fora da região
   central do selftest. Próximo passo: máscara condicional ao modo.
4. **Exportações prontas, mas sem fiação.** SVG em mm, DXF e sessão
   existem no Senado, mas a tecla `v` continua exportando só SVG em px
   e não há flag/tecla para DXF nem para sessão na main. Próximo passo:
   fiação na CLI, sem mudar o padrão.
5. **v0.3 Paint ainda não existe** (backlog item 2): reproduzir traços
   movendo o mouse, com modo seco que só lista os movimentos. A ordem
   temporal do Senado é o pré-requisito que faltava.
6. **Reconhecimento parado em 3 classes.** Seta, elipse, `text_region`,
   letras/OCR e diagramas (conector + origem + destino) ficam para
   depois, por classe, com matriz de confusão — nunca um classificador
   único "reconhece tudo".
7. **Câmera: diagnóstico WSL ainda é texto de script.** O Senado propôs
   `diagnose_wsl` só-leitura (estágios PASS/FAIL/SKIP) e `--diagnose-only`
   no script; falta implementar. Nada de attach automático em comando
   de diagnóstico.

## Datasets pesquisados — o que serve, o que não serve

Regra geral: dataset externo bom para calibrar limiar e para treinar
classificador de forma, ruim para validar o pipeline completo do Draw,
porque nenhum deles é foto de folha A4 na mesa com a câmera do usuário.
O conjunto de frames reais do laboratório continua sendo a régua final.

| Dataset | Tamanho / conteúdo | Serve para | Cuidado |
|---|---|---|---|
| **QuickDraw (Google)** | 50M+ rabiscos, 345 categorias, vetorial (sequência de traços) | Reconhecimento de forma, ordem de traço, treino/validação de classificador; formato vetorial casa com `strokes.py` | Rabisco digital, não foto de papel; não valida detecção de folha nem tinta |
| **TU-Berlin** | ~20 mil desenhos à mão, 250 categorias, vetorial | Classificação de sketch, validação de formas, matriz de confusão por classe | Desenho isolado, sem folha, sem luz, sem câmera |
| **Sketchy** | 75 mil sketches, 125 categorias, pareado com foto | Forma + foto correspondente, teste de generalização | Pesado; foto não é folha na mesa |
| **SketchyScene / FS-COCO / SketchyCOCO** | 7–14 mil cenas com vários objetos | Diagramas e cenas com vários objetos (futuro, não agora) | Complexidade de cena foge do escopo atual |
| **IAM Handwriting** | 657 escritores, 13.353 linhas, 115.320 palavras, 300 dpi, registro gratuito obrigatório | Letras e texto manuscrito, só na etapa de OCR, depois de `text_region` | Licença restrita, não redistribuir; inglês, não português |
| **EMNIST / NIST SD19** | 814 mil caracteres, 47–62 classes, 28×28 | Classificador de caractere isolado, prototipagem rápida | Caractere isolado, não palavra nem diagrama |
| **H-DIBCO / DIBCO (2009–2019)** | Dezenas de pares imagem + ground truth binário, manuscritos degradados | Binarização de tinta: comparar limiar adaptativo atual com Sauvola/Otsu por F-measure, PSNR, DRD | Documento histórico, não desenho; útil só para a etapa de tinta |
| **RDSRD / SDSRD (sombra em documento)** | RDSRD real: 540 fotos, 25 documentos, com/sem sombra + máscara; SDSRD sintético: 970 documentos, 8.309 sombras | Validar normalização de luz e remoção de sombra (gap 1) | Documento impresso, não desenho; usar só como benchmark de luz, não de forma |
| **Frames reais do laboratório (já existe)** | 43 positivos, 6 negativos, 11 máscaras, fora do repo | Régua final de tudo: detecção, canto, tinta, falso positivo | Não publicar sem decidir licença; é o único com a câmera e a mesa reais |

Não recomendado agora: treinar rede neural do zero (sem dataset próprio
rotulado, sem GPU no loop, e o upstream já deixou o portão ONNX para
depois do harness clássico); baixar dataset inteiro para dentro do repo
(peso, licença, CI); usar saída do próprio detector como rótulo
(validação circular, já vetada pelo Senado).

## Melhorias priorizadas (o que eu faria, nesta ordem)

1. **Luz (gap 1) + canto oculto (gap 2)**, atrás de flag, com a régua
   existente travando regressão. É onde o sistema falha hoje na mesa.
2. **Fiação das exportações + v0.3 Paint em modo seco.** Fecha o loop
   papel → traço → reprodução, que é a promessa do roadmap.
3. **Coleta real mínima (20–30 casos) no manifesto do Senado**,
   reconciliada com `eval_real_frames.py`: mesma foto, cantos revisados,
   papel/luz/inclinação registrados, licença `internal-only`. Sem isso,
   todo limiar novo continua calibrado no sintético.
4. **Seta e elipse no reconhecedor**, por classe, com QuickDraw/TU-Berlin
   como validação externa e o dataset real como régua final.
5. **Diagnóstico WSL estruturado** (`--diagnose-only`), porque é o
   atrito que o usuário nomeou como principal.
6. **Depois, e só depois:** letras/OCR (IAM/EMNIST, com `text_region`
   antes), diagramas, modelo ONNX no harness do upstream, e integração
   com CAD/simulação (v2/v3 do roadmap).

## O que NÃO fazer

- Não prometer "funciona em todas as câmeras / papéis / formas".
- Não calibrar limiar olhando o split de teste (held-out só mede).
- Não afrouxar portão de textura, limiar de verificação (5 px / 40%)
  ou recall para teste passar.
- Não colocar foto real no repo antes da decisão de licença (item 10
  do backlog, decisão do dono).
