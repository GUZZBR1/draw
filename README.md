<p align="center">
  <img src="docs/icon.svg" width="160" alt="Draw">
</p>

<h1 align="center">Draw</h1>

<p align="center">
  Uma mesa digitalizadora sem mesa digitalizadora.
</p>

Uma webcam aponta para a mesa. A pessoa escreve e desenha numa folha de papel comum, e o Draw reproduz o traço no digital, ao vivo. Sem caneta especial, sem superfície especial, sem folha-modelo impressa.

A visão completa do projeto fica em [VISION.md](VISION.md). Este documento cobre o estado atual, como rodar e como o sistema funciona por dentro.

## Status

O Draw roda em tempo real na mesa do autor. O detector do dia a dia é o modelo neural v3, com o detector clássico de reserva dentro do mesmo processo. O canvas digital já acumula tinta entre frames e trata a mão como região desconhecida.

O limite honesto do momento é a generalização. A v3 foi julgada nos frames da mesa do autor e em datasets públicos de documentos. Ela nunca passou por um conjunto de teste de outra mesa, outra câmera e outra luz. Esse conjunto ainda não existe, e nenhum número deste README mede outro setup.

## Início rápido

## v0 — o que já faz

- Detecta uma folha-modelo (A4 com marcadores ArUco nos cantos) pela webcam
- Retifica a perspectiva em tempo real: gera uma "vista de cima" perfeita
  da folha, como um scanner
- Extrai a tinta do papel e mostra um **canvas digital** ao lado, ao vivo
- Apagar com a borracha funciona naturalmente: a tinta sumiu do papel, ela
  some do canvas
- Salva o canvas como PNG com a tecla `s`
- Salva os traços vetoriais como SVG e JSON com a tecla `v`

## Como rodar

Pré-requisitos: Python 3.10+ e uma webcam (o autor roda 3.12), além das dependências do projeto.

```bash
pip install -r requirements.txt

# Rodar com qualquer folha sobre a mesa
python3 main.py

# Outras fontes de vídeo: outro índice de webcam ou um stream de rede
python3 main.py --source 1
python3 main.py --source /dev/video2
python3 main.py --source http://192.168.0.10:8080/video
```

## Câmeras: listar, selecionar e diagnosticar

A captura vive em `draw/capture.py`, separada do pipeline de visão.
O pipeline continua recebendo só frames, seja a fonte uma webcam USB,
uma câmera IP/MJPEG, um arquivo ou um índice alternativo.

```bash
# Lista as webcams locais que abrem, sem abrir janela (sai 1 se nenhuma abrir)
python3 main.py --list-cameras

# Escolhe a fonte sem editar código: índice, porta V4L2 ou stream
python3 main.py --source 1
python3 main.py --source /dev/video2
python3 main.py --source rtsp://camera:554/stream

# Tamanho e FPS só valem para webcam local; o padrão é MJPG 1280x720
python3 main.py --source 1 --width 640 --height 480 --fps 30

# Força um backend (Linux: v4l2; Windows: msmf, dshow; qualquer SO: any, ffmpeg)
python3 main.py --source 1 --backend v4l2
```

Comportamento quando algo dá errado:

- A abertura passa os timeouts do OpenCV no construtor da captura e tenta
  backends na ordem do SO. O suporte aos timeouts depende da versão do
  OpenCV e do backend: FFmpeg e GStreamer são os backends documentados.
  Para URL/arquivo, se o backend rejeitar esses parâmetros, o Draw falha
  com diagnóstico e não tenta uma abertura de rede sem limite. Para
  webcam local, ele preserva a tentativa antiga sem parâmetros se o
  backend não os aceitar.
  No Linux é V4L2 e depois o genérico; no Windows é MSMF, DShow e
  depois o genérico. Um backend pedido com `--backend` é o único tentado.
- Erro com a causa e a ação: porta inexistente, falta de permissão no
  grupo `video`, dispositivo ocupado por outro programa, ou stream de
  rede fora do ar. O erro de abertura reutiliza o diagnóstico por
  porta do `--check`, não o reinventa.
- Se os quadros pararem no meio do uso, o app tenta reabrir a mesma
  fonte (padrão: 3 tentativas; `--reconnect-attempts 0` desliga a
  reconexão). Se voltar, continua; se não voltar, encerra limpo
  avisando que foi desconexão.

Plataformas, resumido: Linux usa V4L2 e `/dev/video*`; Windows usa
MSMF/DShow direto; WSL2 continua sendo um caso separado, descrito na
seção abaixo — para uma fonte local, `--check` diagnostica a porta
selecionada e o script de setup cuida de `uvcvideo`, `usbipd`, grupo
`video` e interop com PowerShell. Uma URL de rede também pode ser
validada sem dispositivo `/dev/video*`. Nem toda câmera USB é compatível
com passthrough; se o attach falhar, use a rota via rede ou a ponte de
captura do Windows.

Os testes dessa camada, em `tests/test_capture.py`, usam só mocks:
parsing de fontes, listagem, ordem de backends, cada tipo de erro e
reconexão que volta ou que desiste. Nenhum deles exige câmera física,
então o CI roda todos sem hardware.

O fluxo padrão usa uma folha solta ou uma página de caderno sob a webcam.
Ele dispensa impressão. O Draw encontra os cantos da folha, retifica a
imagem e mostra o canvas digital.
### Windows com WSL
O WSL2 não enxerga a webcam do Windows por padrão. Nenhum `/dev/video*` existe dentro dele até a câmera ser anexada.

O guia completo fica em [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md). O caminho curto roda de dentro do WSL.

```bash
bash scripts/setup_camera_wsl.sh
python3 main.py --check
python3 main.py
```

O script prepara o usbipd, anexa a câmera ao WSL e confere o resultado. Depois de reiniciar o WSL, rode o script de novo para reanexar a câmera. Ele encontra o PowerShell mesmo se o PATH do shell não incluir executáveis do Windows e preserva outros dispositivos USB/IP.

Se algo falhar, rode o diagnóstico antes de mexer em outra coisa.

```bash
python3 main.py --check
python3 main.py --check --source 1
python3 main.py --check --source http://192.168.0.10:8080/video
```

O `--check` valida o OpenCV, a existência de `/dev/video*`, a permissão do processo sobre o dispositivo e a leitura de um quadro real. Cada falha sai nomeada, com a correção na linha seguinte.

### Linux com câmera USB

Teclas dentro do app: `s` salva um PNG em `captures/` · `v` salva
traços vetoriais em `captures/` (SVG + JSON) · `q` ou `Esc` sai.
```bash
python3 main.py
```

`--check` valida exatamente a fonte selecionada por `--source`, passa
timeouts de abertura e leitura no construtor, tenta receber um frame por
até 5 segundos e informa dimensões e FPS quando disponíveis. URLs e
arquivos não dependem de `/dev/video*`. Em fontes locais, a falha
identifica a porta e a permissão aplicável. O limite de leitura depende
do backend: se `capture.read()` bloquear e o backend não aplicar o
timeout, a chamada nativa pode ultrapassar os 5 segundos. Para URL ou
arquivo sem suporte aos parâmetros, o diagnóstico falha sem tentar uma
abertura sem timeout.

### Com o detector neural

O repositório não publica arquivo de modelo. Um clone novo roda só com o detector clássico até alguém colocar um ONNX na pasta `models/`. Na máquina do autor, o comando do dia a dia é este.

```bash
python3 main.py --neural models/draw_seg_v3_dyn.onnx
```

## Uso

| Flag | O que faz |
|---|---|
| `--source N` ou URL | Webcam local por índice (padrão 0) ou stream MJPEG por URL |
| `--neural [modelo.onnx]` | Detector neural como primário, com o clássico de reserva. Sem caminho, lê `DRAW_NEURAL_MODEL` |
| `--ink-model [modelo.onnx]` | Modelo aprendido de tinta como fonte da máscara. Sem caminho, lê `DRAW_INK_MODEL`. Sem arquivo ou com erro de carga, o app avisa uma vez e segue com a heurística |
| `--pair-ink-model [modelo.onnx]` | Modelo experimental que compara dois frames para detectar tinta nova. Sem caminho, lê `DRAW_PAIR_INK_MODEL` |
| `--hand-model [modelo.task]` | Máscara de mão do MediaPipe para o canvas acumulativo. Sem caminho, lê `DRAW_HAND_MODEL` ou `models/hand_landmarker.task` |
| `--markers` | Modo opcional com a folha-modelo impressa e marcadores ArUco |
| `--check` | Valida a câmera porta por porta e sai, sem abrir janela |
| `--save-dir pasta` | Pasta das capturas. O padrão é `captures` |

A variável `DRAW_ORT_THREADS` troca o número de threads da sessão ONNX sem tocar no código. O padrão é 4, medido na máquina do autor.

A máscara de mão é opcional e pede um pacote fora da instalação padrão.

```bash
# Modo padrão, sem marcadores, em qualquer folha ou página de caderno
python3 main.py

# Outra webcam local
python3 main.py --source 1

# Câmera via rede, por um stream MJPEG
python3 main.py --source http://192.168.0.10:8080/video

# Valida a fonte selecionada e sai, sem abrir janela
python3 main.py --check
python3 main.py --check --source http://192.168.0.10:8080/video

# Modo opcional com a folha-modelo impressa
python template/generate_template.py
python3 main.py --markers

# Pacote opcional da máscara de mão
pip install mediapipe
```

Sem o pacote ou sem o modelo, o app roda sem a máscara e avisa uma vez.

Teclas dentro do app:

- `s` salva o canvas digital como PNG em `captures/`
- `f` salva o frame cru atual em `captures/frames/`
- `g` liga ou desliga a captura automática, um frame cru a cada 10 quadros
- `q` ou `Esc` sai

O modo padrão é o sem marcadores, em qualquer folha ou página de caderno. Ele é o padrão desde a v0.2. O modo com marcadores pede a folha-modelo gerada por `python template/generate_template.py`.

## Como funciona

O pipeline tem quatro etapas por frame. A câmera entrega o frame. Um detector encontra a folha e devolve os quatro cantos. A homografia retifica a folha para a vista de cima, em 850 por 1200 px. A etapa de tinta separa o traço do papel nessa vista e atualiza o canvas digital.

### Detecção neural

O detector primário é um SegFormer-B0 exportado para ONNX. Ele devolve duas saídas. Um logit de presença diz se há papel na cena. Uma máscara diz onde o papel está. O quadrilátero da folha sai do maior contorno da máscara, com escada de epsilon, reajuste de cada aresta como reta e refino subpixel nos cantos. Uma memória de arestas guarda a última reta confiável de cada lado. Quando a mão cobre uma aresta, a reta herdada segura o canto no lugar em vez de projetar a borda do braço.

Quando o modelo não vê folha no frame, o detector clássico assume aquele frame. Quando o modelo vê folha e o ajuste do quadrilátero falha, o retângulo de área mínima da máscara entra como reserva.

O ONNX dinâmico aceita qualquer resolução de entrada. O app roda esse arquivo a 384 por 384. O ONNX original, de formato fixo, impõe os 512 por 512 dele. A seção de desempenho traz os números dessa troca.

### Detecção clássica

O detector clássico é visão computacional em OpenCV, em duas etapas.

A primeira etapa propõe candidatos por geometria. O frame passa por Canny e pela árvore completa de contornos. Um contorno vira candidato quando fecha um quadrilátero convexo, com área de pelo menos 12% do frame e razão de aspecto entre 0,4 e 2,5. A aproximação do polígono usa uma escada de epsilons de 0,01 a 0,05, porque cantos arredondados e a lombada de um caderno não fecham num epsilon só. Os cantos aceitos ganham refino subpixel. No loop do app, a detecção roda de forma espaçada e os cantos passam por estabilização temporal.

A segunda etapa é um portão de textura. Ele mede o interior encolhido de cada candidato e devolve um de três vereditos.

- `paper` quando o interior parece papel
- `blank_maybe` quando o candidato é branco o bastante, mas liso demais para afirmar papel
- rejeitado quando o candidato nem branco o bastante é

O portão mede a fração de branco, a densidade de bordas internas e a variância do Laplaciano. Um candidato vira `paper` quando a fração de branco passa de 0,5, a variância do Laplaciano passa de 30 e uma de duas condições de textura vale. A densidade de bordas precisa de pelo menos 0,03, ou a variância do Laplaciano precisa de pelo menos 200. Um candidato reprovado não encerra a busca. O contorno seguinte ainda pode ser a folha.

Um candidato `blank_maybe` recebe tratamento mais duro no app. Ele precisa do dobro de detecções consistentes para travar, 6 em vez de 3. Enquanto a trava vem desse veredito, a tela mostra o aviso de baixa confiança.

O portão existe por um histórico medido. A versão só geométrica aceitava qualquer quadrilátero convexo grande. Numa auditoria com 10 cenas sem papel nenhum, ela disparou em 6. Tapete de mesa, caixa de papelão, monitor e bandeja branca viraram folha. O portão de textura cortou esses casos nas cenas medidas, sem modelo nenhum. Os limiares atuais são calibração de laboratório e precisam ser refeitos com capturas reais.

### Tinta e canvas acumulativo

Um limiar adaptativo separa a tinta do papel na folha retificada. Um veto separável roda junto. Um pixel só vira tinta quando está pelo menos 6 níveis abaixo das medianas da linha e da coluna inteiras dele. O veto mata a pauta do caderno e a textura do papel, que o limiar sozinho marcava como tinta.

O canvas não é uma foto do frame atual. Ele acumula evidência com histerese sobre uma janela de 15 frames. Um pixel entra na tinta com fração de amostras visíveis de 0,45 e sai com 0,20, com um mínimo de 4 amostras visíveis. O traço sobrevive à mão passando por cima. A borracha também funciona por esse caminho. A tinta apagada some do canvas depois de amostras limpas visíveis o bastante, não no frame seguinte.

Com a máscara de mão ligada, a região sob a mão vira desconhecida. Ela não ganha nem perde tinta enquanto está coberta.

O app também aceita o modelo de um frame por `--ink-model` e um modelo experimental que compara frames consecutivos por `--pair-ink-model`. O segundo passou por uma verificação local de CPU, mas ainda não foi validado com webcam e tem latência alta; sem esses modelos, vale o caminho heurístico descrito acima.

## Desempenho medido

Números medidos em 2026-10-29 no Ryzen 7 7700X do autor, com o pipeline completo do app sobre frames reais. Medianas por frame.

| Configuração | Tempo por frame | Taxa |
|---|---|---|
| Sessão padrão do ONNX Runtime, modelo fixo a 512 | 212,8 ms | 4,7 fps |
| 4 threads intra-op, modelo fixo a 512 | 183,5 ms | 5,4 fps |
| 4 threads intra-op, modelo dinâmico a 384 | 127,2 ms | 7,9 fps |

A sessão sozinha explica quase tudo. No padrão do ONNX Runtime, `session.run` marcava 248 ms no mesmo tensor. Com 4 threads, marca 122 ms a 512 e 58,3 ms a 384. Com 16 threads, piora para 210 ms. INT8 dinâmico foi medido e rejeitado. A qualidade dele se manteve, mas o tempo subiu para 366 ms, três vezes o FP32 dessa máquina.

A troca de resolução custou pouco em qualidade. Nos 49 frames do harness, a máscara a 384 tem IoU médio de 0,984 contra a máscara a 512. Nenhum veredito de presença trocou nos positivos. As três frames de uma caixa branca sobre a mesa, que a 512 eram falso positivo fraco, a 384 viram negativo.

## Treino do detector

Todo o treino roda no Kaggle, em GPU. Os frames reais do autor ficam em datasets privados e nunca entram neste repositório.

A mistura da v3 combina quatro fontes. MIDV-2020 photos traz 1.000 fotos reais de documentos com cantos anotados. SmartDoc 2015 traz 24.889 frames com gabarito. O dataset school-notebooks-RU traz páginas de caderno. Os 49 frames da mesa do autor entram com rótulo de presença para todos e polígonos desenhados à mão para 11 deles.

O detector clássico chegou a ser o professor planejado para rotular os frames. Nos 49 frames reais ele rotulou zero, porque o caderno sai cortado pela borda superior do quadro e os quatro cantos nunca aparecem juntos. O gabarito humano substituiu o professor.

O harness `tools/eval_real_frames.py` é o portão de qualquer modelo novo. Nenhum modelo entra no app sem passar por ele. A v3 marca presença em 43 de 43 positivos e 0 falsos positivos nos 6 negativos. Nas 11 anotações, o IoU de máscara é 0,962 de média e 0,841 de mínimo a 512. Na configuração de produção, a 384, presença é 49 de 49 e o IoU é 0,959 de média e 0,851 de mínimo.

O kernel da v3 levou 44,7 minutos numa T4. O kernel da v2 levou 3 horas e 51 minutos. A diferença não veio de dados menores. Veio de pipeline assíncrono com workers, precisão mista, batch 16 e uma agenda de 20 épocas com parada por régua, no lugar de 40 épocas fixas.

## Limites conhecidos

- Uma folha totalmente em branco e lisa mede igual a qualquer retângulo branco e liso, como uma bandeja branca. Nenhuma estatística de interior separa os dois casos para o detector clássico. Quem separa é identidade de objeto aprendida, e é por isso que o neural é o primário.
- A v3 nunca foi julgada na mesa de outra pessoa. O conjunto de teste externo é a régua que falta para chamar o detector de geral.
- A ponte de captura do Windows em `bridge/` nunca foi testada numa máquina real.
- Traço fino de 1 a 3 px sofre com o jitter da retificação. A média medida é de 5,4 px por amostra no replay do autor. O canvas acumulativo reduz o dano, mas não elimina a causa.

## Arquitetura

- `draw/` é o pacote principal. `app.py` tem o loop da webcam, as janelas e as validações de câmera. `paper.py` tem a detecção clássica, o portão de textura e a homografia. `neural.py` tem o detector neural por ONNX, com presença, máscara e quadrilátero. `ink.py` tem a heurística de tinta e o canvas acumulativo. `ink_model.py` tem o extrator do modelo aprendido de tinta. `hand.py` tem a máscara de mão do MediaPipe. `config.py` guarda o tamanho da folha retificada e os dados dos marcadores.
- `template/` tem o gerador da folha-modelo imprimível, usado só pelo modo `--markers`. O PNG gerado não entra no versionamento.
- `bridge/` tem a ponte de captura no Windows. Um `ffmpeg.exe` no Windows serve a webcam como stream MJPEG e o Draw consome pelo `--source`. O passo a passo fica em [bridge/README.md](bridge/README.md).
- `scripts/` tem os scripts de instalação. Hoje guarda `setup_camera_wsl.sh`, o instalador da câmera do WSL.
- `tests/` tem os testes sem webcam. Hoje guarda `selftest.py`, que valida detecção, homografia, extração de tinta e o portão de textura em cenas sintéticas.
- `tools/` tem as ferramentas de apoio do projeto. As ferramentas de treino (`label_frames.py`, `eval_real_frames.py` e outras) ficaram na branch `training-zone`.
- `verification/` é o registro da verificação formal limitada da v0. Ele guarda a skill do projeto e o script que prova propriedades da homografia com sondas numéricas. É um registro histórico de uma etapa concluída, não uma etapa do pipeline.
- `docs/` tem a documentação de instalação, com o guia do Windows com WSL.
- `main.py` é o ponto de entrada. Ele só chama o app.
- `VISION.md` é o documento de visão do projeto.

## Desenvolvimento

Sem webcam por perto, o pipeline inteiro se valida com cenas sintéticas.

```bash
python tests/selftest.py
python -m pytest tests/ -q
python verification/run_verification.py
```

O CI do GitHub Actions roda esses três passos em todo push para a main
e em todo pull request. A verificação geométrica cobre 108 sondas em
12 posições da folha, e o modo `--sabotage` dela deve falhar, para
provar que o verificador não ficou cego.

## Estrutura

```
main.py                     ponto de entrada
draw/
  config.py                 tamanho da folha, marcadores, posições
  paper.py                  detecção dos marcadores + homografia (perspectiva)
  ink.py                    extração da tinta e renderização do canvas digital
  strokes.py                traço vetorial: afinamento, rastreamento,
                            suavização, SVG e JSON
  app.py                    loop da webcam e janelas
  capture.py                fontes, backends, listagem, erros e reconexao
template/
  generate_template.py      gera a folha-modelo imprimível (PNG)
tests/
  selftest.py               teste do pipeline sem webcam
  test_capture.py           captura com mocks, sem camera fisica

O selftest roda os casos com marcadores, sem marcadores, de folha aninhada e do portão de textura. Ele imprime o erro máximo de canto e o recall de tinta de cada caso. Todos precisam passar antes de qualquer publicação.

A verificação formal limitada roda com o comando abaixo.

```bash
python verification/run_verification.py
```

Para rotular frames capturados, use a branch `training-zone`: ela guarda `tools/label_frames.py` e `captures/` fora da main.

## v0.2 — traço vetorial

A máscara de tinta vira traços com `draw/strokes.py`: afinamento da
tinta para um esqueleto de 1 pixel, rastreamento do esqueleto em
caminhos ordenados, suavização por média móvel e exportação em SVG
e JSON, com largura, altura, pontos, comprimento e marca de traço
fechado.

Dois limites são intencionais e documentados no módulo. Junções
dividem traços: um X vira 4 segmentos que se encontram no centro.
E a ordem dos traços é determinística, por centroide, de cima para
baixo e da esquerda para a direita, e não a ordem temporal em que
o traço foi desenhado. Rastrear a ordem temporal, frame a frame,
é trabalho futuro.

## v0.2 — traço vetorial

A máscara de tinta vira traços com `draw/strokes.py`: afinamento da
tinta para um esqueleto de 1 pixel, rastreamento do esqueleto em
caminhos ordenados, suavização por média móvel e exportação em SVG
e JSON, com largura, altura, pontos, comprimento e marca de traço
fechado.

Dois limites são intencionais e documentados no módulo. Junções
dividem traços: um X vira 4 segmentos que se encontram no centro.
E a ordem dos traços é determinística, por centroide, de cima para
baixo e da esquerda para a direita, e não a ordem temporal em que
o traço foi desenhado. Rastrear a ordem temporal, frame a frame,
é trabalho futuro.

## Roadmap

- [x] v0.1 — captura: folha retificada + canvas digital em tempo real
- [x] v0.1.1 — fonte de vídeo flexível (`--source` índice ou URL) para rodar no WSL com câmera via rede
- [x] v0.2 — traço vetorial: ordem, caminho e suavização dos traços
- [ ] v0.3 — reprodução dos traços no Paint via controle do mouse
- [ ] v1 — interpretação: formas, letras e diagramas a partir dos traços
- [ ] v2 — integração com um primeiro sistema alvo (CAD / simulador)
- [ ] v3 — feedback de simulação em tempo real

Feito:

- Captura em tempo real, com folha retificada, tinta extraída e canvas digital ao vivo
- Fonte de vídeo flexível, por índice local ou por URL de stream
- Câmera do Windows anexada ao WSL pelo script de instalação
- Diagnóstico da fonte selecionada com `--check --source`
- Detecção sem marcadores como modo padrão
- Portão de textura com veredito de baixa confiança para folha lisa
- Segmentação de papel treinada no Kaggle, com sinal de presença, nas versões v1, v2 e v3
- Datasets reais de documentos e frames da mesa do autor no treino
- Canvas acumulativo com histerese e máscara de mão opcional
- Pacote de desempenho com threads fixadas e modelo dinâmico a 384

Em andamento:

- Conjunto de teste externo, com frames de outros setups, para julgar a generalização da v3
- Fase do traço, com a fidelidade ao vivo já medida e a ponte de cursor para o Windows em construção
- Modelo aprendido de tinta, com a formulação fechada e o dataset sintético em montagem

Planejado:

- Segmentação de papel treinada no Kaggle, com sinal de presença
- Reprodução dos traços em aplicativos de desenho
- Traço vetorial, com ordem, caminho e suavização dos traços
- Interpretação de formas, letras e diagramas
- Integração com um primeiro sistema alvo, CAD ou simulador
- Feedback de simulação em tempo real
