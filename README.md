# Draw

O Draw é uma mesa digitalizadora sem mesa digitalizadora.
Uma webcam aponta para a mesa, a pessoa desenha numa folha de papel e o sistema reproduz o desenho no digital.
A visão completa fica em [VISION.md](VISION.md).

Estado atual: a captura funciona e roda em tempo real na mesa do autor. A detecção de folha ainda é frágil em uso contínuo. A trilha neural está em construção e ainda não substitui o detector clássico.

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

Pré-requisitos: Python 3.10+ e uma webcam.

```bash
pip install -r requirements.txt

# Rodar com qualquer folha sobre a mesa
python main.py

# Outras fontes de vídeo: outro índice de webcam ou um stream de rede
python main.py --source 1
python main.py --source /dev/video2
python main.py --source http://192.168.0.10:8080/video
```

## Câmeras: listar, selecionar e diagnosticar

A captura vive em `draw/capture.py`, separada do pipeline de visão.
O pipeline continua recebendo só frames, seja a fonte uma webcam USB,
uma câmera IP/MJPEG, um arquivo ou um índice alternativo.

```bash
# Lista as webcams locais que abrem, sem abrir janela (sai 1 se nenhuma abrir)
python main.py --list-cameras

# Escolhe a fonte sem editar código: índice, porta V4L2 ou stream
python main.py --source 1
python main.py --source /dev/video2
python main.py --source rtsp://camera:554/stream

# Tamanho e FPS só valem para webcam local; o padrão é MJPG 1280x720
python main.py --source 1 --width 640 --height 480 --fps 30

# Força um backend (Linux: v4l2; Windows: msmf, dshow; qualquer SO: any, ffmpeg)
python main.py --source 1 --backend v4l2
```

Comportamento quando algo dá errado, sem travar indefinidamente:

- Abertura com timeout e tentativas por backend, na ordem do SO.
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
seção abaixo — o `--check` e o script de setup já cuidam de
`uvcvideo`, `usbipd`, grupo `video` e interop com PowerShell, e a
câmera via rede continua sendo a rota que não depende de passthrough
USB. Nem toda câmera USB é compatível com passthrough; se o attach
falhar, use a rota via rede ou a ponte de captura do Windows.

Os testes dessa camada, em `tests/test_capture.py`, usam só mocks:
parsing de fontes, listagem, ordem de backends, cada tipo de erro e
reconexão que volta ou que desiste. Nenhum deles exige câmera física,
então o CI roda todos sem hardware.

O fluxo padrão usa uma folha solta ou uma página de caderno sob a webcam.
Ele dispensa impressão. O Draw encontra os cantos da folha, retifica a
imagem e mostra o canvas digital.
### Windows com WSL
O WSL2 não enxerga a webcam do Windows por padrão. Nenhum `/dev/video*` existe dentro dele até a câmera ser anexada.

O guia completo de instalação fica em [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md). O caminho curto é um comando só, rodado de dentro do WSL:

```bash
bash scripts/setup_camera_wsl.sh
python main.py
```

O script prepara o usbipd, anexa a câmera ao WSL e confere o resultado. Ele também corrige o grupo `video` do usuário e religa os serviços de áudio que ele mesmo parou.

Se algo falhar, rode o diagnóstico antes de mexer em outra coisa:

```bash
python main.py --check
```

### Linux puro com câmera USB

Conecte a webcam e rode:

Teclas dentro do app: `s` salva um PNG em `captures/` · `v` salva
traços vetoriais em `captures/` (SVG + JSON) · `q` ou `Esc` sai.
```bash
python main.py
python main.py --check
```

O comando com `--check` valida o OpenCV, a existência de `/dev/video*`, a permissão do processo sobre o dispositivo e a leitura de um quadro real. Cada falha sai nomeada, com a correção na linha seguinte.

## Uso

```bash
# Modo padrão, sem marcadores, em qualquer folha ou página de caderno
python main.py

# Outra webcam local
python main.py --source 1

# Câmera via rede, por um stream MJPEG
python main.py --source http://192.168.0.10:8080/video

# Valida a câmera e sai, sem abrir janela
python main.py --check

# Modo opcional com a folha-modelo impressa
python template/generate_template.py
python main.py --markers
```

Teclas dentro do app:

- `s` salva o canvas digital como PNG em `captures/`
- `f` salva o frame cru atual em `captures/frames/`
- `g` liga ou desliga a captura automática, um frame cru a cada 10 quadros
- `q` ou `Esc` sai

O modo padrão é o sem marcadores. Ele é o padrão desde a v0.2. O modo com marcadores continua disponível atrás da flag `--markers` e pede a folha-modelo impressa com quatro marcadores ArUco.

## Como a detecção funciona hoje

A detecção atual é visão computacional clássica, em OpenCV. Ela tem duas etapas.

A primeira etapa propõe candidatos por geometria. O frame passa por Canny e pela árvore completa de contornos. Os maiores contornos viram candidatos quando fecham um quadrilátero convexo, com área de pelo menos 12% do frame e razão de aspecto entre 0,4 e 2,5. A aproximação do polígono usa uma escada de epsilons, porque cantos arredondados e a lombada de um caderno não fecham num epsilon só. Os cantos aceitos ganham refino subpixel. No loop do app, a detecção roda de forma espaçada e os cantos passam por estabilização temporal.

A segunda etapa é um portão de textura. Ele mede o interior de cada candidato e devolve um de três vereditos:

- `paper`: o interior parece papel
- `blank_maybe`: o candidato é branco o bastante, mas liso demais para afirmar papel
- rejeitado: o candidato nem branco o bastante é

O portão mede três coisas no interior encolhido do candidato. Ele mede a fração de branco, a densidade de bordas internas e a variância do Laplaciano. Um candidato vira `paper` quando a fração de branco passa de 0,5, a variância do Laplaciano passa de 30 e uma das duas condições de textura vale: densidade de bordas de pelo menos 0,03 ou variância do Laplaciano de pelo menos 200. Um candidato reprovado não encerra a busca. O contorno seguinte ainda pode ser a folha.

Um candidato `blank_maybe` recebe tratamento mais duro no app. Ele precisa do dobro de detecções consistentes para travar, 6 em vez de 3. Enquanto a trava vem desse veredito, a tela mostra o aviso de baixa confiança.

A limitação conhecida é a folha totalmente em branco e lisa. Por dentro, ela mede igual a qualquer outro retângulo branco e liso, como uma bandeja branca. Nenhuma estatística de interior separa os dois casos. Quem separa é identidade de objeto, e identidade de objeto se aprende. Essa é uma das razões da trilha neural.

O histórico explica o portão. A versão só geométrica aceitava qualquer quadrilátero convexo grande. Numa auditoria com 10 cenas sintéticas sem papel nenhum, ela disparou em 6. Tapete de mesa, caixa de papelão, monitor e bandeja branca viraram folha. O portão de textura cortou esses casos nas cenas medidas, sem modelo nenhum. Os limiares atuais são calibração de laboratório, feita sobre cenas sintéticas e um frame real. Eles precisam ser refeitos com capturas reais.

Depois da detecção, o pipeline retifica a folha por homografia para a vista de cima. Um limiar adaptativo separa a tinta do papel nessa vista. O resultado aparece como canvas digital ao lado da câmera. Apagar com a borracha funciona de forma natural. A tinta some do papel e some do canvas no frame seguinte.

## Trilha neural

A direção do projeto é trocar a detecção artesanal por um modelo que reconheça papel como objeto. O plano usa o detector clássico como professor. O app captura frames crus da mesa com as teclas `f` e `g`. O rotulador roda o detector clássico nesses frames e grava os cantos, com imagens de revisão para conferência humana. Esses frames rotulados viram dataset de treino, junto de datasets reais de documentos. Todo o treinamento roda no Kaggle, em GPU ou TPU, e os frames reais do usuário ficam em datasets privados. A arquitetura alvo é segmentação de papel com um sinal de presença, para o sistema aprender também a dizer que não há papel na cena. Os modelos treinados só em dados sintéticos já falharam no frame real, com erro médio de centenas de pixels. O gap entre o sintético e a mesa real é o problema que essa trilha ataca agora.

## Arquitetura

A árvore do repositório, pasta por pasta:

- `draw/`: o pacote principal. `app.py` tem o loop da webcam, as janelas e as validações de câmera. `paper.py` tem a detecção de folha, o portão de textura e a homografia. `ink.py` extrai a tinta e renderiza o canvas digital. `config.py` guarda o tamanho da folha retificada e os dados dos marcadores.
- `template/`: o gerador da folha-modelo imprimível, usado só pelo modo `--markers`. O PNG gerado não entra no versionamento.
- `bridge/`: a ponte de captura no Windows. Um `ffmpeg.exe` no Windows serve a webcam como stream MJPEG e o Draw consome pelo `--source`. É a rota reserva e nunca foi testada numa máquina real. O passo a passo fica em [bridge/README.md](bridge/README.md).
- `scripts/`: os scripts de instalação. Hoje guarda `setup_camera_wsl.sh`, o instalador da câmera do WSL.
- `tests/`: os testes sem webcam. Hoje guarda `selftest.py`, que valida detecção, homografia, extração de tinta e o portão de textura em cenas sintéticas.
- `tools/`: as ferramentas de apoio. Hoje guarda `label_frames.py`, o rotulador da trilha neural.
- `verification/`: o registro da verificação formal limitada da v0. Guarda a skill do projeto e o script que prova propriedades da homografia com sondas numéricas. É um registro histórico de uma etapa concluída, não uma etapa do pipeline.
- `docs/`: a documentação de instalação. Guarda o guia do Windows com WSL em [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md).
- `main.py`: o ponto de entrada. Ele só chama o app.
- `VISION.md`: o documento de visão do projeto.

## Desenvolvimento

Sem webcam por perto, o pipeline inteiro se valida com cenas sintéticas:

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

Para rotular frames capturados:

```bash
python tools/label_frames.py captures/frames dataset/rotulos```

A saída é um `labels.json` com os frames rotulados e os rejeitados. Cada frame rotulado ganha uma imagem de revisão na subpasta `overlays`. Os frames de origem nunca são modificados.

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

- Captura em tempo real: folha retificada, tinta extraída e canvas digital ao vivo
- Fonte de vídeo flexível, por índice local ou por URL de stream
- Câmera do Windows anexada ao WSL pelo script de instalação
- Diagnóstico de câmera porta por porta com `--check`
- Detecção sem marcadores como modo padrão
- Portão de textura com veredito de baixa confiança para folha lisa
- Captura de frames crus e rotulador do professor clássico

Em andamento:

- Teste da detecção em uso contínuo na mesa real
- Reconhecimento de datasets reais de documentos para o treino de segmentação
- Coleta de frames reais da mesa, com papel e sem papel

Planejado:

- Segmentação de papel treinada no Kaggle, com sinal de presença
- Reprodução dos traços em aplicativos de desenho
- Interpretação de formas, letras e diagramas
- Integração com um primeiro sistema alvo, CAD ou simulador
- Feedback de simulação em tempo real