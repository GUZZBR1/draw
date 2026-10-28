# Draw

**Uma mesa digitalizadora sem mesa digitalizadora — e, no futuro, uma ponte
entre o papel e os simuladores.**

O Draw usa uma webcam comum apontada para a sua mesa: você desenha numa
folha de papel e o sistema reproduz o desenho digitalmente, em tempo real.

A visão completa — desenhar no papel e receber feedback imediato de
sistemas como AutoCAD e NVIDIA Isaac Sim enquanto você desenha — está em
[VISION.md](VISION.md). Este repositório começa pela primeira barreira:
a captura confiável do papel para o digital.

## v0 — o que já faz

- Detecta uma folha-modelo (A4 com marcadores ArUco nos cantos) pela webcam
- Retifica a perspectiva em tempo real: gera uma "vista de cima" perfeita
  da folha, como um scanner
- Extrai a tinta do papel e mostra um **canvas digital** ao lado, ao vivo
- Apagar com a borracha funciona naturalmente: a tinta sumiu do papel, ela
  some do canvas
- Salva o canvas como PNG com a tecla `s`

## Como rodar

Pré-requisitos: Python 3.10+ e uma webcam.

```bash
pip install -r requirements.txt

# 1. Gerar e imprimir a folha-modelo (100% do tamanho, A4)
python template/generate_template.py

# 2. Rodar (aponte a webcam para a folha sobre a mesa)
python main.py

# Outras fontes de vídeo: outro índice de webcam ou um stream de rede
python main.py --source 1
python main.py --source http://192.168.0.10:8080/video
```

## Rodando no WSL (Windows)

O WSL2 **não enxerga a webcam do Windows por padrão**: não existe nenhum
`/dev/video*` dentro dele, então `python main.py` falha com
"Camera index out of range / nb_devices is 0". Isso não é um bug do Draw —
é o WSL sem acesso ao hardware de câmera. Como todo o processamento e o ML
futuro vivem no Linux, a solução é separar **captura** de **processamento**:

- **Recomendado — câmera via rede:** qualquer câmera que transmita MJPEG
  pela rede vira fonte do Draw, sem instalar nada no Windows. A opção mais
  simples é um celular sobre a mesa com um app de IP camera (ex.: IP Webcam
  no Android), que transmite em `http://<ip-do-celular>:8080/video`:
  `python main.py --source http://<ip-do-celular>:8080/video`
  De quebra, o celular num suporte acima da folha enxerga o papel melhor
  que a webcam do notebook.
- **Webcam do Windows via usbipd — um comando só:** para usar a webcam do
  notebook ou uma webcam USB dentro do WSL, rode (de dentro do WSL):
  `./scripts/setup_camera_wsl.sh`
  O script confere o driver `uvcvideo`, instala o que falta (ferramentas
  USB no WSL e o usbipd-win no Windows), acha sua câmera, compartilha e
  anexa ela ao WSL, e confirma que `/dev/video*` apareceu. Ele é
  idempotente — depois de reiniciar o WSL, rode o mesmo comando de novo.
  Se o `attach` falhar com "Device busy", o script limpa registros antigos
  de attach nos dois lados e tenta de novo com `bind --force`. Se ainda
  falhar, ele lê o log de eventos do Windows. Esse log nomeia o componente
  que bloqueou a remoção da câmera. O script para esse componente uma vez
  e faz uma última tentativa de attach. Se mesmo assim falhar, ele imprime
  o texto completo do bloqueio. Esse texto é o diagnóstico.
  A única interação que sobra é clicar **Sim** na confirmação de
  administrador do Windows (UAC) na primeira vez: nenhum script pode
  passar por ela, por design de segurança do Windows. Depois é só
  `python main.py`. Atenção: se o kernel do WSL não tiver o módulo
  `uvcvideo`, o script avisa e para (a alternativa seria um kernel
  customizado — caminho pesado e frágil). Enquanto anexada ao WSL, a
  câmera fica indisponível para o Windows.

  A v6 do script corrige a leitura desse log e vai um nível abaixo. O
  provedor certo dos eventos é `usbipd-win`. Quando o veto nomeia uma
  função de uma câmera composta, como a função `MI_02`, o script extrai
  esse identificador do texto e identifica a função com `Get-PnpDevice`
  antes de agir. Se a função for de áudio, ele para os serviços de
  áudio do Windows e tenta o attach. Se ainda falhar, ele desabilita
  só essa função, sem tocar na função de vídeo, e confere o status. O
  attach final só é tentado se a desabilitação tiver pegado. Se a função for de
  vídeo, ele para o serviço FrameServer. Se a classe for outra, ele
  imprime a identificação e para, porque não há ação segura conhecida
  para ela.

Teclas dentro do app: `s` salva um PNG em `captures/` · `q` ou `Esc` sai.

Sem webcam por perto, dá para validar o pipeline inteiro com uma cena
sintética (folha inclinada, desenhada por código):

```bash
python tests/selftest.py
```

## Estrutura

```
main.py                     ponto de entrada
draw/
  config.py                 tamanho da folha, marcadores, posições
  paper.py                  detecção dos marcadores + homografia (perspectiva)
  ink.py                    extração da tinta e renderização do canvas digital
  app.py                    loop da webcam e janelas
template/
  generate_template.py      gera a folha-modelo imprimível (PNG)
tests/
  selftest.py               teste do pipeline sem webcam
```

## Como funciona (resumo técnico)

1. Quatro marcadores ArUco impressos nos cantos da folha são detectados em
   cada frame da webcam.
2. Com os centros dos marcadores, uma homografia mapeia a folha inclinada
   para uma imagem retangular perfeita (vista de cima).
3. Sobre essa imagem, um limiar adaptativo separa a tinta escura do papel
   branco, ignorando as áreas dos marcadores. O resultado é o canvas
   digital.

A IA (redes neurais, transformers, LLM) entra nas próximas camadas —
interpretação dos traços, integração com CAD/simuladores e feedback de
simulação — descritas em [VISION.md](VISION.md). A geometria da v0 é
visão computacional clássica (OpenCV) de propósito: é o caminho mais
rápido, confiável e em tempo real para derrubar a primeira barreira.

## Roadmap

- [x] v0.1 — captura: folha retificada + canvas digital em tempo real
- [x] v0.1.1 — fonte de vídeo flexível (`--source` índice ou URL) para rodar no WSL com câmera via rede
- [ ] v0.2 — traço vetorial: ordem, caminho e suavização dos traços
- [ ] v0.3 — reprodução dos traços no Paint via controle do mouse
- [ ] v1 — interpretação: formas, letras e diagramas a partir dos traços
- [ ] v2 — integração com um primeiro sistema alvo (CAD / simulador)
- [ ] v3 — feedback de simulação em tempo real
