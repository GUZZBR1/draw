# Protocolo do ajudante Windows (v1)

Especificação do canal entre o ajudante residente no Windows e o Draw
rodando no WSL. Este documento é o contrato: o receptor de referência
está em [`../tools/helper_receiver.py`](../tools/helper_receiver.py) e
o esboço do ajudante, em [`windows_helper.py`](windows_helper.py).

## Por que este canal existe

Hoje a câmera entra no WSL via usbipd, e essa rota é frágil: os
serviços de áudio do Windows (Audiosrv e AudioEndpointBuilder)
seguram a função de microfone da câmera e a anexação falha com
`Device busy`. A ponte ffmpeg atual desta pasta
([`capture_windows.bat`](capture_windows.bat)) resolve a captura, mas
entrega um stream MJPEG sobre HTTP sem identidade de frame e sem
carimbo de tempo: o WSL recebe imagens, não frames datados.

O desenho aprovado troca as duas rotas por um ajudante residente no
Windows que captura via DirectShow/MSMF, carimba cada frame **na
captura** e envia JPEG por um socket local. Todo o processamento e
todo o ML continuam no WSL. Nada de ML roda no Windows.

Sem carimbo de captura não há como medir a latência vidro→tinta, que
nunca foi medida ponta a ponta neste projeto. Este protocolo existe
para tornar essa medida possível (ver "Passo 0", abaixo).

## Transporte

- TCP, um único sentido na v1: **ajudante → WSL**.
- O ajudante **escuta** numa porta TCP (padrão **8765**, configurável)
  e o WSL **conecta**. O ajudante é residente e vive mais que o app;
  o lado do WSL reinicia com frequência durante o desenvolvimento,
  então é ele quem inicia a conexão.
- No WSL2, o IP do host Windows é o `nameserver` de
  `/etc/resolv.conf`, como na ponte atual.
- O firewall do Windows deve liberar a porta em redes privadas, como
  já acontece com a ponte ffmpeg.
- Sem TLS e sem autenticação na v1: o canal é local (host Windows ↔
  WSL do mesmo PC) e o desenho assume esse perímetro. Não exponha a
  porta fora da máquina.

## Registro de frame

O fluxo é uma sequência de registros, sem separadores. Cada registro
é um **cabeçalho binário de tamanho fixo (28 bytes)** seguido do
payload JPEG. Todos os inteiros são **little-endian**, sem padding
além do declarado na tabela.

| Desloc. | Tamanho | Tipo   | Campo           | Valor na v1 |
|--------:|--------:|--------|-----------------|-------------|
| 0       | 4       | bytes  | `magic`         | `DRAW` (0x44 0x52 0x41 0x57) |
| 4       | 2       | uint16 | `version`       | 1 |
| 6       | 2       | uint16 | `reserved`      | 0 (ignorar na leitura) |
| 8       | 8       | uint64 | `frame_id`      | contador por sessão, começa em 0 |
| 16      | 8       | int64  | `capture_ts_us` | microssegundos, ver âncora abaixo |
| 24      | 4       | uint32 | `payload_len`   | tamanho do JPEG em bytes |
| 28      | `payload_len` | bytes | payload  | um frame JPEG completo |

Em Python, o cabeçalho é `struct.Struct("<4sHHQqI")`, 28 bytes.

Regras do fluxo:

- `frame_id` é estritamente crescente dentro de uma sessão e nunca
  reutilizado nela. Lacunas são possíveis e significam frames que a
  captura descartou; o receptor **não** deve travar esperando um id.
- O payload é exatamente um JPEG autocontido (começa em `FF D8`,
  termina em `FF D9`). Resolução e qualidade são configuração do
  ajudante, fora de banda na v1; o receptor descobre a resolução
  decodificando o JPEG.
- Não há checksum na v1: o TCP já garante integridade e ordem no
  transporte, e o arquivo gravado (abaixo) é escrito e lido na mesma
  máquina. Um payload com `magic` inválido no registro seguinte
  indica corrupção ou dessincronia; o receptor da v1 **falha** com
  erro de protocolo, sem tentar ressincronizar.

## Âncora do timestamp

`capture_ts_us` é o relógio do ajudante no momento mais próximo que a
pilha de captura permite da entrega do frame ao ajudante, expresso em
microssegundos sobre um **relógio monotônico cuja época é o início do
processo do ajudante** (no Windows, `QueryPerformanceCounter`
convertido; em Python, `time.perf_counter_ns() // 1000` tem a mesma
semântica).

Ressalvas, que são parte do contrato:

- Relógio monotônico não é relógio de parede: o valor só serve para
  **diferenças dentro da mesma sessão do ajudante** (intervalos entre
  frames, ordem, jitter da fonte).
- **Relógios de máquinas diferentes não se comparam sem uma âncora
  comum.** `t_receive_us - capture_ts_us` entre WSL e Windows mede o
  deslocamento entre os dois relógios, não a latência. Latência ponta
  a ponta se mede com a âncora comum do passo 0: um cronômetro de
  milissegundos filmado pela própria câmera, cuja leitura viaja
  dentro da imagem.
- Se o ajudante reiniciar, a época muda e `frame_id` volta a 0. Uma
  sessão nova é uma época nova; séries de sessões diferentes não se
  misturam sem registrar a troca.

## Arquivo gravado (replay)

O formato de gravação é **a mesma sequência de registros** que o
socket carrega: cabeçalho de 28 bytes + payload, repetidos, sem
cabeçalho de arquivo e sem rodapé. Um arquivo gravado é, byte a byte,
o que passou pelo socket.

Consequência deliberada: o receptor não distingue a fonte. O mesmo
código que lê o socket lê o arquivo, e qualquer análise (inclusive o
passo 0) roda igual sobre a sessão ao vivo e sobre o replay.

- Fim de arquivo no meio de um registro: os registros completos
  anteriores são válidos; o receptor reporta o rabo truncado como
  erro, não como frame.
- No socket, o fechamento da conexão encerra a iteração da mesma
  forma que o fim do arquivo.

## Passo 0 (a medida que este protocolo habilita)

Protocolo de medição da latência vidro→tinta, a rodar **duas vezes na
mesma régua**: uma com a câmera via usbipd (antes) e uma com o
ajudante (depois). Filma-se um cronômetro de milissegundos; para cada
frame gravam-se `frame_id`, `t_capture_us` (relógio da fonte),
`t_receive_us` (relógio do WSL) e, quando houver, `t_screen_ms` (a
leitura do cronômetro dentro daquele frame). A ferramenta é
[`../tools/step0_latency.py`](../tools/step0_latency.py).

A métrica que decide é o **jitter, p95 − mediana**, sobre **200+
frames**; a mediana sozinha não decide. O erro de leitura da tela
(±16 ms a 60 Hz) é comum às duas corridas e cancela na comparação.

## O que a v1 é e o que fica para depois

A v1 é só isto: **fluxo de frames num sentido só, ajudante → WSL**,
com o gravador e o replay acima. Explicitamente fora da v1:

- **Ack e controle de fluxo**: o ajudante envia na cadência da
  captura; se o WSL atrasar, o TCP segura o excesso e a latência
  aparece no passo 0, não some. Nenhum frame é confirmado.
- **Canal de controle WSL → ajudante**: iniciar/parar, trocar
  resolução e qualidade, pedir estado. Hoje isso é configuração
  manual do ajudante.
- **Injeção de cursor**: o esboço do ajudante já declara a injeção
  via `SendInput` (coordenadas absolutas normalizadas com desktop
  virtual), mas o transporte dos comandos de cursor **não** faz parte
  deste protocolo na v1. Será um canal próprio quando a ponte de
  cursor for construída.
- **Troca de âncora de relógio** (o ajudante declarar sua época em
  relógio de parede): só entra se um consumidor real precisar
  comparar relógios entre máquinas sem o cronômetro filmado.
- **Compressão alternativa** (H.264, JPEG por blocos): a v1 é JPEG
  frame a frame, igual à ponte atual, para o replay ser trivial.

Mudanças incompatíveis sobem `version`. Campos novos entram primeiro
no `reserved` ou em versão nova de cabeçalho, nunca mudando o
significado de um campo existente.
