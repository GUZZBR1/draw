# Ponte de captura no Windows (rota reserva)

Esta pasta é o plano B da câmera. A rota principal continua sendo a
webcam anexada ao WSL via usbipd, com
[`scripts/setup_camera_wsl.sh`](../scripts/setup_camera_wsl.sh). Use a
ponte só se essa rota falhar na sua máquina.

## Como funciona

Um único arquivo `ffmpeg.exe` roda no Windows, lê a webcam pelo
DirectShow e serve o vídeo como um stream MJPEG em
`http://0.0.0.0:8080/video`. O Draw continua rodando no WSL e consome
esse stream pela opção `--source`, que ele já aceita. Nada mais vai
para o Windows: sem Python, sem bibliotecas, só o `ffmpeg.exe` solto
ao lado do `capture_windows.bat`.

## Antes de começar

- A câmera **não pode estar anexada ao WSL** enquanto a ponte roda.
  Se você anexou com usbipd, solte antes, no PowerShell:
  `usbipd detach --busid <BUSID>` (o mesmo BUSID do `usbipd list`).
  Câmera anexada ao WSL some do Windows, e o ffmpeg não a encontra.
- Os serviços de áudio do Windows devem estar rodando normalmente
  nesta rota. Se a rota usbipd os parou, religue com
  `Start-Service Audiosrv` no PowerShell como administrador.

## Passo 1: baixar o ffmpeg

Baixe um build do ffmpeg para Windows e extraia só o `ffmpeg.exe`
para dentro desta pasta `bridge`, no lado Windows do disco:

- gyan.dev: https://www.gyan.dev/ffmpeg/builds/ (pacote "release
  full"; o `ffmpeg.exe` fica na pasta `bin` do arquivo extraído), ou
- builds do BtbN no GitHub: pacote `ffmpeg-master-latest-win64-gpl`
  (o `ffmpeg.exe` também fica na pasta `bin`).

O `capture_windows.bat` espera o `ffmpeg.exe` na mesma pasta que
ele. Sem isso, ele para e mostra este mesmo aviso.

## Passo 2: conferir o nome da câmera

Num prompt dentro desta pasta, liste os dispositivos de vídeo:

```bat
ffmpeg.exe -list_devices true -f dshow -i dummy
```

O nome da câmera aparece na lista. O padrão do `.bat` é
`1200W Camera`. Se a lista mostrar outro nome, abra o
`capture_windows.bat` num editor de texto e troque o valor da
variável `CAM`, no bloco marcado CONFIGURACAO, logo no topo.

## Passo 3: rodar a ponte

Rode o `capture_windows.bat` com duplo clique ou pelo prompt. Na
primeira execução, o Windows mostra o aviso do Firewall pedindo
permissão para o ffmpeg na rede: permita em **redes privadas**.
A janela fica aberta servindo o vídeo. Fechar a janela para a ponte.

## No WSL: conectar o Draw

O Draw alcança o Windows pelo IP que aparece como `nameserver` no
`/etc/resolv.conf` do WSL:

```bash
grep nameserver /etc/resolv.conf
```

Com esse IP, rode o Draw apontando para o stream da ponte:

```bash
python3 main.py --source http://<IP-DO-WINDOWS>:8080/video
```

Pronto: a captura acontece no Windows e todo o processamento, no
WSL, como no resto do projeto.
