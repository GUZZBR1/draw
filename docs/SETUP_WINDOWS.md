# Setup da câmera no Windows com WSL

Guia universal: vale para qualquer pessoa com Windows e WSL2 que
queira rodar o Draw com a própria webcam. Nenhum passo depende de
uma máquina, uma câmera ou um fabricante específicos.

## Requisitos

- Windows 10 ou Windows 11.
- WSL2 instalado e atualizado (`wsl --update` no PowerShell).
- Uma webcam USB ou a câmera integrada do notebook.
- O repositório do Draw clonado dentro do WSL.

## Instalação em um comando

De dentro do WSL, na raiz do projeto:

```bash
./scripts/setup_camera_wsl.sh
```

Só isso. O script pede confirmações de administrador do Windows
em alguns passos (é o UAC abrindo). Clique em **Sim**. No fim, ele
roda o teste de aceitação do próprio Draw e imprime o veredito.

Opções úteis:

- `./scripts/setup_camera_wsl.sh --busid 1-4` aponta uma câmera
  específica pelo BUSID dela no `usbipd list`.
- `./scripts/setup_camera_wsl.sh --dry-run` só mostra o que o
  script faria, sem executar nada.

Depois de reiniciar o WSL, rode o script de novo para conferir e,
se necessário, refazer o attach da câmera. As outras conexões USB/IP
do WSL são preservadas.

## O que o script faz, e por quê

1. **Confere o WSL.** Os módulos `vhci_hcd` (USB virtual) e
   `uvcvideo` (câmera) precisam existir no kernel. Sem eles, nada
   funciona. Instala `usbip` e `lsusb` se faltarem.
2. **Confere o Windows.** O `usbipd-win` precisa estar instalado.
   Se não estiver, o script instala via `winget` e imprime a
   versão encontrada.
3. **Acha a câmera.** Lê o `usbipd list`. Se o nome for ambíguo,
   confirma pelas classes Camera e Image do Windows. Com uma
   câmera só, escolhe sozinho. Com várias, pergunta pelo número.
   Sem nenhuma, para com uma mensagem clara.
4. **Compartilha e anexa.** Faz o `bind` e o `attach` da câmera
   para dentro do WSL sem desanexar outros dispositivos USB/IP.
5. **Confere o resultado.** Espera o dispositivo de vídeo aparecer
   no WSL e corrige a permissão: quem roda o Draw precisa estar no
   grupo `video`.
6. **Roda o teste de aceitação.** `python3 main.py --check` diz se
   o Draw enxerga e abre a câmera de verdade.

### Por que às vezes o Windows segura a câmera (o veto de áudio)

Muitas webcams são dispositivos compostos: num cabo só elas têm a
função de vídeo e a função de microfone. Os serviços de áudio do
Windows mantêm a função de microfone aberta o tempo todo. Quando
o usbipd tenta levar o dispositivo inteiro para o WSL, o Windows
recusa a remoção por causa dessa função aberta. O erro aparece
como **Device busy**.

Quando isso acontece, o script lê o log de eventos do próprio
usbipd-win. O log nomeia a função exata que vetou a remoção. Aí:

- Se a função for de **áudio**, o script para os serviços de áudio
  do Windows, anexa a câmera, tenta desabilitar só essa função de
  microfone (a de vídeo fica intacta) e religa os serviços de
  áudio, conferindo se a câmera sobreviveu com o áudio de volta.
  Com a função desabilitada, as próximas execuções nem precisam
  parar o áudio.
- Se a função for de **vídeo**, o script para o serviço
  FrameServer, que é quem costuma segurar câmeras no Windows, e
  tenta o attach de novo. Função de vídeo nunca é desabilitada.
- Se o veto nomear um **filtro USB incompatível** (drivers de
  captura de rede, por exemplo), o script nomeia o filtro e pede
  a desinstalação do produto dono dele. Não há correção segura
  por script nesse caso.

## Problemas conhecidos

| Sintoma | Causa provável | Correção |
| --- | --- | --- |
| Nenhuma câmera encontrada | Câmera desconectada, desligada ou em uso exclusivo por outro aplicativo | Conecte a câmera, feche aplicativos que a usam e rode o script de novo. Ou ache o BUSID com `usbipd list` no PowerShell e rode com `--busid <BUSID>`. |
| Device busy que sobrevive ao script | Um componente que o script não conhece está segurando a câmera | O script imprime um bloco de diagnóstico no fim da execução. Copie o bloco inteiro e mande para quem for te ajudar. Ele nomeia o componente do veto. |
| Permission denied ao abrir a câmera no Draw | Seu usuário do WSL ainda não está no grupo `video` nesta sessão | O script já adiciona o usuário ao grupo. Na sessão atual, rode `sg video -c "python3 main.py"`. Nas próximas sessões, a permissão já vale. |
| A câmera abre, mas não entrega quadros | Formato de vídeo pesado demais para o USB virtual | Atualize o repositório (`git pull`). O Draw atual já pede MJPEG em 1280x720 para a câmera local, que é o formato que passa bem pelo usbipd. |
| Windows ficou sem som depois do setup | Os serviços de áudio foram parados durante o attach e não voltaram | No PowerShell como administrador: `Start-Service Audiosrv`. O script religa o áudio sozinho quando foi ele quem parou. Se religar o áudio derrubar a câmera, rode o script de novo. |
| Erro de módulo `vhci_hcd` ou `uvcvideo` faltando | Kernel do WSL antigo ou sem suporte a câmera | Rode `wsl --update` no PowerShell, reinicie o WSL e tente de novo. Persistindo, use a rota reserva abaixo. |

## Rota reserva

Se o usbipd não funcionar na sua máquina por qualquer motivo, a
pasta [`bridge/`](../bridge/) traz a ponte de captura: um
`ffmpeg.exe` no Windows serve a webcam como stream MJPEG e o Draw,
no WSL, consome esse stream com `--source`. Nenhum dispositivo USB
é movido entre os sistemas. Veja o `bridge/README.md` para o passo
a passo.
