#!/usr/bin/env bash
# Draw. Conecta a webcam do Windows ao WSL em um comando so.
#
# Uso, de dentro do WSL e na raiz do projeto:
#   ./scripts/setup_camera_wsl.sh
#   ./scripts/setup_camera_wsl.sh --busid 1-4
#
# O script confere o WSL e o driver uvcvideo. Instala o que falta.
# Acha a camera na lista do usbipd. Limpa registros antigos de attach.
# Compartilha e anexa a camera. Depois confere /dev/video*.
# Se o attach falhar com Device busy, o script le o log de eventos do Windows.
# Esse log nomeia a funcao do dispositivo que vetou a remocao.
# O script identifica essa funcao com Get-PnpDevice antes de agir.
# Funcao de audio aberta vira Disable-PnpDevice dessa funcao.
# Funcao de video aberta vira parada do servico FrameServer.
# Depois ele tenta o attach uma ultima vez.
#
# Rodar de novo e seguro. Apos reiniciar o WSL, so o attach precisa ser refeito.
# DRAW_DRY_RUN=1 mostra os comandos sem executar nada.

set -uo pipefail

DRY_RUN="${DRAW_DRY_RUN:-0}"
BUSID_ARG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --busid) BUSID_ARG="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Uso: $0 [--busid 1-4] [--dry-run]" >&2; exit 2 ;;
  esac
done

echo "Draw. Setup da camera no WSL. Script v5."

say()  { printf '\n==> %s\n' "$*"; }
warn() { printf 'AVISO: %s\n' "$*" >&2; }
die()  { printf 'ERRO: %s\n' "$*" >&2; exit 1; }

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

# --- 1. Ambiente ------------------------------------------------------------

if [ "${DRAW_ASSUME_WSL:-0}" != "1" ]; then
  grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null \
    || die "Este script e para rodar de dentro do WSL. Fora dele, 'python main.py' usa a webcam direto."
fi

say "Kernel do WSL: driver de camera (uvcvideo)..."
if modinfo uvcvideo >/dev/null 2>&1 || [ -d /sys/module/uvcvideo ] \
   || grep -q uvcvideo "/lib/modules/$(uname -r)/modules.builtin" 2>/dev/null; then
  echo "uvcvideo presente. Ok."
else
  die "uvcvideo NAO encontrado neste kernel. O usbipd sozinho nao resolve:
     seria preciso um kernel customizado do WSL com suporte a camera.
     Rode 'wsl --update' no PowerShell, reinicie o WSL e tente de novo;
     se continuar faltando, a rota recomendada e a camera via rede
     (veja a secao 'Rodando no WSL' do README)."
fi

# --- 2. Ferramentas ---------------------------------------------------------

say "Ferramentas USB no WSL (usbip, lsusb)..."
if ! command -v lsusb >/dev/null 2>&1 || ! ls /usr/lib/linux-tools/*/usbip >/dev/null 2>&1; then
  run sudo apt-get update
  run sudo apt-get install -y linux-tools-generic hwdata usbutils
  if ls /usr/lib/linux-tools/*/usbip >/dev/null 2>&1; then
    run sudo update-alternatives --install /usr/local/bin/usbip usbip /usr/lib/linux-tools/*/usbip 20
  fi
else
  echo "Ja instaladas. Ok."
fi

say "usbipd-win (lado Windows)..."
USBIPD=""
if command -v usbipd.exe >/dev/null 2>&1; then
  USBIPD="$(command -v usbipd.exe)"
elif [ -f "/mnt/c/Program Files/usbipd-win/usbipd.exe" ]; then
  USBIPD="/mnt/c/Program Files/usbipd-win/usbipd.exe"
fi

if [ -z "$USBIPD" ]; then
  command -v powershell.exe >/dev/null 2>&1 \
    || die "powershell.exe nao acessivel pelo WSL (interop desligado?). Instale o usbipd-win manualmente no Windows."
  echo "usbipd-win nao encontrado. Instalando via winget..."
  run powershell.exe -NoProfile -Command \
    "winget install --exact --id dorssel.usbipd-win --accept-source-agreements --accept-package-agreements"
  if [ -f "/mnt/c/Program Files/usbipd-win/usbipd.exe" ]; then
    USBIPD="/mnt/c/Program Files/usbipd-win/usbipd.exe"
  elif command -v usbipd.exe >/dev/null 2>&1; then
    USBIPD="$(command -v usbipd.exe)"
  else
    die "Instalacao do usbipd-win nao confirmada. Feche e reabra o terminal do WSL e rode o script de novo."
  fi
fi
echo "usbipd: $USBIPD"

to_winpath() { # /mnt/c/Program Files/... -> C:\Program Files\...
  echo "$1" | sed -e 's|^/mnt/\([a-z]\)/|\U\1:\\|' -e 's|/|\\|g'
}

# --- 3. Achar a camera --------------------------------------------------------

say "Procurando sua camera na lista do usbipd..."
LIST_OUT="$("$USBIPD" list 2>/dev/null | tr -d '\r')"
[ -n "$LIST_OUT" ] || die "usbipd list nao retornou nada. A webcam esta conectada ao Windows?"

# Linhas candidatas: antes da secao "Persisted:", com cara de camera.
CANDIDATES="$(printf '%s\n' "$LIST_OUT" | awk '
  /^Persisted:/ { stop=1 }
  !stop && /^[0-9]+-[0-9]+/ {
    line = tolower($0)
    if (line ~ /(camera|webcam|web cam|uvc|facetime|user facing|hd cam)/) print
  }')"

BUSID=""
STATE=""
if [ -n "$BUSID_ARG" ]; then
  BUSID="$BUSID_ARG"
  STATE="$(printf '%s\n' "$LIST_OUT" | awk -v b="$BUSID" '$1 == b { if (/Not shared/) print "Not shared"; else if (/Attached/) print "Attached"; else if (/Shared/) print "Shared" }')"
else
  [ -n "$CANDIDATES" ] || die "Nenhuma camera reconhecida em 'usbipd list'. Rode 'usbipd list' no PowerShell, ache o BUSID da camera e rode: $0 --busid <BUSID>"
  COUNT="$(printf '%s\n' "$CANDIDATES" | grep -c .)"
  if [ "$COUNT" -gt 1 ] && [ -t 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "Mais de uma candidata:"
    printf '%s\n' "$CANDIDATES" | cat -n
    printf 'Qual numero? '
    read -r CHOICE
    CHOSEN="$(printf '%s\n' "$CANDIDATES" | sed -n "${CHOICE}p")"
  else
    [ "$COUNT" -gt 1 ] && warn "Varias candidatas e sessao nao interativa: usando a primeira."
    CHOSEN="$(printf '%s\n' "$CANDIDATES" | head -1)"
  fi
  BUSID="$(printf '%s' "$CHOSEN" | awk '{print $1}')"
  case "$CHOSEN" in
    *"Not shared"*) STATE="Not shared" ;;
    *Attached*)     STATE="Attached" ;;
    *Shared*)       STATE="Shared" ;;
  esac
  echo "Camera: $CHOSEN"
fi
echo "BUSID=$BUSID estado=$STATE"

# --- 4/5. Bind (se preciso) + attach -----------------------------------------

bind_usbipd() { # $1 = 1 para --force (arranca a camera do Windows a forca)
  local args="@('bind','--busid','$BUSID')"
  if [ "${1:-0}" = "1" ]; then
    args="@('bind','--force','--busid','$BUSID')"
  fi
  say "Compartilhando (bind). Vai abrir a confirmacao de administrador do Windows. Clique em Sim."
  local win_usbipd
  win_usbipd="$(to_winpath "$USBIPD")"
  run powershell.exe -NoProfile -Command \
    "Start-Process -FilePath '$win_usbipd' -ArgumentList $args -Verb RunAs -Wait"
}

ATTACH_OUT=""
try_attach() {
  if [ "$DRY_RUN" = "1" ]; then
    run "$USBIPD" attach --wsl --busid "$BUSID"
    return 0
  fi
  ATTACH_OUT="$("$USBIPD" attach --wsl --busid "$BUSID" 2>&1)"
  local status=$?
  [ -n "$ATTACH_OUT" ] && printf '%s\n' "$ATTACH_OUT"
  return $status
}

attach_is_busy() {
  printf '%s' "$ATTACH_OUT" | grep -qi 'device busy'
}

clear_stale_state() {
  say "Limpando registros antigos de attach dos dois lados."
  if command -v usbip >/dev/null 2>&1; then
    if [ "$DRY_RUN" = "1" ]; then
      run sudo usbip port
    else
      local ports_out ports port
      ports_out="$(sudo usbip port 2>/dev/null || true)"
      echo "Portas usbip neste WSL agora:"
      printf '%s\n' "$ports_out"
      ports="$(printf '%s\n' "$ports_out" | sed -n 's/^Port \([0-9][0-9]*\): <Port in Use>.*/\1/p')"
      for port in $ports; do
        echo "Desanexando a porta usbip $port."
        run sudo usbip detach -p "$port"
      done
    fi
  else
    echo "usbip nao esta no PATH deste WSL. A limpeza de portas foi pulada."
  fi
  echo "Reiniciando o servico usbipd no Windows. Clique em Sim na confirmacao de administrador."
  run powershell.exe -NoProfile -Command \
    "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-Command','Restart-Service usbipd -Force') -Verb RunAs -Wait"
}

VETO_OUT=""
VETO_INSTANCE=""
PNP_OUT=""
PNP_CLASS=""
PNP_NAME=""

stop_frameserver() {
  say "Parando o servico FrameServer no Windows."
  run powershell.exe -NoProfile -Command \
    "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-Command','Stop-Service -Name FrameServer -Force -ErrorAction SilentlyContinue') -Verb RunAs -Wait"
}

identify_vetoed_function() {
  local ps_cmd="\$d = Get-PnpDevice -InstanceId '$VETO_INSTANCE*' -ErrorAction SilentlyContinue; if (\$d) { \$d | Select-Object Status, Class, FriendlyName | Format-List }"
  PNP_OUT="$(powershell.exe -NoProfile -Command "$ps_cmd" 2>/dev/null | tr -d '\r')"
  if [ -z "$PNP_OUT" ]; then
    return 1
  fi
  echo "Identificacao dessa funcao no Windows:"
  printf '%s\n' "$PNP_OUT"
  PNP_CLASS="$(printf '%s\n' "$PNP_OUT" | sed -n 's/^Class[[:space:]]*:[[:space:]]*//p' | head -1)"
  PNP_NAME="$(printf '%s\n' "$PNP_OUT" | sed -n 's/^FriendlyName[[:space:]]*:[[:space:]]*//p' | head -1)"
  if [ -z "$PNP_CLASS" ]; then
    PNP_CLASS="$(printf '%s\n' "$PNP_OUT" | grep -oiE '(^|[[:space:]])(Audio|MEDIA|Camera|Image)([[:space:]]|$)' | head -1 | tr -d '[:space:]')"
  fi
  if [ -z "$PNP_NAME" ] && [ -n "$PNP_CLASS" ]; then
    PNP_NAME="$(printf '%s\n' "$PNP_OUT" | awk -v c="$PNP_CLASS" 'toupper($2) == toupper(c) { $1 = ""; $2 = ""; sub(/^ +/, ""); sub(/ [^ ]+$/, ""); print; exit }')"
  fi
  [ -n "$PNP_CLASS" ]
}

act_on_vetoed_function() {
  local upper_class
  upper_class="$(printf '%s' "$PNP_CLASS" | tr '[:lower:]' '[:upper:]')"
  # A guarda vem antes do ramo de audio. Classe de video nunca e desabilitada.
  if [ "$upper_class" = "CAMERA" ] || [ "$upper_class" = "IMAGE" ]; then
    echo "A funcao vetada e de video. Classe $PNP_CLASS. Nome $PNP_NAME."
    stop_frameserver
  elif [ "$upper_class" = "AUDIO" ] || [ "$upper_class" = "MEDIA" ] \
       || printf '%s' "$PNP_NAME" | grep -qiE 'microphone|microfone|audio'; then
    echo "A funcao vetada e de audio. Classe $PNP_CLASS. Nome $PNP_NAME."
    say "Desabilitando so essa funcao no Windows. A funcao de video fica intacta."
    local disable_cmd="Disable-PnpDevice -InstanceId \"$VETO_INSTANCE\" -Confirm:\$false"
    run powershell.exe -NoProfile -Command \
      "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-Command','$disable_cmd') -Verb RunAs -Wait"
  else
    warn "A funcao vetada tem classe $PNP_CLASS e nome $PNP_NAME."
    warn "O script nao conhece uma acao segura para essa classe. A evidencia impressa acima decide o proximo passo."
    return 1
  fi
  say "Ultima tentativa de attach depois de agir sobre a funcao vetada."
  try_attach
}

handle_veto_and_retry() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] A leitura do log de eventos foi pulada."
    return 1
  fi
  say "Lendo o log de eventos do Windows. O servidor usbipd registra a funcao que vetou a remocao."
  local ps_cmd='$m = Get-WinEvent -FilterHashtable @{LogName = "Application"; ProviderName = "usbipd-win"} -MaxEvents 50 -ErrorAction SilentlyContinue | Where-Object { $_.Message -match "CR_REMOVE_VETOED" } | Select-Object -First 1; if ($m) { $m.Message }'
  VETO_OUT="$(powershell.exe -NoProfile -Command "$ps_cmd" 2>/dev/null | tr -d '\r')"
  if [ -z "$VETO_OUT" ]; then
    warn "Nenhum evento com CR_REMOVE_VETOED foi achado no log do usbipd-win."
    return 1
  fi
  echo "Texto completo do veto registrado pelo Windows:"
  printf '%s\n' "$VETO_OUT"

  if printf '%s' "$VETO_OUT" | grep -qiE 'USBPcap|UsbDk|TsUsbFlt|EUsbHubFilter' \
     || printf '%s' "$LIST_OUT" | grep -qi 'incompatible'; then
    die "Um filtro USB incompativel com o usbipd esta instalado no Windows. O texto acima nomeia o filtro. Desinstale no Windows o produto dono desse filtro e reinicie o computador. Depois rode o script de novo."
  fi

  VETO_INSTANCE="$(printf '%s' "$VETO_OUT" | grep -oE 'USB\\VID_[^[:space:]]+' | grep '&MI_' | head -1)"
  if [ -n "$VETO_INSTANCE" ]; then
    echo "Funcao vetada: $VETO_INSTANCE"
    if identify_vetoed_function; then
      act_on_vetoed_function
      return $?
    fi
    warn "O Get-PnpDevice nao achou essa funcao no Windows."
    warn "O texto do veto impresso acima e a evidencia que resta."
    return 1
  fi

  if printf '%s' "$VETO_OUT" | grep -qi 'FrameServer'; then
    stop_frameserver
  else
    local names target="" name lower svchost_seen=""
    names="$(printf '%s' "$VETO_OUT" | grep -oiE '[A-Za-z0-9_.-]+\.exe' | awk '!seen[tolower($0)]++')"
    while IFS= read -r name; do
      [ -z "$name" ] && continue
      lower="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
      if [ "$lower" = "svchost.exe" ]; then
        svchost_seen=1
      else
        target="$name"
        break
      fi
    done <<EOF
$names
EOF
    if [ -n "$target" ]; then
      local proc_name
      proc_name="$(printf '%s' "$target" | sed 's/\.[eE][xX][eE]$//')"
      say "O veto nomeia o processo $target. Parando esse processo no Windows."
      run powershell.exe -NoProfile -Command \
        "Stop-Process -Name '$proc_name' -Force -ErrorAction SilentlyContinue"
    elif [ -n "$svchost_seen" ]; then
      warn "O veto nomeia o svchost.exe. Esse processo hospeda servicos do Windows e nao deve ser encerrado."
      warn "O texto do veto impresso acima e a evidencia para agir. Ele nomeia o servico ou o driver que vetou a remocao."
      return 1
    else
      warn "O texto do veto nao nomeou um alvo que o script saiba parar."
      return 1
    fi
  fi

  say "Ultima tentativa de attach depois de parar o componente do veto."
  try_attach
}

if [ "$STATE" = "Attached" ]; then
  say "A camera ja esta anexada ao WSL. Nada a fazer aqui."
else
  clear_stale_state
  case "$STATE" in
    Shared*) : ;;  # ja compartilhada (normal ou forcada)
    *) bind_usbipd 0 ;;
  esac
  say "Anexando a camera ao WSL (attach)..."
  if ! try_attach; then
    if attach_is_busy; then
      warn "O Windows recusou o attach com Device busy."
      warn "Tentando de novo com bind --force."
      bind_usbipd 1
      if [ "$DRY_RUN" != "1" ]; then
        echo "Estado da camera apos o bind --force (o esperado e 'Shared (forced)'):"
        "$USBIPD" list 2>/dev/null | tr -d '\r' | grep "^$BUSID" || true
      fi
      say "Anexando de novo (attach, apos --force)..."
      if ! try_attach; then
        if attach_is_busy; then
          handle_veto_and_retry \
            || die "O attach falhou de novo. Me mande o texto completo do veto impresso acima nesta conversa. Ele nomeia o componente que bloqueia a camera."
        else
          die "O attach falhou (saida acima). Rode o script de novo; se persistir, me mostre a saida."
        fi
      fi
    else
      die "O attach falhou (saida acima). Rode o script de novo; se persistir, me mostre a saida."
    fi
  fi
fi

# --- Verificacao --------------------------------------------------------------

say "Conferindo o resultado..."
if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] verificacao pulada."
  exit 0
fi

for _ in 1 2 3 4 5 6 7 8 9 10; do
  ls /dev/video* >/dev/null 2>&1 && break
  sleep 1
done

if ls /dev/video* >/dev/null 2>&1; then
  ls -l /dev/video*
  command -v lsusb >/dev/null 2>&1 && lsusb | grep -iE 'camera|webcam|uvc' || true
  FIRST_VIDEO="$(ls /dev/video* | head -1)"
  if [ ! -r "$FIRST_VIDEO" ]; then
    warn "$FIRST_VIDEO sem permissao de leitura. Rode: sudo usermod -aG video $USER. Depois reinicie o WSL com wsl --shutdown no PowerShell."
  fi
  say "Camera conectada ao WSL. Agora e so: python main.py"
else
  die "A camera foi anexada mas /dev/video* nao apareceu. Rode 'lsusb' e 'dmesg | tail' no WSL e me mostre a saida."
fi
