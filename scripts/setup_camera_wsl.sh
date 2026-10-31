#!/usr/bin/env bash
# Draw. Conecta a webcam do Windows ao WSL em um comando so.
#
# Uso, de dentro do WSL e na raiz do projeto:
#   ./scripts/setup_camera_wsl.sh
#   ./scripts/setup_camera_wsl.sh --busid 1-4
#
# O script serve para qualquer maquina Windows com WSL2. Nada nele
# assume uma camera, um BUSID ou um fabricante especificos.
# Ele confere o WSL (modulos vhci_hcd e uvcvideo) e as ferramentas.
# Confere o usbipd-win no Windows e imprime a versao dele.
# Acha a camera na lista do usbipd. Se o nome na lista for ambiguo,
# confirma pelas classes Camera e Image do Get-PnpDevice.
# Com mais de uma camera, pergunta qual usar. Sem nenhuma, para com
# uma mensagem clara e um bloco de diagnostico.
# Compartilha (bind) e anexa (attach) a camera. Depois confere
# /dev/video* e a permissao do usuario (grupo video).
# Se o attach falhar com Device busy, o script le o log de eventos
# do Windows. Esse log nomeia a funcao do dispositivo que vetou a
# remocao. O script identifica essa funcao com Get-PnpDevice antes
# de agir. Funcao de audio (MEDIA ou Audio) faz o script parar os
# servicos de audio do Windows e tentar o attach de novo. Com o
# attach feito, ele tenta desabilitar essa funcao de audio de vez,
# religa os servicos de audio e confere se a camera sobreviveu.
# Funcao de video (Camera ou Image) vira parada do servico
# FrameServer. Filtro USB incompativel vira instrucao de desinstalar.
# No fim roda o teste de aceitacao: python3 main.py --check.
# Se algo falhar sem saida, o script imprime um bloco de diagnostico
# pronto para copiar e colar num pedido de ajuda.
#
# Rodar de novo verifica o estado atual. Apos reiniciar o WSL, o attach
# da camera pode precisar ser refeito; outras conexoes USB sao preservadas.
# DRAW_DRY_RUN=1 mostra os comandos sem executar nada.

set -uo pipefail

SCRIPT_VERSION="v9"
DRY_RUN="${DRAW_DRY_RUN:-0}"
BUSID_ARG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --busid) BUSID_ARG="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Uso: $0 [--busid 1-4] [--dry-run]" >&2; exit 2 ;;
  esac
done

echo "Draw. Setup da camera no WSL. Script $SCRIPT_VERSION."

# Raiz do projeto: a pasta acima de scripts/. O teste de aceitacao
# roda a partir dela, nao importa de onde o script foi chamado.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# DRAW_VIDEO_DIR so existe para os testes com stubs. Padrao: /dev.
VIDEO_DIR="${DRAW_VIDEO_DIR:-/dev}"

USBIPD=""
LIST_OUT=""
VETO_OUT=""

say()  { printf '\n==> %s\n' "$*"; }
warn() { printf 'AVISO: %s\n' "$*" >&2; }

print_diagnostics() {
  echo
  echo "===== DIAGNOSTICO (copie e cole tudo) ====="
  echo "Script: setup_camera_wsl.sh $SCRIPT_VERSION"
  echo "Kernel do WSL: $(uname -r 2>/dev/null || echo desconhecido)"
  echo "Usuario e grupos: $(id 2>/dev/null || echo desconhecido)"
  echo "--- usbipd list ---"
  if [ -n "$USBIPD" ]; then
    if command -v timeout >/dev/null 2>&1; then
      timeout 15 "$USBIPD" list 2>&1 | tr -d '\r' || echo "(usbipd list falhou)"
    else
      "$USBIPD" list 2>&1 | tr -d '\r' || echo "(usbipd list falhou)"
    fi
  else
    echo "(usbipd-win nao foi localizado nesta execucao)"
  fi
  echo "--- dispositivos de video no WSL ---"
  if ls "$VIDEO_DIR"/video* >/dev/null 2>&1; then
    ls -l "$VIDEO_DIR"/video*
  else
    echo "(nenhum dispositivo de video)"
  fi
  echo "--- ultimo veto registrado no log do Windows ---"
  if [ -n "$VETO_OUT" ]; then
    printf '%s\n' "$VETO_OUT"
  else
    echo "(nenhum veto foi lido nesta execucao)"
  fi
  echo "===== FIM DO DIAGNOSTICO ====="
}

die() {
  printf 'ERRO: %s\n' "$*" >&2
  print_diagnostics
  exit 1
}

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

video_devices() {
  ls "$VIDEO_DIR"/video* 2>/dev/null
}

# --- 1. Preflight do WSL ------------------------------------------------------

if [ "${DRAW_ASSUME_WSL:-0}" != "1" ]; then
  grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null \
    || die "Este script e para rodar de dentro do WSL. Fora dele, 'python main.py' usa a webcam direto."
fi

module_available() { # $1 = nome do modulo, com _ ou -
  if command -v modinfo >/dev/null 2>&1 && modinfo "$1" >/dev/null 2>&1; then
    return 0
  fi
  [ -d "/sys/module/$1" ] && return 0
  [ -d "/sys/module/${1//_/-}" ] && return 0
  local builtin="/lib/modules/$(uname -r)/modules.builtin"
  [ -f "$builtin" ] || return 1
  grep -q "$1" "$builtin" 2>/dev/null && return 0
  grep -q "${1//_/-}" "$builtin" 2>/dev/null && return 0
  return 1
}

say "Kernel do WSL: modulos vhci_hcd (USB virtual) e uvcvideo (camera)..."
if ! module_available vhci_hcd; then
  die "vhci_hcd NAO encontrado neste kernel. Sem ele o WSL nao recebe dispositivos USB virtuais:
     rode 'wsl --update' no PowerShell, reinicie o WSL e tente de novo;
     se continuar faltando, use a rota reserva da ponte MJPEG (pasta bridge/)."
fi
if ! module_available uvcvideo; then
  die "uvcvideo NAO encontrado neste kernel. O usbipd sozinho nao resolve:
     seria preciso um kernel customizado do WSL com suporte a camera.
     Rode 'wsl --update' no PowerShell, reinicie o WSL e tente de novo;
     se continuar faltando, use a rota reserva da ponte MJPEG (pasta bridge/)."
fi
echo "vhci_hcd e uvcvideo presentes. Ok."

say "Ferramentas USB no WSL (usbip, lsusb)..."
tools_ok() {
  command -v lsusb >/dev/null 2>&1 && command -v usbip >/dev/null 2>&1
}
if tools_ok; then
  echo "Ja instaladas. Ok."
else
  run sudo apt-get update
  run sudo apt-get install -y linux-tools-generic hwdata usbutils
  if ! command -v usbip >/dev/null 2>&1; then
    for cand in /usr/lib/linux-tools/*/usbip; do
      if [ -x "$cand" ]; then
        run sudo update-alternatives --install /usr/local/bin/usbip usbip "$cand" 20
        break
      fi
    done
  fi
  tools_ok || warn "usbip nao ficou no PATH. O setup continuara sem a ferramenta de inspecao USB local."
fi

# --- 2. Preflight do Windows ---------------------------------------------------

say "PowerShell e usbipd-win (lado Windows)..."
# Alguns ambientes WSL iniciam o shell sem acrescentar os caminhos do
# Windows ao PATH. A interoperabilidade ainda pode estar ativa e o exe
# existir no caminho padrao; nesse caso, use-o sem classificar a interop
# como desligada.
if ! command -v powershell.exe >/dev/null 2>&1; then
  for candidate in \
    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe" \
    "/mnt/c/Windows/SysWOW64/WindowsPowerShell/v1.0/powershell.exe" \
    "/mnt/c/Windows/System32/powershell.exe"; do
    if [ -x "$candidate" ]; then
      export PATH="$(dirname "$candidate"):$PATH"
      break
    fi
  done
fi
command -v powershell.exe >/dev/null 2>&1 \
  || die "powershell.exe nao foi encontrado no PATH nem nos caminhos padrao do Windows montado em /mnt/c. Confira se a interop do WSL esta ativa e se o Windows esta montado."

if command -v usbipd.exe >/dev/null 2>&1; then
  USBIPD="$(command -v usbipd.exe)"
elif [ -f "/mnt/c/Program Files/usbipd-win/usbipd.exe" ]; then
  USBIPD="/mnt/c/Program Files/usbipd-win/usbipd.exe"
fi

if [ -z "$USBIPD" ]; then
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
USBIPD_VERSION="$("$USBIPD" --version 2>/dev/null | tr -d '\r' | head -1)"
if [ -n "$USBIPD_VERSION" ]; then
  echo "usbipd-win: $USBIPD_VERSION"
else
  echo "usbipd-win encontrado em $USBIPD (a versao nao foi informada)."
fi

to_winpath() { # /mnt/c/Program Files/... -> C:\Program Files\...
  echo "$1" | sed -e 's|^/mnt/\([a-z]\)/|\U\1:\\|' -e 's|/|\\|g'
}

# --- 3. Descoberta da camera ---------------------------------------------------

say "Procurando sua camera na lista do usbipd..."
LIST_OUT="$("$USBIPD" list 2>/dev/null | tr -d '\r')"
[ -n "$LIST_OUT" ] || die "usbipd list nao retornou nada. A webcam esta conectada ao Windows?"

# Linhas da secao de dispositivos conectados (antes de "Persisted:").
connected_lines() {
  printf '%s\n' "$LIST_OUT" | awk '
    /^Persisted:/ { stop=1 }
    !stop && /^[0-9]+-[0-9]+/ { print }'
}

CAND_LINES=()
CAND_BUSIDS=" "
add_candidate() { # $1 = linha completa da lista do usbipd
  local b
  b="$(printf '%s' "$1" | awk '{print $1}')"
  [ -n "$b" ] || return 0
  case "$CAND_BUSIDS" in
    *" $b "*) return 0 ;;
  esac
  CAND_BUSIDS="$CAND_BUSIDS$b "
  CAND_LINES+=("$1")
}

# Primeira passada: nomes que parecem camera na propria lista.
while IFS= read -r line; do
  lower="$(printf '%s' "$line" | tr '[:upper:]' '[:lower:]')"
  case "$lower" in
    *camera*|*webcam*|"web cam"*|*video*|*uvc*|*facetime*|"user facing"*|"hd cam"*)
      add_candidate "$line" ;;
  esac
done <<EOF
$(connected_lines)
EOF

# Segunda passada, so se a lista nao resolveu sozinha: confirmar pelas
# classes Camera e Image do Windows e casar VID:PID com a lista.
if [ "${#CAND_LINES[@]}" -ne 1 ] && [ "$DRY_RUN" != "1" ]; then
  echo "Conferindo as classes Camera e Image no Windows (Get-PnpDevice)..."
  pnp_ps='$d = Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $_.Class -eq "Camera" -or $_.Class -eq "Image" }; if ($d) { $d | ForEach-Object { "$($_.Class)|$($_.FriendlyName)|$($_.InstanceId)" } }'
  PNP_LIST="$(powershell.exe -NoProfile -Command "$pnp_ps" 2>/dev/null | tr -d '\r')"
  if [ -n "$PNP_LIST" ]; then
    while IFS= read -r pnp_line; do
      [ -n "$pnp_line" ] || continue
      instance="$(printf '%s' "$pnp_line" | awk -F'|' '{print $3}')"
      vid="$(printf '%s' "$instance" | grep -oiE 'VID_[0-9A-F]{4}' | head -1 | sed 's/VID_//I' | tr '[:upper:]' '[:lower:]')"
      pid="$(printf '%s' "$instance" | grep -oiE 'PID_[0-9A-F]{4}' | head -1 | sed 's/PID_//I' | tr '[:upper:]' '[:lower:]')"
      [ -n "$vid" ] && [ -n "$pid" ] || continue
      while IFS= read -r line; do
        lower="$(printf '%s' "$line" | tr '[:upper:]' '[:lower:]')"
        case "$lower" in
          *"$vid:$pid"*) add_candidate "$line" ;;
        esac
      done <<EOF2
$(connected_lines)
EOF2
    done <<EOF3
$PNP_LIST
EOF3
  fi
fi

BUSID=""
STATE=""
state_of_line() { # $1 = linha da lista
  case "$1" in
    *"Not shared"*) echo "Not shared" ;;
    *Attached*)     echo "Attached" ;;
    *Shared*)       echo "Shared" ;;
    *)              echo "" ;;
  esac
}

if [ -n "$BUSID_ARG" ]; then
  BUSID="$BUSID_ARG"
  while IFS= read -r line; do
    if [ "$(printf '%s' "$line" | awk '{print $1}')" = "$BUSID" ]; then
      STATE="$(state_of_line "$line")"
    fi
  done <<EOF
$(connected_lines)
EOF
else
  if [ "${#CAND_LINES[@]}" -eq 0 ]; then
    die "Nenhuma camera foi encontrada. Confira se a camera esta conectada ao Windows e se nenhum aplicativo a esta usando. Para apontar manualmente, rode 'usbipd list' no PowerShell, ache o BUSID da camera e rode: $0 --busid <BUSID>"
  fi
  if [ "${#CAND_LINES[@]}" -gt 1 ] && [ -t 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "Mais de uma camera encontrada:"
    printf '%s\n' "${CAND_LINES[@]}" | cat -n
    printf 'Qual numero? '
    read -r CHOICE
    case "$CHOICE" in
      ''|*[!0-9]*) die "Escolha invalida: '$CHOICE'." ;;
    esac
    [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le "${#CAND_LINES[@]}" ] \
      || die "Escolha fora da lista: $CHOICE."
    CHOSEN="${CAND_LINES[$((CHOICE-1))]}"
  else
    [ "${#CAND_LINES[@]}" -gt 1 ] && warn "Varias cameras e sessao nao interativa: usando a primeira."
    CHOSEN="${CAND_LINES[0]}"
  fi
  BUSID="$(printf '%s' "$CHOSEN" | awk '{print $1}')"
  STATE="$(state_of_line "$CHOSEN")"
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
AUDIO_STOPPED=0
AUDIO_DISABLE_PENDING=0
AUDIO_FUNCTION_DISABLED=0
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

VETO_INSTANCE=""
PNP_OUT=""
PNP_CLASS=""
PNP_NAME=""

stop_frameserver() {
  say "Parando o servico FrameServer no Windows."
  run powershell.exe -NoProfile -Command \
    "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-Command','Stop-Service -Name FrameServer -Force -ErrorAction SilentlyContinue') -Verb RunAs -Wait"
}

stop_audio_services() {
  run powershell.exe -NoProfile -Command \
    "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-Command','Stop-Service -Name Audiosrv -Force -ErrorAction SilentlyContinue') -Verb RunAs -Wait"
  run powershell.exe -NoProfile -Command \
    "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-Command','Stop-Service -Name AudioEndpointBuilder -Force -ErrorAction SilentlyContinue') -Verb RunAs -Wait"
}

verify_audio_services_stopped() {
  local ps_cmd="Get-Service -Name Audiosrv, AudioEndpointBuilder -ErrorAction SilentlyContinue | ForEach-Object { \"\$(\$_.Name) \$(\$_.Status)\" }"
  local out
  out="$(powershell.exe -NoProfile -Command "$ps_cmd" 2>/dev/null | tr -d '\r')"
  echo "Status dos servicos de audio depois da tentativa de parar:"
  printf '%s\n' "$out"
  printf '%s\n' "$out" | grep -qE '^Audiosrv[[:space:]]+Stopped' \
    && printf '%s\n' "$out" | grep -qE '^AudioEndpointBuilder[[:space:]]+Stopped'
}

verify_function_disabled() {
  local ps_cmd="\$d = Get-PnpDevice -InstanceId '$VETO_INSTANCE*' -ErrorAction SilentlyContinue; if (\$d) { \$d | Select-Object Status, Problem | Format-List }"
  local out status problem
  out="$(powershell.exe -NoProfile -Command "$ps_cmd" 2>/dev/null | tr -d '\r')"
  echo "Status dessa funcao depois da tentativa de desabilitar:"
  printf '%s\n' "$out"
  [ -n "$out" ] || return 1
  status="$(printf '%s\n' "$out" | sed -n 's/^Status[[:space:]]*:[[:space:]]*//p' | head -1)"
  problem="$(printf '%s\n' "$out" | sed -n 's/^Problem[[:space:]]*:[[:space:]]*//p' | head -1)"
  if [ -n "$status" ] && [ "$status" != "OK" ]; then
    return 0
  fi
  printf '%s' "$problem" | grep -qi 'CM_PROB_DISABLED'
}

disable_vetoed_function() { # elevado; nao confere o resultado
  local disable_cmd="Disable-PnpDevice -InstanceId \"$VETO_INSTANCE\" -Confirm:\$false"
  run powershell.exe -NoProfile -Command \
    "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-Command','$disable_cmd') -Verb RunAs -Wait"
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
    say "Parando os servicos de audio do Windows. A confirmacao de administrador vai abrir duas vezes."
    stop_audio_services
    AUDIO_STOPPED=1
    verify_audio_services_stopped \
      || die "A parada dos servicos de audio nao pegou. Os status impressos acima sao a prova disso. Rode o script de novo e aceite as confirmacoes de administrador."
    say "Tentando o attach com os servicos de audio parados."
    if try_attach; then
      # Attach feito. Na etapa final o script tenta desabilitar essa
      # funcao de audio de vez e religa os servicos de audio.
      AUDIO_DISABLE_PENDING=1
      return 0
    fi
    attach_is_busy || return 1
    say "Desabilitando so essa funcao no Windows. A funcao de video fica intacta."
    disable_vetoed_function
    verify_function_disabled \
      || die "A funcao continua ativa depois da desabilitacao. O status impresso acima e a prova disso. Desabilite essa funcao no Gerenciador de Dispositivos do Windows e rode o script de novo."
    AUDIO_FUNCTION_DISABLED=1
    say "Ultima tentativa de attach depois de desabilitar a funcao vetada."
    try_attach
    return $?
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
  say "Preservando outras conexoes USB/IP; nenhuma porta sera desanexada globalmente."
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
            || die "O attach falhou de novo. Mande o bloco de diagnostico abaixo para quem for te ajudar. Ele nomeia o componente que bloqueia a camera."
        else
          die "O attach falhou (saida acima). Rode o script de novo; se persistir, mande o bloco de diagnostico abaixo."
        fi
      fi
    else
      die "O attach falhou (saida acima). Rode o script de novo; se persistir, mande o bloco de diagnostico abaixo."
    fi
  fi
fi

# --- 6. Verificacao, correcoes finais e aceitacao ------------------------------

say "Conferindo o resultado..."
if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] verificacao pulada."
  exit 0
fi

for _ in 1 2 3 4 5 6 7 8 9 10; do
  [ -n "$(video_devices)" ] && break
  sleep 1
done

if [ -z "$(video_devices)" ]; then
  die "A camera foi anexada mas nenhum dispositivo de video apareceu no WSL. Rode 'lsusb' e 'dmesg | tail' no WSL e mande a saida junto com o bloco de diagnostico abaixo."
fi

video_devices | while IFS= read -r dev; do ls -l "$dev"; done
if command -v lsusb >/dev/null 2>&1; then
  lsusb | grep -iE 'camera|webcam|uvc' || true
fi

FIRST_VIDEO="$(video_devices | head -1)"
if [ ! -r "$FIRST_VIDEO" ] || [ ! -w "$FIRST_VIDEO" ]; then
  say "Ajustando a permissao de $FIRST_VIDEO. Adicionando ${USER:-$(id -un)} ao grupo video."
  run sudo usermod -aG video "${USER:-$(id -un)}"
  echo "As proximas sessoes ja entram com a permissao."
  echo "Nesta sessao, rode o Draw assim: sg video -c \"python3 main.py\""
else
  echo "Permissao de $FIRST_VIDEO ok para o usuario ${USER:-$(id -un)}."
fi

if [ "$AUDIO_STOPPED" = "1" ]; then
  if [ "$AUDIO_DISABLE_PENDING" = "1" ] && [ "$AUDIO_FUNCTION_DISABLED" = "0" ]; then
    say "Tentando desabilitar de vez a funcao de audio que vetou o attach. So ela. A funcao de video fica intacta."
    disable_vetoed_function
    if verify_function_disabled; then
      AUDIO_FUNCTION_DISABLED=1
      echo "Funcao de audio desabilitada. Nas proximas vezes, o attach nao deve precisar parar o audio do Windows."
    else
      warn "A desabilitacao da funcao de audio nao pegou. O attach continua valendo. Numa proxima execucao o script repete a parada dos servicos."
    fi
  fi
  say "Religando o servico de audio do Windows (Audiosrv). Clique em Sim na confirmacao de administrador."
  run powershell.exe -NoProfile -Command \
    "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-Command','Start-Service Audiosrv') -Verb RunAs -Wait"
  sleep 1
  if [ -n "$(video_devices)" ]; then
    echo "Audio religado e a camera continua anexada: dispositivos de video presentes."
  else
    die "Depois de religar o audio, os dispositivos de video sumiram. O Windows retomou a camera. Rode o script de novo para anexar outra vez e mande o bloco de diagnostico abaixo se isso se repetir."
  fi
fi

say "Teste de aceitacao: python3 main.py --check"
CHECK_OUT="$(cd "$ROOT" && python3 main.py --check 2>&1)"
CHECK_STATUS=$?
printf '%s\n' "$CHECK_OUT"
if [ "$CHECK_STATUS" -ne 0 ]; then
  die "A camera anexou, mas o teste de aceitacao do Draw nao passou. A saida dele esta impressa acima. Mande essa saida junto com o bloco de diagnostico abaixo."
fi

say "Camera conectada ao WSL e teste de aceitacao aprovado. Agora e so: python3 main.py"
