#!/usr/bin/env bash
# Draw — conecta a webcam do Windows ao WSL em um comando só.
#
# Uso (de dentro do WSL, na raiz do projeto):
#   ./scripts/setup_camera_wsl.sh              # detecta a camera sozinho
#   ./scripts/setup_camera_wsl.sh --busid 1-4  # usa esse BUSID do usbipd
#
# O que ele faz, na ordem:
#   1. Confere se voce esta no WSL e se o kernel tem o driver uvcvideo
#   2. Instala as ferramentas USB no WSL (apt) e o usbipd-win no Windows
#      (winget), se faltarem
#   3. Acha a sua camera na lista do usbipd
#   4. Compartilha (bind) — abre UMA janela de confirmacao de administrador
#      do Windows (UAC); e a unica parte que nenhum script pode fazer por
#      voce, por design de seguranca do Windows
#   5. Anexa (attach) a camera ao WSL e confere /dev/video*
#
# O script e idempotente: pode rodar de novo em toda sessao (apos reiniciar
# o WSL, so o attach precisa ser refeito — e e o que ele faz sozinho).
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

echo "Draw — setup da camera no WSL (script v3: detecta 'Device busy' e tenta bind --force sozinho)"

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
  say "Compartilhando (bind) — vai abrir a confirmacao de administrador do Windows; clique em Sim."
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

if [ "$STATE" = "Attached" ]; then
  say "A camera ja esta anexada ao WSL. Nada a fazer aqui."
else
  case "$STATE" in
    Shared*) : ;;  # ja compartilhada (normal ou forcada)
    *) bind_usbipd 0 ;;
  esac
  say "Anexando a camera ao WSL (attach)..."
  if ! try_attach; then
    if printf '%s' "$ATTACH_OUT" | grep -qiE 'device busy|--force'; then
      warn "O Windows esta usando a camera agora (app de Camera, navegador, Teams/Zoom, Windows Hello...)."
      warn "Tentando de novo com bind --force, que desliga a camera do Windows a forca."
      bind_usbipd 1
      if [ "$DRY_RUN" != "1" ]; then
        echo "Estado da camera apos o bind --force (o esperado e 'Shared (forced)'):"
        "$USBIPD" list 2>/dev/null | tr -d '\r' | grep "^$BUSID" || true
      fi
      say "Anexando de novo (attach, apos --force)..."
      try_attach \
        || die "Mesmo com --force o attach falhou. Feche qualquer programa usando a camera no Windows e rode o script de novo."
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
    warn "$FIRST_VIDEO sem permissao de leitura. Rode: sudo usermod -aG video $USER — e reinicie o WSL (wsl --shutdown no PowerShell)."
  fi
  say "Camera conectada ao WSL. Agora e so: python main.py"
else
  die "A camera foi anexada mas /dev/video* nao apareceu. Rode 'lsusb' e 'dmesg | tail' no WSL e me mostre a saida."
fi
