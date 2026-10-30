"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --check                          # valida a camera e sai
  python main.py --markers                        # folha-modelo impressa
  python main.py --neural modelo.onnx             # detector neural primario
  python main.py --ink-model modelo.onnx          # modelo aprendido de tinta
  python main.py --hand-model modelo.task         # mascara de mao no canvas
  python main.py --source 1                       # outra webcam local
  python main.py --source http://192.168.0.10:8080/video   # camera via rede

A fonte via rede (MJPEG/RTSP) e o caminho recomendado no WSL. O WSL nao
enxerga webcams USB do Windows por padrao, mas enxerga streams de rede.
Assim todo o processamento (e o ML futuro) fica no Linux. A camera so
envia os frames. Ela pode ser um celular, outro computador ou uma camera IP.

Teclas: s = salvar o canvas como PNG, f = capturar um frame cru,
g = ligar ou desligar a captura automatica de frames,
c = congelar a pagina base (so com --session-baseline), q/Esc = sair.
"""

import argparse
import glob
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import hand as hand_module
from . import ink as ink_module
try:
    from . import ink_model as ink_model_module
except ImportError:
    # O extrator de tinta por modelo aprendido e opcional e o modulo
    # dele ainda nao faz parte do pacote publicado. Sem o modulo,
    # --ink-model avisa e o app segue so com a heuristica.
    ink_model_module = None
from . import neural as neural_module
from . import paper as paper_module


_SHEET_NEAR_LIMIT_PX = 40.0
_SHEET_BLEND_ALPHA = 0.3
# Confirmação de salto por TEMPO, não por contagem de detecções
# (medido em 2026-10-30 no laboratório,
# draw-lab/noite-2026-10-30/block6/): com a contagem de 3, um salto
# selvagem confirmava em ~0,1 s na cadência densa de "missed" (a
# detecção roda a cada quadro), e os vídeos de replay mediam 64
# passos oficiais acima de 100 px em 1.857 quadros. Exigir que o
# candidato persista 0,3 s de tempo de vídeo (o dobro para
# blank_maybe, como era o dobro na contagem) derruba os passos
# >100 px para 15 e o jitter médio da grade de 29,39 para 9,66 px,
# sem nenhuma queda nova e com o atraso de remoção intacto em
# 0,53 s. Ressalva: a medição é sobre gravações de tela com overlay
# queimado; reconfirmar em captura crua.
_SHEET_JUMP_PERSIST_SECONDS = 0.3
_SHEET_DETECT_INTERVAL = 5
# Perda da folha medida em tempo, nao em quadros: com a contagem em
# quadros, o mesmo limite valia ~1 s a 30 fps e ~3 s numa sessao lenta,
# e era esse o banner que ficava sobre a mesa vazia. O piso em quadros
# impede a queda entre duas deteccoes agendadas quando o loop roda
# devagar: a folha presente confirma de novo antes dos dois criterios.
_SHEET_LOST_SECONDS = 0.5
_SHEET_LOST_MIN_FRAMES = 10
# A tinta e lida numa geometria estabilizada: a mediana dos ultimos
# conjuntos de cantos do rastreador. O jitter de canto entre deteccoes
# (media de ~5 px, com saltos de dezenas de pixels quando a mao cobre
# um canto) desloca o traco de 1 a 3 px no espaco do papel e desfaz o
# voto por pixel do canvas acumulativo; a mediana de 5 segura a
# geometria no lugar por tempo bastante para a janela do acumulador.
_INK_CORNER_HISTORY = 5


def _corner_distance(current, detected):
    return float(np.max(np.linalg.norm(current - detected, axis=1)))


def _blend_corners(current, detected):
    return (
        (1.0 - _SHEET_BLEND_ALPHA) * current
        + _SHEET_BLEND_ALPHA * detected
    ).astype(np.float32)


class _SheetTracker:
    """Estabilizacao temporal do quadrilatero da folha.

    Guarda o quadrilatero atual e decide, quadro a quadro, entre
    misturar uma deteccao vizinha, exigir persistencia para um salto
    de posicao e descartar a folha perdida. Deteccao perto mistura
    com _SHEET_BLEND_ALPHA; salto (deteccao a mais de
    _SHEET_NEAR_LIMIT_PX do quad atual) so confirma quando o
    candidato persiste _SHEET_JUMP_PERSIST_SECONDS de relogio com
    deteccoes seguintes pousando perto dele (o dobro para
    blank_maybe). A perda dispara quando os dois criterios de
    _SHEET_LOST_* passam: tempo sem confirmacao e quadros sem
    confirmacao.

    Presenca e posicao sao perguntas separadas no relogio de perda:
    enquanto deteccoes "paper" plausiveis continuam chegando, a
    folha esta la, mesmo com a posicao incerta. Por isso um salto
    ainda nao confirmado com veredito "paper" renova o relogio de
    perda sem mover o quadrilatero exibido, e um quadro com
    veredito "paper" sem quadrilatero (o modelo afirmou papel e
    nenhum ajuste fechou) tambem renova. Sem isso, a espera da
    confirmacao por tempo esfomeia o relogio e a folha cai no meio
    da janela da mao (medido no replay: 11 quadros sem folha e
    atraso de remocao de 0,30 s, contra 0,53 s com a renovacao).
    """

    def __init__(self):
        self.corners = None
        self.verdict = None
        self.missed = False
        self._pending = None
        self._pending_hits = 0
        self._pending_since = None
        self._frames_since_seen = 0
        self._last_seen_at = None

    def should_detect(self, frame_number):
        return (
            self.corners is None
            or self.missed
            or frame_number % _SHEET_DETECT_INTERVAL == 0
        )

    def _confirm(self, corners, verdict, now):
        self.corners = corners
        self.verdict = verdict
        self._pending = None
        self._pending_hits = 0
        self._pending_since = None
        self._frames_since_seen = 0
        self._last_seen_at = now
        self.missed = False

    def update(self, detected, verdict, ran_detection, now):
        """Alimenta o rastreador com o resultado do quadro.

        detected e o quadrilatero detectado ou None; ran_detection
        diz se a deteccao rodou neste quadro; now e um relogio
        monotonico em segundos. Retorna "seen" quando uma deteccao
        confirma a folha, "drop" quando a folha e descartada e None
        nos demais quadros (so para diagnostico).
        """
        event = None
        if (
            detected is None
            and ran_detection
            and verdict == "paper"
            and self.corners is not None
        ):
            # Evento so de presenca: quem detectou afirma papel mas
            # nenhum quadrilatero utilizavel fechou (o ajuste neural
            # foi recusado pela guarda de plausibilidade, ou nao
            # fechou). A folha esta la; a posicao e que nao veio.
            # Renova o relogio de perda, nao mexe no quadrilatero
            # exibido nem no candidato pendente.
            self._frames_since_seen = 0
            self._last_seen_at = now
            self.missed = True
            return None
        if detected is not None:
            if self.corners is None and verdict == "paper":
                self._confirm(detected.copy(), verdict, now)
                event = "seen"
            elif (
                self.corners is not None
                and _corner_distance(self.corners, detected)
                <= _SHEET_NEAR_LIMIT_PX
            ):
                self._confirm(
                    _blend_corners(self.corners, detected), verdict, now
                )
                event = "seen"
            else:
                # Salto de posicao ou candidato ainda sem trava.
                # Candidato blank_maybe (branco e liso, sem textura
                # de papel) exige o dobro do tempo de persistencia.
                confirm_seconds = (
                    2 * _SHEET_JUMP_PERSIST_SECONDS
                    if verdict == "blank_maybe"
                    else _SHEET_JUMP_PERSIST_SECONDS
                )
                if (
                    self._pending is not None
                    and _corner_distance(self._pending, detected)
                    <= _SHEET_NEAR_LIMIT_PX
                ):
                    self._pending = _blend_corners(self._pending, detected)
                    self._pending_hits += 1
                else:
                    self._pending = detected.copy()
                    self._pending_hits = 1
                    self._pending_since = now

                if (
                    self._pending_since is not None
                    and now - self._pending_since >= confirm_seconds
                ):
                    self._confirm(self._pending.copy(), verdict, now)
                    event = "seen"
                else:
                    if verdict == "paper":
                        # Presenca renovada, posicao ainda incerta:
                        # o relogio de perda nao corre contra uma
                        # folha que o detector continua vendo.
                        self._frames_since_seen = 0
                        self._last_seen_at = now
                    else:
                        self._frames_since_seen += 1
                    self.missed = True
        elif self.corners is not None:
            if ran_detection:
                self._pending = None
                self._pending_hits = 0
                self._pending_since = None
                self.missed = True
            self._frames_since_seen += 1

        if (
            self.corners is not None
            and self._frames_since_seen > _SHEET_LOST_MIN_FRAMES
            and self._last_seen_at is not None
            and now - self._last_seen_at > _SHEET_LOST_SECONDS
        ):
            self.corners = None
            self.verdict = None
            self._pending = None
            self._pending_hits = 0
            self._pending_since = None
            self._frames_since_seen = 0
            self.missed = False
            event = "drop"
        return event


def parse_source(value):
    """'0'/'1' viram indice de webcam local; qualquer outra coisa e URL/arquivo."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def is_wsl():
    """Detecta WSL pelo nome do kernel em /proc (ex.: '...-microsoft-standard-WSL2')."""
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _configure_local_capture(capture):
    # MJPG comprimido e 720p: o formato cru padrao (YUYV em 1080p)
    # estoura a banda do USB virtual do usbip no WSL e a entrega de
    # quadros trava com timeout de select() no V4L2.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def _local_open_hint(source):
    """Diagnostico por porta: nomeia a primeira condicao que impede a abertura."""
    node = f"/dev/video{source}"
    if not sys.platform.startswith("linux"):
        return (
            "Nenhuma webcam local respondeu nesse indice. "
            "Confira se a webcam esta conectada e livre."
        )
    if not os.path.exists(node):
        hint = f"A camera nao esta conectada a este sistema: {node} nao existe. "
        if is_wsl():
            hint += (
                "Anexe a camera ao WSL com: bash scripts/setup_camera_wsl.sh. "
                "A alternativa e uma camera via rede: python main.py "
                "--source http://<ip-da-camera>:8080/video."
            )
        else:
            hint += "Confira se a webcam esta conectada a este computador."
        return hint
    if not os.access(node, os.R_OK | os.W_OK):
        return (
            f"{node} existe, mas este processo nao tem permissao de "
            "leitura e escrita nele. A porta que falhou e a permissao. "
            "Correcao definitiva: sudo usermod -aG video $USER "
            "(vale nas proximas sessoes). "
            'Correcao nesta sessao: sg video -c "python3 main.py".'
        )
    return (
        f"{node} existe e este processo tem permissao nele, mas a "
        "abertura falhou. Outro programa provavelmente esta usando a "
        "camera. Feche esse programa e tente de novo."
    )


def _open_capture(source):
    capture = cv2.VideoCapture(source)
    if capture.isOpened():
        if isinstance(source, int):
            _configure_local_capture(capture)
        return capture
    hint = f"Nao consegui abrir a fonte de video: {source!r}. "
    if isinstance(source, int):
        hint += _local_open_hint(source)
    else:
        hint += (
            "Confira se a URL esta certa, se a camera esta transmitindo e "
            "se este computador alcanca o endereco dela na rede."
        )
    raise RuntimeError(hint)


def _video_nodes():
    return sorted(
        path
        for path in glob.glob("/dev/video*")
        if path[len("/dev/video"):].isdigit()
    )


def run_check(source=0):
    """Validacao da camera local, porta por porta. Retorna 0 ou 1."""
    print(f"PASS: OpenCV disponivel. Versao {cv2.__version__}.")

    nodes = _video_nodes()
    if not nodes:
        print("FAIL: nenhum /dev/video* existe neste sistema.")
        if is_wsl():
            print(
                "  Correcao: anexe a camera com "
                "bash scripts/setup_camera_wsl.sh. A alternativa e uma "
                "camera via rede com --source."
            )
        else:
            print("  Correcao: conecte uma webcam a este computador.")
        return 1
    print(f"PASS: dispositivos de video encontrados: {', '.join(nodes)}.")

    all_ok = True
    node = nodes[0]

    access = os.access(node, os.R_OK | os.W_OK)
    group_note = ""
    try:
        import grp

        gid = os.stat(node).st_gid
        try:
            group_name = grp.getgrgid(gid).gr_name
        except KeyError:
            group_name = str(gid)
        in_group = gid in os.getgroups()
        group_note = (
            f" Grupo dono de {node}: {group_name}. "
            f"Este processo esta nesse grupo: {'sim' if in_group else 'nao'}."
        )
    except Exception:
        group_note = ""
    if access:
        print(f"PASS: este processo le e escreve em {node}.{group_note}")
    else:
        print(
            f"FAIL: este processo nao tem permissao de leitura e "
            f"escrita em {node}.{group_note}"
        )
        print(
            "  Correcao definitiva: sudo usermod -aG video $USER "
            "(vale nas proximas sessoes)."
        )
        print('  Correcao nesta sessao: sg video -c "python3 main.py".')
        all_ok = False

    index = int(node[len("/dev/video"):])
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        print(
            f"FAIL: nao consegui abrir {node}. Outro programa "
            "provavelmente esta usando a camera."
        )
        return 1
    _configure_local_capture(capture)
    frame = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            break
        time.sleep(0.1)
    capture.release()
    if frame is not None:
        print(
            f"PASS: quadro lido de {node}. "
            f"Formato (altura, largura, canais): {frame.shape}."
        )
    else:
        print(
            f"FAIL: {node} abriu, mas nenhum quadro chegou em 3 "
            "segundos. A camera pode estar ocupada ou travada."
        )
        all_ok = False

    return 0 if all_ok else 1


def _fit(image, max_h=700):
    h, w = image.shape[:2]
    if h <= max_h:
        return image
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h))


_BLANK_SHEET = None


def _blank_sheet():
    global _BLANK_SHEET
    if _BLANK_SHEET is None:
        _BLANK_SHEET = ink_module.render_canvas(
            np.zeros(
                (ink_module.config.PAPER_H, ink_module.config.PAPER_W),
                dtype=np.uint8,
            )
        )
    return _BLANK_SHEET


def _compose_view(camera_img, canvas_img):
    """Janela unica: camera a esquerda, folha ao vivo a direita.

    A folha ao vivo e o canvas digital (fundo branco, tinta preta) na
    mesma janela e na mesma altura da camera: o traco no papel e o
    traco digital aparecem lado a lado, sem janelas separadas.
    """
    target_h = 640
    cam = _fit(camera_img, target_h)
    sheet = _fit(canvas_img, target_h)
    if cam.shape[0] != sheet.shape[0]:
        h = min(cam.shape[0], sheet.shape[0])
        cam = cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h))
        sheet = cv2.resize(
            sheet, (int(sheet.shape[1] * h / sheet.shape[0]), h))
    cv2.putText(
        sheet,
        "folha ao vivo",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (140, 140, 140),
        1,
    )
    divider = np.full((cam.shape[0], 6, 3), (60, 60, 60), dtype=np.uint8)
    view = np.hstack([cam, divider, sheet])
    if view.shape[1] > 1600:
        scale = 1600 / view.shape[1]
        view = cv2.resize(view, (1600, int(view.shape[0] * scale)))
    return view


def _detect_sheet_combined(frame, neural_detector):
    """Detecção do modo sem marcadores: neural primário, clássico reserva.

    Sem detector neural, é o clássico puro, o comportamento de sempre.
    Com detector, ele decide primeiro; fechando o quadrilátero, vale o
    veredito dele. Não fechando, a reserva clássica só fala quando a
    neural não negou a folha com força: logit de presença até
    neural.PRESENCE_VETO_LOGIT devolve (None, None) direto, sem deixar
    o clássico declarar "paper" nem "blank_maybe" contra um "não há
    papel" forte do modelo. É o portão do incidente da caixa branca
    (2026-10-29): a v2 media logit ≈ −10,9 na caixa e o "Folha
    detectada" ao vivo vinha do clássico. A justificativa medida do
    limiar mora em draw.neural.PRESENCE_VETO_LOGIT. Logit negativo
    fraco (acima do limiar) não veta: a reserva segue valendo para a
    folha real que a neural subdispara num setup novo.

    Terceiro desfecho: se nenhum quadrilátero fechou (o ajuste neural
    foi recusado pela guarda de plausibilidade, ou não fechou, e o
    clássico também não achou candidato) mas o logit de presença é
    positivo, devolve (None, "paper"): presença sem posição. O
    rastreador lê isso como "a folha está lá, segura o quadrilátero
    anterior e renova o relógio de perda", em vez de contar o quadro
    como ausência (medido no replay do laboratório em 2026-10-30:
    sem esse canal, cada ajuste recusado vira contagem de perda e a
    folha cai durante os episódios de oclusão).
    """
    if neural_detector is not None:
        corners, verdict = neural_detector.detect_sheet_with_confidence(frame)
        if corners is not None:
            return corners, verdict
        logit = neural_detector.last_presence_logit
        if logit is not None and logit <= neural_module.PRESENCE_VETO_LOGIT:
            return None, None
        corners, verdict = paper_module.detect_sheet_with_confidence(frame)
        if corners is None and logit is not None and logit > 0:
            return None, "paper"
        return corners, verdict
    return paper_module.detect_sheet_with_confidence(frame)


def run(
    source=0,
    save_dir="captures",
    markers_mode=False,
    neural_model=None,
    hand_model=None,
    ink_model=None,
    session_baseline=False,
):
    neural_detector = None
    if neural_model is not None and not markers_mode:
        neural_detector = neural_module.NeuralSheetDetector(neural_model)
        neural_detector.load()
        print(
            f"Detector neural ativo: {neural_model}. "
            "O classico cobre os frames em que o neural nao ve folha."
        )

    ink_extractor = None
    if ink_model is not None and not markers_mode:
        if ink_model_module is None:
            print(
                "Modelo de tinta solicitado, mas o modulo "
                "draw/ink_model.py nao esta presente neste pacote. "
                "Seguindo com a heuristica."
            )
        else:
            ink_extractor = ink_model_module.InkModelExtractor(ink_model)
            if ink_extractor.load():
                print(
                    f"Modelo de tinta ativo: {ink_model}. "
                    "A heuristica segue como reserva por frame."
                )

    session_ink = None
    if session_baseline and not markers_mode:
        session_ink = ink_module.SessionBaselineInk()
        print(
            "Tinta da sessao ativa: com a folha na cena, aperte C "
            "para congelar a pagina base; so o traco feito depois "
            "dela entra no canvas."
        )

    def ink_source(warped_img):
        if ink_extractor is not None:
            mask = ink_extractor.ink_mask(warped_img)
            if mask is not None:
                return mask
        return ink_module.extract_ink(warped_img)

    hand_masker = None
    if hand_model is not None and not markers_mode:
        hand_masker = hand_module.HandMasker(hand_model or None)
        print(
            "Mascara de mao solicitada: mao sobre a folha vira regiao "
            "desconhecida no canvas acumulativo."
        )

    def detect_sheet(frame):
        return _detect_sheet_combined(frame, neural_detector)

    capture = _open_capture(parse_source(source))
    save_path = Path(save_dir)
    if markers_mode:
        print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
    else:
        print("Draw v0 rodando. Mostre uma folha para a camera.")
    print(
        "Teclas: s = salvar canvas, f = capturar frame, "
        "g = captura automatica (desligada), q/Esc = sair."
    )

    frames_path = save_path / "frames"
    frames_saved = 0
    auto_capture = False
    restart_reference = False
    baseline_armed = False
    # Geometria da tinta CONGELADA na tecla C: os warps da sessao
    # usam estes cantos, nao os do rastreador ao vivo. Medido nos
    # frames reais de 2026-10-30: sobre o caderno aberto, o ajuste
    # do detector oscila ate 314 px entre frames de uma cena
    # parada; com a geometria viva, o warp teleporta e nenhuma
    # referencia sobrevive. O rastreador segue vivo para o
    # contorno verde e os banners; a tinta vive na geometria do C.
    session_corners = None
    freeze_pending = False
    loop_frames = 0

    def save_raw_frame(img):
        nonlocal frames_saved
        frames_path.mkdir(parents=True, exist_ok=True)
        out = frames_path / f"frame_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(out), img)
        frames_saved += 1
        print(f"Frame salvo em {out} (total: {frames_saved}).")

    def handle_capture_key(key, img):
        nonlocal auto_capture, restart_reference
        if key == ord("f"):
            save_raw_frame(img)
        elif key == ord("c") and session_ink is not None:
            restart_reference = True
            print(
                "Pagina base pedida: capturando os proximos quadros "
                "retificados. Maos fora da folha."
            )
        elif key == ord("g"):
            auto_capture = not auto_capture
            if auto_capture:
                print(
                    "Captura automatica ligada: um frame a cada "
                    f"10 quadros vai para {frames_path}."
                )
            else:
                print("Captura automatica desligada.")

    tracker = _SheetTracker()
    ink_accumulator = ink_module.InkAccumulator()
    corner_history = deque(maxlen=_INK_CORNER_HISTORY)
    ink_session_open = False
    accumulated_canvas = None
    frame_number = 0
    falhas_seguidas = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            falhas_seguidas += 1
            if falhas_seguidas >= 15:
                print(
                    "A camera abriu, mas parou de entregar quadros. "
                    "Feche outros programas usando a camera e tente de novo."
                )
                break
            continue
        falhas_seguidas = 0
        # Instante de captura aproximado: a câmera não expõe carimbo
        # de hardware neste caminho, então vale o instante logo após
        # a leitura do frame, o mais próximo da captura que o laço
        # conhece. É ele que carimba os pixels no acumulador.
        frame_capture_ms = int(time.monotonic() * 1000)

        loop_frames += 1
        if auto_capture and loop_frames % 10 == 0:
            save_raw_frame(frame)

        if markers_mode:
            warped, markers = paper_module.warp_paper(frame)
        else:
            markers = {}
            if tracker.should_detect(frame_number):
                detected_corners, detected_verdict = detect_sheet(frame)
                ran_detection = True
            else:
                detected_corners, detected_verdict = None, None
                ran_detection = False
            tracker_event = tracker.update(
                detected_corners,
                detected_verdict,
                ran_detection,
                time.monotonic(),
            )
            if tracker_event == "drop":
                ink_session_open = False
                session_corners = None
                freeze_pending = False

            frame_number += 1
            warped = (
                paper_module.warp_sheet(frame, tracker.corners)
                if tracker.corners is not None
                else None
            )

            if ran_detection and tracker.corners is not None:
                if not ink_session_open:
                    # Folha nova na cena: canvas e geometria recomecam.
                    ink_accumulator.reset()
                    if session_ink is not None:
                        session_ink.reset()
                        baseline_armed = False
                        session_corners = None
                        freeze_pending = False
                    corner_history.clear()
                    ink_session_open = True
                if restart_reference:
                    # Tecla C: novo ponto zero da sessao. O canvas
                    # acumulado zera junto, porque a pagina base nova
                    # redefine o que conta como traco desta sessao.
                    # A geometria congela nos cantos deste momento.
                    restart_reference = False
                    ink_accumulator.reset()
                    session_ink.reset()
                    baseline_armed = True
                    freeze_pending = True
                corner_history.append(tracker.corners.copy())
                ink_corners = np.median(
                    np.stack(corner_history), axis=0
                ).astype(np.float32)
                if freeze_pending:
                    session_corners = ink_corners.copy()
                    freeze_pending = False
                warp_corners = (
                    session_corners
                    if session_ink is not None
                    and session_corners is not None
                    else ink_corners
                )
                ink_warped = paper_module.warp_sheet(frame, warp_corners)
                visible = None
                if hand_masker is not None:
                    camera_mask = hand_masker.mask(frame)
                    if camera_mask is not None:
                        hand_warped = paper_module.warp_sheet(
                            camera_mask, warp_corners
                        )
                        visible = hand_warped == 0
                if session_ink is not None:
                    if not session_ink.ready:
                        if baseline_armed:
                            session_ink.add_sample(ink_warped, visible)
                        observed = np.zeros(
                            ink_warped.shape[:2], dtype=np.uint8)
                    else:
                        observed = session_ink.observe(
                            ink_warped, visible)
                else:
                    observed = ink_source(ink_warped)
                accumulated_canvas = ink_module.render_canvas(
                    ink_accumulator.update(
                        observed, visible,
                        t_capture_ms=frame_capture_ms
                    )
                )

        status = frame.copy()
        if markers_mode:
            for (x, y) in markers.values():
                cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
        elif tracker.corners is not None:
            outline = tracker.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(status, [outline], True, (0, 255, 0), 2)
            for x, y in tracker.corners:
                cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
        if warped is None:
            if markers_mode:
                status_text = (
                    f"Mostre a folha-modelo ({len(markers)}/4 marcadores)"
                )
            else:
                status_text = "Mostre uma folha para a camera"
            cv2.putText(
                status,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            panel = (
                accumulated_canvas
                if accumulated_canvas is not None
                else _blank_sheet()
            )
            cv2.imshow("Draw", _compose_view(status, panel))
        else:
            if not markers_mode:
                if tracker.verdict == "blank_maybe":
                    status_text = "Folha detectada (baixa confianca)"
                    status_color = (0, 255, 255)
                else:
                    status_text = "Folha detectada"
                    status_color = (0, 255, 0)
                if session_ink is not None and not session_ink.ready:
                    if baseline_armed:
                        status_text = (
                            "Folha detectada: capturando a pagina "
                            "base, maos fora da folha"
                        )
                    else:
                        status_text = (
                            "Folha detectada: aperte C para comecar "
                            "(pagina base)"
                        )
                    status_color = (0, 255, 255)
                cv2.putText(
                    status,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                )
            if markers_mode:
                ink_mask = ink_module.extract_ink(
                    warped, clear_markers=True
                )
                canvas = ink_module.render_canvas(ink_mask)
            elif accumulated_canvas is not None:
                canvas = accumulated_canvas
            else:
                ink_mask = ink_source(warped)
                canvas = ink_module.render_canvas(ink_mask)
            cv2.imshow("Draw", _compose_view(status, canvas))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_path.mkdir(parents=True, exist_ok=True)
                out = save_path / f"canvas_{int(time.time())}.png"
                cv2.imwrite(str(out), canvas)
                print(f"Canvas salvo em {out}")
            handle_capture_key(key, frame)
            if key in (ord("q"), 27):
                break
            continue

        key = cv2.waitKey(1) & 0xFF
        handle_capture_key(key, frame)
        if key in (ord("q"), 27):
            break

    capture.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Draw v0, papel para o digital")
    parser.add_argument(
        "--source",
        default="0",
        help="indice da webcam local (0, 1...) ou URL do stream "
        "(ex.: http://192.168.0.10:8080/video). Padrao: 0",
    )
    parser.add_argument(
        "--markers",
        action="store_true",
        help="usa a folha-modelo impressa com marcadores ArUco",
    )
    parser.add_argument(
        "--neural",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="detector neural (ONNX) como primario, com o classico de "
        "reserva. Sem caminho, usa a variavel de ambiente "
        "DRAW_NEURAL_MODEL",
    )
    parser.add_argument(
        "--ink-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="modelo aprendido de tinta (ONNX) como fonte da mascara "
        "no modo sem marcadores, no lugar da heuristica. Sem caminho, "
        "usa a variavel de ambiente DRAW_INK_MODEL. Sem o arquivo ou "
        "com erro de carga, o app avisa uma vez e segue com a "
        "heuristica",
    )
    parser.add_argument(
        "--hand-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.task",
        help="mascara de mao (MediaPipe Hand Landmarker) para o "
        "canvas acumulativo: a mao vira regiao desconhecida e nao "
        "ganha nem perde tinta. Sem caminho, usa a variavel de "
        "ambiente DRAW_HAND_MODEL ou models/hand_landmarker.task. "
        "Sem o pacote mediapipe ou sem o modelo, o app roda sem a "
        "mascara e avisa uma vez",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="valida a camera local porta por porta e sai, sem abrir janela",
    )
    parser.add_argument(
        "--session-baseline",
        action="store_true",
        help="tinta da sessao: a pagina como esta quando voce aperta "
        "a tecla C vira fundo (pauta, margem impressa e desenho "
        "antigo somem do canvas); so o traco feito depois da pagina "
        "base entra. Apertar C de novo congela uma base nova. "
        "Incompativel com --markers",
    )
    parser.add_argument("--save-dir", default="captures")
    args = parser.parse_args()
    if args.check:
        parsed = parse_source(args.source)
        sys.exit(run_check(parsed if isinstance(parsed, int) else 0))
    neural_model = None
    if args.neural is not None:
        if args.markers:
            parser.error("--neural nao combina com --markers")
        neural_model = args.neural or os.environ.get("DRAW_NEURAL_MODEL")
        if not neural_model:
            parser.error(
                "--neural sem caminho: passe o arquivo .onnx ou defina "
                "a variavel DRAW_NEURAL_MODEL"
            )
    ink_model = None
    if args.ink_model is not None:
        if args.markers:
            parser.error("--ink-model nao combina com --markers")
        ink_model = args.ink_model or os.environ.get("DRAW_INK_MODEL")
        if not ink_model:
            parser.error(
                "--ink-model sem caminho: passe o arquivo .onnx ou "
                "defina a variavel DRAW_INK_MODEL"
            )
    if args.session_baseline and args.markers:
        parser.error("--session-baseline nao combina com --markers")
    hand_model = None
    if args.hand_model is not None:
        if args.markers:
            parser.error("--hand-model nao combina com --markers")
        hand_model = (
            args.hand_model or os.environ.get("DRAW_HAND_MODEL") or ""
        )
    try:
        run(
            source=args.source,
            save_dir=args.save_dir,
            markers_mode=args.markers,
            neural_model=neural_model,
            hand_model=hand_model,
            ink_model=ink_model,
            session_baseline=args.session_baseline,
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
