"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --check                          # valida a fonte e sai
  python main.py --list-cameras                   # lista webcams que abrem
  python main.py --markers                        # folha-modelo impressa
  python main.py --neural modelo.onnx             # detector neural primario
  python main.py --ink-model modelo.onnx          # modelo aprendido de tinta
  python main.py --hand-model modelo.task         # mascara de mao no canvas
  python main.py --source 1                       # outra webcam local
  python main.py --source /dev/video2             # porta V4L2 especifica
  python main.py --source http://192.168.0.10:8080/video   # camera via rede
  python main.py --light-normalize both   # sombra e luz baixa

A fonte via rede (MJPEG/RTSP) e o caminho recomendado no WSL. O WSL nao
enxerga webcams USB do Windows por padrao, mas enxerga streams de rede.
Assim todo o processamento (e o ML futuro) fica no Linux. A camera so
envia os frames. Ela pode ser um celular, outro computador ou uma camera IP.

Teclas: s = salvar o canvas como PNG, v = salvar tracos vetoriais
(SVG + JSON), f = capturar um frame cru,
g = ligar ou desligar a captura automatica de frames, q/Esc = sair.
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import capture as capture_module
from . import diagnose as diagnose_module
from . import hand as hand_module
from . import ink as ink_module
from . import light as light_module
from . import light_guard as light_guard_module
try:
    from . import ink_model as ink_model_module
except ImportError:
    # O extrator de tinta por modelo aprendido e opcional e o modulo
    # dele ainda nao faz parte do pacote publicado. Sem o modulo,
    # --ink-model avisa e o app segue so com a heuristica.
    ink_model_module = None
from . import neural as neural_module
from . import paper as paper_module
from . import sequence as sequence_module
from . import strokes as strokes_module


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
    """Delega para draw/capture.py, mantendo o mesmo comportamento."""
    return capture_module.parse_source(value)


def is_wsl():
    """Detecta WSL pelo nome do kernel em /proc."""
    return capture_module.is_wsl()


def _configure_local_capture(capture, width=None, height=None, fps=None):
    """MJPG 720p por padrao; ver draw/capture.apply_local_settings."""
    capture_module.apply_local_settings(
        capture, width=width, height=height, fps=fps
    )


def _local_open_hint(source):
    """Diagnostico por porta, o mesmo criterio do --check."""
    return capture_module.diagnose_local(source)[1]


def _detection_frame(frame, light_normalize="off"):
    """Frame que vai para a detecção da folha.

    Com a flag desligada devolve o frame sem tocar nele. Com ela ligada,
    devolve o frame normalizado (ver draw/light.py). A detecção por
    marcadores não usa este caminho: ela continua no frame original,
    para não mudar o comportamento do modo com marcadores.
    """
    return light_module.normalize_light(frame, light_normalize)


def _ink_input(warped, light_normalize="off"):
    """Folha retificada que vai para a extração de tinta.

    Passo separado da detecção: a folha retificada é a do frame
    original, normalizada uma única vez aqui. Com a flag desligada
    devolve a folha sem tocar nela.
    """
    return light_module.normalize_light(warped, light_normalize)


def detect_sheet_with_light_guard(frame, light_normalize="off"):
    """Delega para draw/light_guard.py (off primeiro, background depois)."""
    return light_guard_module.detect_sheet_with_light_guard(
        frame, light_normalize
    )


def _open_capture(source, **kwargs):
    """Abre via draw/capture.py, com backends, timeout e erro especifico."""
    return capture_module.open_capture(source, **kwargs)


def _video_nodes():
    return capture_module.video_nodes()


def _read_first_frame(capture, timeout_s):
    """Tenta obter um frame até o prazo; capture.read() pode bloquear."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            ok, frame = capture.read()
        except Exception as error:
            return None, error
        if time.monotonic() > deadline:
            return None, TimeoutError("nenhum frame recebido dentro do prazo")
        if ok and frame is not None:
            return frame, None

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, TimeoutError("nenhum frame recebido dentro do prazo")
        time.sleep(min(0.05, remaining))


def run_check(source=0, open_timeout_ms=3000, first_frame_timeout_s=5.0):
    """Valida a fonte informada, lê um frame e sempre libera a captura."""
    parsed = parse_source(source)
    print(f"PASS: OpenCV disponivel. Versao {cv2.__version__}.")
    print(f"Testando fonte: {parsed!r}.")

    capture = None
    try:
        read_timeout_ms = max(1, int(first_frame_timeout_s * 1000))
        capture = _open_capture(
            parsed,
            open_timeout_ms=open_timeout_ms,
            read_timeout_ms=read_timeout_ms,
        )
        if not capture.isOpened():
            print(f"FAIL: a fonte {parsed!r} nao abriu.")
            return 1

        print(f"PASS: fonte {parsed!r} aberta.")
        frame, read_error = _read_first_frame(capture, first_frame_timeout_s)
        if read_error is not None:
            if isinstance(read_error, TimeoutError):
                print(
                    f"FAIL: a fonte {parsed!r} abriu, mas nao entregou "
                    f"um frame antes do prazo de {first_frame_timeout_s:g} "
                    "segundos. O backend pode ultrapassar o prazo se "
                    "capture.read() bloquear."
                )
            else:
                print(f"FAIL: erro ao ler frame da fonte {parsed!r}: {read_error}")
            return 1

        height, width = frame.shape[:2]
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        except Exception:
            fps = 0.0
        fps_text = f"{fps:g} FPS" if fps > 0 else "FPS indisponivel"
        print(f"PASS: primeiro frame recebido: {width}x{height}, {fps_text}.")
        return 0
    except capture_module.CaptureError as error:
        print(f"FAIL: {error}")
        return 1
    except Exception as error:
        print(f"FAIL: erro ao testar a fonte {parsed!r}: {error}")
        return 1
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass


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


def _detect_sheet_combined(frame, neural_detector, classic_detect=None):
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
    if classic_detect is None:
        classic_detect = paper_module.detect_sheet_with_confidence
    if neural_detector is not None:
        corners, verdict = neural_detector.detect_sheet_with_confidence(frame)
        if corners is not None:
            return corners, verdict
        logit = neural_detector.last_presence_logit
        if logit is not None and logit <= neural_module.PRESENCE_VETO_LOGIT:
            return None, None
        corners, verdict = classic_detect(frame)
        if corners is None and logit is not None and logit > 0:
            return None, "paper"
        return corners, verdict
    return classic_detect(frame)


def run(source=0, save_dir="captures", markers_mode=False, width=None,
        height=None, fps=None, backend=None, reconnect_attempts=3,
        open_timeout_ms=3000, neural_model=None, capture_all_frames=False,
        light_normalize="off", session_mode=False, hand_model=None,
        ink_model=None):
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

    winning_light = {"mode": light_normalize}

    def detect_sheet(frame):
        # Reserva classica do fork: a guarda de luz, em vez do
        # classico puro do upstream. O portao de veto neural e a
        # presenca sem quadrilatero vem de _detect_sheet_combined.
        def classic_with_light_guard(frame):
            corners, verdict, winning = detect_sheet_with_light_guard(
                frame, light_normalize
            )
            winning_light["mode"] = winning
            return corners, verdict

        corners, verdict = _detect_sheet_combined(
            frame, neural_detector, classic_with_light_guard
        )
        if corners is not None:
            winning_light["mode"] = light_normalize
        return corners, verdict
    parsed = parse_source(source)
    capture = capture_module.ReconnectingCapture(
        parsed, attempts=reconnect_attempts, width=width, height=height,
        fps=fps, backend=backend, open_timeout_ms=open_timeout_ms,
    ).attach(_open_capture(
        parsed, width=width, height=height, fps=fps, backend=backend,
        open_timeout_ms=open_timeout_ms,
    ))
    save_path = Path(save_dir)
    dataset_path = None
    dataset_writer = None
    dataset_manifest = None
    if capture_all_frames:
        session_name = (
            f"session_{time.strftime('%Y%m%d_%H%M%S')}_"
            f"{time.time_ns() % 1_000_000:06d}"
        )
        dataset_path = save_path / "training" / session_name
        dataset_path.mkdir(parents=True, exist_ok=False)
        dataset_manifest = (dataset_path / "frames.csv").open(
            "w", newline="", encoding="utf-8", buffering=1
        )
        dataset_writer = csv.writer(dataset_manifest)
        dataset_writer.writerow(
            ["frame_index", "timestamp_ns", "filename", "width", "height"]
        )
        print(f"Salvando todos os frames em {dataset_path}.")
    if markers_mode:
        print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
    else:
        print("Draw v0 rodando. Mostre uma folha para a camera.")
    print(
        "Teclas: s = salvar canvas, v = salvar tracos vetoriais, "
        "f = capturar frame, "
        "g = captura automatica (desligada), q/Esc = sair."
    )
    # Modo sessão (etapa 5 do Senado): rastreia a ordem temporal
    # dos traços enquanto o desenho acontece. Desligado por padrão;
    # a tecla v continua exportando o snapshot por centroide, e a
    # tecla o exporta a sessão ordenada, só neste modo.
    session_tracker = (
        sequence_module.SequenceTracker() if session_mode else None
    )
    session_counter = 0
    if session_mode:
        print(
            "Modo sessao ligado: a ordem dos tracos e registrada "
            "enquanto eles aparecem (tecla o = salvar sessao ordenada)."
        )

    frames_path = save_path / "frames"
    frames_saved = 0
    auto_capture = False
    loop_frames = 0

    def save_raw_frame(img):
        nonlocal frames_saved
        frames_path.mkdir(parents=True, exist_ok=True)
        out = frames_path / f"frame_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(out), img)
        frames_saved += 1
        print(f"Frame salvo em {out} (total: {frames_saved}).")

    def handle_capture_key(key, img):
        nonlocal auto_capture
        if key == ord("f"):
            save_raw_frame(img)
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
                if capture.try_reconnect():
                    print("Camera reconectada, continuando.")
                    falhas_seguidas = 0
                    continue
                print(
                    "A camera abriu, mas parou de entregar quadros e "
                    "nao voltou apos a reconexao. Feche outros programas "
                    "usando a camera, reconecte-a e tente de novo."
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
        if dataset_writer is not None:
            frame_name = f"frame_{loop_frames:08d}.jpg"
            frame_path = dataset_path / frame_name
            if not cv2.imwrite(
                str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                print(f"Erro ao salvar frame: {frame_path}")
                break
            height_px, width_px = frame.shape[:2]
            dataset_writer.writerow(
                [loop_frames, time.time_ns(), frame_name, width_px, height_px]
            )
            dataset_manifest.flush()
            if loop_frames % 100 == 0:
                print(f"Frames salvos nesta sessao: {loop_frames}.")
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
                    corner_history.clear()
                    ink_session_open = True
                corner_history.append(tracker.corners.copy())
                ink_corners = np.median(
                    np.stack(corner_history), axis=0
                ).astype(np.float32)
                ink_warped = paper_module.warp_sheet(frame, ink_corners)
                observed = ink_source(ink_warped)
                visible = None
                if hand_masker is not None:
                    camera_mask = hand_masker.mask(frame)
                    if camera_mask is not None:
                        hand_warped = paper_module.warp_sheet(
                            camera_mask, ink_corners
                        )
                        visible = hand_warped == 0
                accumulated_canvas = ink_module.render_canvas(
                    ink_accumulator.update(
                        observed, visible, t_capture_ms=frame_capture_ms
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
                cv2.putText(
                    status,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                )
            ink_mask = ink_module.extract_ink(
                _ink_input(warped, winning_light["mode"]),
                clear_markers=markers_mode,
            )
            if session_tracker is not None:
                # Amostragem reduzida: uma amostra a cada 5 quadros
                # retificados, não a cada quadro — o thinning e o
                # rastreamento a cada quadro são custo sem ganho.
                session_counter += 1
                if session_counter % 5 == 1:
                    session_tracker.update(
                        strokes_module.extract_strokes(ink_mask),
                        frame_index=loop_frames,
                        timestamp_ms=int(time.time() * 1000),
                        corners=(
                            None if markers_mode else tracker.corners
                        ),
                        sheet_locked=True,
                    )
            if markers_mode:
                canvas = ink_module.render_canvas(ink_mask)
            elif accumulated_canvas is not None:
                canvas = accumulated_canvas
            else:
                # ink_mask ja veio de extract_ink sobre a entrada
                # com luz normalizada; o modelo de tinta do
                # upstream so o substitui quando esta ativo.
                if ink_extractor is not None:
                    ink_mask = ink_source(warped)
                canvas = ink_module.render_canvas(ink_mask)
            cv2.imshow("Draw", _compose_view(status, canvas))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_path.mkdir(parents=True, exist_ok=True)
                out = save_path / f"canvas_{int(time.time())}.png"
                cv2.imwrite(str(out), canvas)
                print(f"Canvas salvo em {out}")
            if key == ord("v"):
                save_path.mkdir(parents=True, exist_ok=True)
                base = save_path / f"strokes_{int(time.time())}"
                strokes = strokes_module.extract_strokes(ink_mask)
                svg_path, json_path = strokes_module.export_strokes(
                    strokes, base
                )
                print(f"{len(strokes)} tracos salvos em {svg_path} "
                      f"e {json_path}")
            if key == ord("o") and session_tracker is not None:
                save_path.mkdir(parents=True, exist_ok=True)
                base = save_path / f"session_{int(time.time())}"
                svg_path, json_path = sequence_module.export_session(
                    session_tracker, base
                )
                print(f"sessao com {len(session_tracker.ordered)} tracos "
                      f"ordenados salva em {svg_path} e {json_path}")
            handle_capture_key(key, frame)
            if key in (ord("q"), 27):
                break
            continue

        key = cv2.waitKey(1) & 0xFF
        handle_capture_key(key, frame)
        if key in (ord("q"), 27):
            break

    capture.release()
    if dataset_manifest is not None:
        dataset_manifest.close()
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
        help="valida a fonte escolhida com --source e sai, sem abrir janela",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="lista as webcams locais que abrem e sai, sem abrir janela",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="com --list-cameras, acrescenta backend e props negociadas",
    )
    parser.add_argument(
        "--test-source",
        default=None,
        metavar="FONTE",
        help="sonda uma fonte (indice, /dev/videoN, arquivo ou URL) "
        "e sai, sem abrir janela e sem acao no sistema",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="sonda as webcams locais que abrem e sai, sem abrir janela",
    )
    parser.add_argument(
        "--include-urls",
        nargs="*",
        default=[],
        help="com --test-all, sonda tambem estas URLs explicitamente",
    )
    parser.add_argument(
        "--report",
        choices=["text", "json"],
        default="text",
        help="formato do relatorio de --test-source/--test-all",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="largura pedida para webcam local (padrao: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="altura pedida para webcam local (padrao: 720)",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="FPS pedido para webcam local (padrao: o da camera)",
    )
    parser.add_argument(
        "--backend", default=None,
        help="backend OpenCV: any, v4l2, msmf, dshow, ffmpeg, "
        "avfoundation, gstreamer (padrao: o do SO, para webcam local)",
    )
    parser.add_argument(
        "--reconnect-attempts", type=int, default=3,
        help="tentativas de reabrir a camera quando os quadros param",
    )
    parser.add_argument(
        "--open-timeout-ms", type=int, default=3000,
        help="timeout de abertura da fonte, em milissegundos",
    )
    parser.add_argument(
        "--light-normalize", default="off",
        choices=list(light_module.LIGHT_MODES),
        help="normalizacao de luz: off (padrao, sem mudanca), "
        "background (fundo + divisao), clahe (contraste local) ou "
        "both (fundo + CLAHE, o melhor medido em sombra e luz baixa)",
    )
    parser.add_argument(
        "--session",
        action="store_true",
        help="modo sessao: registra a ordem temporal dos tracos "
        "enquanto eles aparecem (tecla o = salvar sessao ordenada); "
        "sem ele, v continua exportando o snapshot por centroide",
    )
    parser.add_argument("--save-dir", default="captures")
    parser.add_argument(
        "--capture-all-frames",
        action="store_true",
        help="salva cada frame JPEG e seu timestamp para coleta de dados",
    )
    args = parser.parse_args()
    if args.test_source is not None or args.test_all or args.include_urls:
        reports = []
        if args.test_source is not None:
            reports.append(diagnose_module.probe_source(args.test_source))
        if args.test_all:
            reports.extend(diagnose_module.probe_all(
                include_urls=args.include_urls
            ))
        elif args.include_urls:
            reports.extend(diagnose_module.probe_all(
                include_urls=args.include_urls,
                list_fn=lambda: [],
            ))
        if args.report == "json":
            import json

            print(json.dumps(
                [report.to_dict() for report in reports],
                indent=2, default=str,
            ))
        else:
            for report in reports:
                print(diagnose_module.format_report(report))
        if not reports:
            sys.exit(2)
        sys.exit(0 if all(report.ok for report in reports) else 1)
    if args.list_cameras:
        results = capture_module.list_cameras()
        if args.verbose:
            probed = {}
            for item in results:
                if item["opened"]:
                    probed[item["index"]] = diagnose_module.probe_source(
                        item["index"], read_frames=1
                    )
            lines = []
            for item in results:
                status = "ok" if item["opened"] else "nao abriu"
                node = f" ({item['node']})" if item.get("node") else ""
                detail = ""
                report = probed.get(item["index"])
                if report is not None:
                    props = []
                    if report.width and report.height:
                        props.append(f"{report.width}x{report.height}")
                    if report.fps:
                        props.append(f"@{report.fps}fps")
                    if report.fourcc:
                        props.append(str(report.fourcc))
                    if report.backend_used:
                        props.append(str(report.backend_used))
                    if props:
                        detail = f" [{', '.join(props)}]"
                lines.append(f"  {item['index']}{node}: {status}{detail}")
            print("\n".join(lines))
        else:
            print(capture_module.format_camera_list(results))
        sys.exit(0 if any(item["opened"] for item in results) else 1)
    if args.check:
        parsed = parse_source(args.source)
        sys.exit(run_check(parsed, open_timeout_ms=args.open_timeout_ms))
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
    hand_model = None
    if args.hand_model is not None:
        if args.markers:
            parser.error("--hand-model nao combina com --markers")
        hand_model = (
            args.hand_model or os.environ.get("DRAW_HAND_MODEL") or ""
        )
    try:
        backend = capture_module.backend_from_name(args.backend)
    except ValueError as error:
        parser.error(str(error))
    try:
        run(
            source=args.source,
            save_dir=args.save_dir,
            markers_mode=args.markers,
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend=backend,
            reconnect_attempts=args.reconnect_attempts,
            open_timeout_ms=args.open_timeout_ms,
            neural_model=neural_model,
            capture_all_frames=args.capture_all_frames,
            light_normalize=args.light_normalize,
            session_mode=args.session,
            hand_model=hand_model,
            ink_model=ink_model,
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
