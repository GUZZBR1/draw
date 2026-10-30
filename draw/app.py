"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --check                          # valida a fonte e sai
  python main.py --markers                        # folha-modelo impressa
  python main.py --neural modelo.onnx             # detector neural primario
  python main.py --async-detect                   # detector em thread dedicada
  python main.py --hand-model modelo.task         # mascara de mao + cursor
  python main.py --session-baseline               # tinta da sessao (tecla C)
  python main.py --source 1                       # outra webcam local
  python main.py --source http://192.168.0.10:8080/video   # camera via rede
  python main.py --source tcp://192.168.0.10:8765          # helper Windows TCP
  python main.py --source sessao.drawcap                   # replay .drawcap

Teclas: s = salvar o canvas como PNG (+ carimbos .npz), f = capturar um frame cru,
g = ligar ou desligar a captura automatica de frames,
c = congelar a pagina base (so com --session-baseline), q/Esc = sair.
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from . import hand as hand_module
from . import ink as ink_module
try:
    from . import ink_model as ink_model_module
except ImportError:
    ink_model_module = None
from . import neural as neural_module
from .pipeline import (
    DrawPipeline,
    _INK_CORNER_HISTORY,
    _SHEET_BLEND_ALPHA,
    _SHEET_DETECT_INTERVAL,
    _SHEET_JUMP_PERSIST_SECONDS,
    _SHEET_LOST_MIN_FRAMES,
    _SHEET_LOST_SECONDS,
    _SHEET_NEAR_LIMIT_PX,
    _SheetTracker,
    _blend_corners,
    _corner_distance,
    _detect_sheet_combined,
)
from .protocol import HelperCapture, is_helper_source


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
                "--source tcp://<ip-do-windows>:8765 ou http://<ip>:8080/video."
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
    if is_helper_source(source):
        helper = HelperCapture(source)
        if helper.isOpened():
            return helper
        raise RuntimeError(
            f"Nao consegui abrir a fonte de protocolo Draw: {source!r}. "
            "Confira se o arquivo .drawcap existe ou se o helper Windows "
            "esta ouvindo na porta TCP informada."
        )
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
            "Confira se a URL/caminho esta certo, se a camera esta "
            "transmitindo e se este computador alcanca o endereco na rede."
        )
    raise RuntimeError(hint)


def _video_nodes():
    return sorted(
        path
        for path in glob.glob("/dev/video*")
        if path[len("/dev/video") :].isdigit()
    )


def run_check(source=0):
    """Validacao da fonte de video porta por porta. Retorna 0 ou 1."""
    print(f"PASS: OpenCV disponivel. Versao {cv2.__version__}.")
    parsed = parse_source(source)

    if not isinstance(parsed, int):
        capture = (
            HelperCapture(parsed, timeout=3.0)
            if is_helper_source(parsed)
            else cv2.VideoCapture(parsed)
        )
        if not capture.isOpened():
            print(
                f"FAIL: nao consegui abrir a fonte {parsed!r}. "
                "Confira o caminho ou URL e se a transmissao esta ativa."
            )
            return 1
        frame = None
        deadline = time.time() + 3.0
        while time.time() < deadline:
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
                break
            time.sleep(0.05)
        capture.release()
        if frame is None:
            print(
                f"FAIL: {parsed!r} abriu, mas nenhum quadro chegou em "
                "3 segundos."
            )
            return 1
        extra = ""
        if isinstance(capture, HelperCapture) and capture.last_capture_ts_ms is not None:
            extra = (
                f" Protocolo Draw: frame_id={capture.last_frame_id}, "
                f"capture_ts_ms={capture.last_capture_ts_ms}."
            )
        print(
            f"PASS: quadro lido de {parsed!r}. "
            f"Formato (altura, largura, canais): {frame.shape}.{extra}"
        )
        return 0

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
    node = f"/dev/video{parsed}"
    if node not in nodes:
        print(
            f"FAIL: {node} nao existe neste sistema "
            f"(disponiveis: {', '.join(nodes)})."
        )
        first_idx = nodes[0][len("/dev/video") :]
        print(
            f"  Correcao: use --source {first_idx} ou confira o indice "
            "da webcam desejada."
        )
        return 1

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

    capture = cv2.VideoCapture(parsed)
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
        return image.copy()
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h))


_BLANK_SHEET = None


def _blank_sheet():
    global _BLANK_SHEET
    if _BLANK_SHEET is None:
        _BLANK_SHEET = ink_module.render_canvas(
            np.zeros((config.PAPER_H, config.PAPER_W), dtype=np.uint8)
        )
    return _BLANK_SHEET


def _draw_cursor_overlay(sheet_img, cursor):
    """Desenha indicador de cursor apenas na cópia de visualização."""
    if cursor is None or not cursor.has_position or cursor.position_stale:
        return
    if cursor.x is None or cursor.y is None:
        return
    sh, sw = sheet_img.shape[:2]
    cx = int(round(cursor.x * sw / float(config.PAPER_W)))
    cy = int(round(cursor.y * sh / float(config.PAPER_H)))
    if not (0 <= cx < sw and 0 <= cy < sh):
        return
    if cursor.pen_down:
        cv2.circle(sheet_img, (cx, cy), 5, (20, 160, 40), -1)
    else:
        cv2.circle(sheet_img, (cx, cy), 6, (200, 120, 20), 2)


def _compose_view(camera_img, canvas_img, cursor=None):
    """Janela unica: camera a esquerda, folha ao vivo a direita."""
    target_h = 640
    cam = _fit(camera_img, target_h)
    sheet = _fit(canvas_img, target_h)
    if cam.shape[0] != sheet.shape[0]:
        h = min(cam.shape[0], sheet.shape[0])
        cam = cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h))
        sheet = cv2.resize(
            sheet, (int(sheet.shape[1] * h / sheet.shape[0]), h)
        )
    _draw_cursor_overlay(sheet, cursor)
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


def run(
    source=0,
    save_dir="captures",
    markers_mode=False,
    neural_model=None,
    hand_model=None,
    ink_model=None,
    session_baseline=False,
    async_detect=False,
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

    if session_baseline and not markers_mode:
        print(
            "Tinta da sessao ativa: com a folha na cena, aperte C "
            "para congelar a pagina base; so o traco feito depois "
            "dela entra no canvas."
        )

    hand_masker = None
    if hand_model is not None and not markers_mode:
        hand_masker = hand_module.HandMasker(hand_model or None)
        print(
            "Mascara de mao solicitada: mao sobre a folha vira regiao "
            "desconhecida no canvas acumulativo e alimenta fusao do cursor."
        )

    parsed_source = parse_source(source)
    capture = _open_capture(parsed_source)
    is_replay_file = isinstance(parsed_source, str) and (
        parsed_source.lower().endswith(".drawcap")
        or Path(parsed_source).is_file()
    )

    pipeline = DrawPipeline(
        markers_mode=markers_mode,
        neural_detector=neural_detector,
        hand_masker=hand_masker,
        ink_extractor=ink_extractor,
        session_baseline=session_baseline,
        async_detect=async_detect,
    )

    save_path = Path(save_dir)
    if markers_mode:
        print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
    else:
        print("Draw v0 rodando. Mostre uma folha para a camera.")
    print(
        "Teclas: s = salvar canvas, f = capturar frame, "
        "g = captura automatica (desligada), c = pagina base, q/Esc = sair."
    )

    frames_path = save_path / "frames"
    frames_saved = 0
    auto_capture = False
    loop_frames = 0
    falhas_seguidas = 0

    def save_raw_frame(img):
        nonlocal frames_saved
        frames_path.mkdir(parents=True, exist_ok=True)
        out = frames_path / f"frame_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(out), img)
        frames_saved += 1
        print(f"Frame salvo em {out} (total: {frames_saved}).")

    def handle_key(key, img, current_canvas):
        nonlocal auto_capture
        if key == ord("s"):
            png_out, npz_out = pipeline.save_artifacts(
                save_path, canvas_override=current_canvas
            )
            if png_out is not None:
                extra = f" (carimbos: {npz_out})" if npz_out is not None else ""
                print(f"Canvas salvo em {png_out}{extra}")
        elif key == ord("f"):
            save_raw_frame(img)
        elif key == ord("c") and pipeline.trigger_baseline():
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

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if is_replay_file:
                    break
                falhas_seguidas += 1
                if falhas_seguidas >= 15:
                    print(
                        "A camera abriu, mas parou de entregar quadros. "
                        "Feche outros programas usando a camera e tente de novo."
                    )
                    break
                continue
            falhas_seguidas = 0

            now_s = time.monotonic()
            if (
                isinstance(capture, HelperCapture)
                and capture.last_capture_ts_ms is not None
            ):
                frame_capture_ms = capture.last_capture_ts_ms
            else:
                frame_capture_ms = int(now_s * 1000)

            loop_frames += 1
            if auto_capture and loop_frames % 10 == 0:
                save_raw_frame(frame)

            res = pipeline.process_frame(
                frame, t_capture_ms=frame_capture_ms, now_s=now_s
            )

            status = frame.copy()
            if markers_mode:
                for x, y in res.markers.values():
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
            elif res.corners is not None:
                outline = res.corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(status, [outline], True, (0, 255, 0), 2)
                for x, y in res.corners:
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)

            if not res.sheet_present:
                status_text = (
                    f"Mostre a folha-modelo ({len(res.markers)}/4 marcadores)"
                    if markers_mode
                    else "Mostre uma folha para a camera"
                )
                cv2.putText(
                    status,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                panel = res.canvas if res.canvas is not None else _blank_sheet()
                cv2.imshow("Draw", _compose_view(status, panel, res.cursor))
            else:
                if not markers_mode:
                    if res.verdict == "blank_maybe":
                        status_text = "Folha detectada (baixa confianca)"
                        status_color = (0, 255, 255)
                    else:
                        status_text = "Folha detectada"
                        status_color = (0, 255, 0)
                    if (
                        pipeline.session_ink is not None
                        and not pipeline.session_ink.ready
                    ):
                        if pipeline.baseline_armed:
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
                panel = res.canvas if res.canvas is not None else _blank_sheet()
                cv2.imshow("Draw", _compose_view(status, panel, res.cursor))

            key = cv2.waitKey(1) & 0xFF
            handle_key(key, frame, res.canvas)
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.close()
        capture.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Draw v0, papel para o digital"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="indice da webcam local (0, 1...), URL MJPEG/RTSP, "
        "tcp://host[:8765] do helper Windows ou arquivo .drawcap. "
        "Padrao: 0",
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
        "--async-detect",
        action="store_true",
        help="executa a deteccao da folha numa thread dedicada em "
        "segundo plano, desacoplando o custo do ONNX da cadencia de "
        "extracao de tinta e fusao do cursor",
    )
    parser.add_argument(
        "--ink-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="modelo aprendido de tinta (ONNX) opcional; sem modulo "
        "presente, o app avisa uma vez e segue com a heuristica",
    )
    parser.add_argument(
        "--hand-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.task",
        help="mascara de mao (MediaPipe Hand Landmarker) para o "
        "canvas acumulativo e fusao do cursor. Sem caminho, usa "
        "DRAW_HAND_MODEL ou models/hand_landmarker.task",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="valida a fonte de captura e sai, sem abrir janela",
    )
    parser.add_argument(
        "--session-baseline",
        action="store_true",
        help="tinta da sessao: a pagina como esta quando voce aperta "
        "a tecla C vira fundo; so o traco feito depois entra. "
        "Incompativel com --markers",
    )
    parser.add_argument("--save-dir", default="captures")
    args = parser.parse_args()
    if args.check:
        sys.exit(run_check(parse_source(args.source)))
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
    if args.async_detect and args.markers:
        parser.error("--async-detect nao combina com --markers")
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
            async_detect=args.async_detect,
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
