"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --check                          # valida a camera e sai
  python main.py --list-cameras                   # lista webcams que abrem
  python main.py --markers                        # folha-modelo impressa
  python main.py --source 1                       # outra webcam local
  python main.py --source /dev/video2             # porta V4L2 especifica
  python main.py --source http://192.168.0.10:8080/video   # camera via rede

A fonte via rede (MJPEG/RTSP) e o caminho recomendado no WSL. O WSL nao
enxerga webcams USB do Windows por padrao, mas enxerga streams de rede.
Assim todo o processamento (e o ML futuro) fica no Linux. A camera so
envia os frames. Ela pode ser um celular, outro computador ou uma camera IP.

Teclas: s = salvar o canvas como PNG, v = salvar tracos vetoriais
(SVG + JSON), f = capturar um frame cru,
g = ligar ou desligar a captura automatica de frames, q/Esc = sair.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import capture as capture_module
from . import diagnose as diagnose_module
from . import ink as ink_module
from . import paper as paper_module
from . import strokes as strokes_module


_SHEET_NEAR_LIMIT_PX = 40.0
_SHEET_BLEND_ALPHA = 0.3
_SHEET_JUMP_CONFIRMATIONS = 3
_SHEET_LOST_FRAME_LIMIT = 30
_SHEET_DETECT_INTERVAL = 5


def _corner_distance(current, detected):
    return float(np.max(np.linalg.norm(current - detected, axis=1)))


def _blend_corners(current, detected):
    return (
        (1.0 - _SHEET_BLEND_ALPHA) * current
        + _SHEET_BLEND_ALPHA * detected
    ).astype(np.float32)


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


def _open_capture(source, **kwargs):
    """Abre via draw/capture.py, com backends, timeout e erro especifico."""
    return capture_module.open_capture(source, **kwargs)


def _video_nodes():
    return capture_module.video_nodes()


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


def run(source=0, save_dir="captures", markers_mode=False, width=None,
        height=None, fps=None, backend=None, reconnect_attempts=3,
        open_timeout_ms=3000):
    parsed = parse_source(source)
    capture = capture_module.ReconnectingCapture(
        parsed, attempts=reconnect_attempts, width=width, height=height,
        fps=fps, backend=backend, open_timeout_ms=open_timeout_ms,
    ).attach(_open_capture(
        parsed, width=width, height=height, fps=fps, backend=backend,
        open_timeout_ms=open_timeout_ms,
    ))
    save_path = Path(save_dir)
    if markers_mode:
        print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
    else:
        print("Draw v0 rodando. Mostre uma folha para a camera.")
    print(
        "Teclas: s = salvar canvas, v = salvar tracos vetoriais, "
        "f = capturar frame, "
        "g = captura automatica (desligada), q/Esc = sair."
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

    current_corners = None
    current_verdict = None
    pending_corners = None
    pending_hits = 0
    frames_since_seen = 0
    sheet_missed = False
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

        loop_frames += 1
        if auto_capture and loop_frames % 10 == 0:
            save_raw_frame(frame)

        if markers_mode:
            warped, markers = paper_module.warp_paper(frame)
        else:
            markers = {}
            should_detect = (
                current_corners is None
                or sheet_missed
                or frame_number % _SHEET_DETECT_INTERVAL == 0
            )
            if should_detect:
                detected_corners, detected_verdict = (
                    paper_module.detect_sheet_with_confidence(frame)
                )
            else:
                detected_corners, detected_verdict = None, None
            if detected_corners is not None:
                if (
                    current_corners is None
                    and detected_verdict == "paper"
                ):
                    current_corners = detected_corners.copy()
                    current_verdict = detected_verdict
                    pending_corners = None
                    pending_hits = 0
                    frames_since_seen = 0
                    sheet_missed = False
                elif (
                    current_corners is not None
                    and _corner_distance(current_corners, detected_corners)
                    <= _SHEET_NEAR_LIMIT_PX
                ):
                    current_corners = _blend_corners(
                        current_corners, detected_corners
                    )
                    current_verdict = detected_verdict
                    pending_corners = None
                    pending_hits = 0
                    frames_since_seen = 0
                    sheet_missed = False
                else:
                    # Salto de posicao ou candidato ainda sem trava.
                    # Candidato blank_maybe (branco e liso, sem textura
                    # de papel) exige o dobro de confirmacoes seguidas.
                    confirmations_needed = (
                        2 * _SHEET_JUMP_CONFIRMATIONS
                        if detected_verdict == "blank_maybe"
                        else _SHEET_JUMP_CONFIRMATIONS
                    )
                    if (
                        pending_corners is not None
                        and _corner_distance(
                            pending_corners, detected_corners
                        )
                        <= _SHEET_NEAR_LIMIT_PX
                    ):
                        pending_corners = _blend_corners(
                            pending_corners, detected_corners
                        )
                        pending_hits += 1
                    else:
                        pending_corners = detected_corners.copy()
                        pending_hits = 1

                    if pending_hits >= confirmations_needed:
                        current_corners = pending_corners.copy()
                        current_verdict = detected_verdict
                        pending_corners = None
                        pending_hits = 0
                        frames_since_seen = 0
                        sheet_missed = False
                    else:
                        frames_since_seen += 1
                        sheet_missed = True
            elif current_corners is not None:
                if should_detect:
                    pending_corners = None
                    pending_hits = 0
                    sheet_missed = True
                frames_since_seen += 1

            if (
                current_corners is not None
                and frames_since_seen > _SHEET_LOST_FRAME_LIMIT
            ):
                current_corners = None
                current_verdict = None
                pending_corners = None
                pending_hits = 0
                frames_since_seen = 0
                sheet_missed = False

            frame_number += 1
            warped = (
                paper_module.warp_sheet(frame, current_corners)
                if current_corners is not None
                else None
            )

        status = frame.copy()
        if markers_mode:
            for (x, y) in markers.values():
                cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
        elif current_corners is not None:
            outline = current_corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(status, [outline], True, (0, 255, 0), 2)
            for x, y in current_corners:
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
            cv2.imshow("Draw - camera", _fit(status, 700))
        else:
            if not markers_mode:
                if current_verdict == "blank_maybe":
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
            ink_mask = ink_module.extract_ink(warped)
            canvas = ink_module.render_canvas(ink_mask)
            cv2.imshow("Draw - camera", _fit(status, 700))
            cv2.imshow("Draw - papel (vista de cima)", _fit(warped, 700))
            cv2.imshow("Draw - canvas digital", _fit(canvas, 700))

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
        "--check",
        action="store_true",
        help="valida a camera local porta por porta e sai, sem abrir janela",
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
    parser.add_argument("--save-dir", default="captures")
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
        sys.exit(run_check(parsed if isinstance(parsed, int) else 0))
    try:
        backend = capture_module.backend_from_name(args.backend)
    except ValueError as error:
        parser.error(str(error))
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
    )


if __name__ == "__main__":
    main()
