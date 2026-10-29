"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --check                          # valida a fonte e sai
  python main.py --list-cameras                   # lista webcams que abrem
  python main.py --markers                        # folha-modelo impressa
  python main.py --neural modelo.onnx             # detector neural primario
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
from pathlib import Path

import cv2
import numpy as np

from . import capture as capture_module
from . import diagnose as diagnose_module
from . import ink as ink_module
from . import light as light_module
from . import neural as neural_module
from . import paper as paper_module
from . import sequence as sequence_module
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


def run(source=0, save_dir="captures", markers_mode=False, width=None,
        height=None, fps=None, backend=None, reconnect_attempts=3,
        open_timeout_ms=3000, neural_model=None, capture_all_frames=False,
        light_normalize="off", session_mode=False):
    neural_detector = None
    if neural_model is not None and not markers_mode:
        neural_detector = neural_module.NeuralSheetDetector(neural_model)
        neural_detector.load()
        print(
            f"Detector neural ativo: {neural_model}. "
            "O classico cobre os frames em que o neural nao ve folha."
        )

    def detect_sheet(frame):
        if neural_detector is not None:
            corners, verdict = (
                neural_detector.detect_sheet_with_confidence(frame)
            )
            if corners is not None:
                return corners, verdict
        return paper_module.detect_sheet_with_confidence(
            _detection_frame(frame, light_normalize)
        )
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
    tracker = (
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
            should_detect = (
                current_corners is None
                or sheet_missed
                or frame_number % _SHEET_DETECT_INTERVAL == 0
            )
            if should_detect:
                detected_corners, detected_verdict = detect_sheet(frame)
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
            ink_mask = ink_module.extract_ink(
                _ink_input(warped, light_normalize),
                clear_markers=markers_mode,
            )
            if tracker is not None:
                # Amostragem reduzida: uma amostra a cada 5 quadros
                # retificados, não a cada quadro — o thinning e o
                # rastreamento a cada quadro são custo sem ganho.
                session_counter += 1
                if session_counter % 5 == 1:
                    tracker.update(
                        strokes_module.extract_strokes(ink_mask),
                        frame_index=loop_frames,
                        timestamp_ms=int(time.time() * 1000),
                        corners=(
                            None if markers_mode else current_corners
                        ),
                        sheet_locked=True,
                    )
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
            if key == ord("o") and tracker is not None:
                save_path.mkdir(parents=True, exist_ok=True)
                base = save_path / f"session_{int(time.time())}"
                svg_path, json_path = sequence_module.export_session(
                    tracker, base
                )
                print(f"sessao com {len(tracker.ordered)} tracos "
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
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
