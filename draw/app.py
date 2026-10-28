"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --markers                        # folha-modelo impressa
  python main.py --source 1                       # outra webcam local
  python main.py --source http://192.168.0.10:8080/video   # camera via rede

A fonte via rede (MJPEG/RTSP) e o caminho recomendado no WSL. O WSL nao
enxerga webcams USB do Windows por padrao, mas enxerga streams de rede.
Assim todo o processamento (e o ML futuro) fica no Linux. A camera so
envia os frames. Ela pode ser um celular, outro computador ou uma camera IP.

Teclas: s = salvar o canvas como PNG, q/Esc = sair.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from . import ink as ink_module
from . import paper as paper_module


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


def _open_capture(source):
    capture = cv2.VideoCapture(source)
    if capture.isOpened():
        if isinstance(source, int):
            # MJPG comprimido e 720p: o formato cru padrao (YUYV em 1080p)
            # estoura a banda do USB virtual do usbip no WSL e a entrega de
            # quadros trava com timeout de select() no V4L2.
            capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
            )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return capture
    hint = (
        f"Nao consegui abrir a fonte de video: {source!r}. "
    )
    if isinstance(source, int):
        hint += "Nenhuma webcam local respondeu nesse indice. "
        if is_wsl():
            hint += (
                "Voce esta no WSL: por padrao ele nao enxerga a webcam do "
                "Windows (nenhum /dev/video* existe ai). Use uma camera via "
                "rede: python main.py --source http://<ip-da-camera>:8080/video. "
                "Veja a secao 'Rodando no WSL' do README."
            )
        else:
            hint += "Confira se a webcam esta conectada e livre."
    else:
        hint += (
            "Confira se a URL esta certa, se a camera esta transmitindo e "
            "se este computador alcanca o endereco dela na rede."
        )
    raise RuntimeError(hint)


def _fit(image, max_h=700):
    h, w = image.shape[:2]
    if h <= max_h:
        return image
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h))


def run(source=0, save_dir="captures", markers_mode=False):
    capture = _open_capture(parse_source(source))
    save_path = Path(save_dir)
    if markers_mode:
        print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
    else:
        print("Draw v0 rodando. Mostre uma folha para a camera.")
    print("Teclas: s = salvar canvas, q/Esc = sair.")

    current_corners = None
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
                print(
                    "A camera abriu, mas parou de entregar quadros. "
                    "Feche outros programas usando a camera e tente de novo."
                )
                break
            continue
        falhas_seguidas = 0

        if markers_mode:
            warped, markers = paper_module.warp_paper(frame)
        else:
            markers = {}
            should_detect = (
                current_corners is None
                or sheet_missed
                or frame_number % _SHEET_DETECT_INTERVAL == 0
            )
            detected_corners = (
                paper_module.detect_sheet(frame) if should_detect else None
            )
            if detected_corners is not None:
                if current_corners is None:
                    current_corners = detected_corners.copy()
                    pending_corners = None
                    pending_hits = 0
                    frames_since_seen = 0
                    sheet_missed = False
                elif (
                    _corner_distance(current_corners, detected_corners)
                    <= _SHEET_NEAR_LIMIT_PX
                ):
                    current_corners = _blend_corners(
                        current_corners, detected_corners
                    )
                    pending_corners = None
                    pending_hits = 0
                    frames_since_seen = 0
                    sheet_missed = False
                else:
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

                    if pending_hits >= _SHEET_JUMP_CONFIRMATIONS:
                        current_corners = pending_corners.copy()
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
                cv2.putText(
                    status,
                    "Folha detectada",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
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
            if key in (ord("q"), 27):
                break
            continue

        key = cv2.waitKey(1) & 0xFF
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
    parser.add_argument("--save-dir", default="captures")
    args = parser.parse_args()
    run(
        source=args.source,
        save_dir=args.save_dir,
        markers_mode=args.markers,
    )


if __name__ == "__main__":
    main()
