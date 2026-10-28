"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --source 1                       # outra webcam local
  python main.py --source http://192.168.0.10:8080/video   # camera via rede

A fonte via rede (MJPEG/RTSP) e o caminho recomendado no WSL: o WSL nao
enxerga webcams USB do Windows por padrao, mas enxerga streams de rede.
Assim todo o processamento (e o ML futuro) fica no Linux, e a camera —
celular, outro computador ou uma camera IP — so envia os frames.

Teclas: s = salvar o canvas como PNG, q/Esc = sair.
"""

import argparse
import time
from pathlib import Path

import cv2

from . import ink as ink_module
from . import paper as paper_module


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
                "rede: python main.py --source http://<ip-da-camera>:8080/video "
                "— veja a secao 'Rodando no WSL' do README."
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


def run(source=0, save_dir="captures"):
    capture = _open_capture(parse_source(source))
    save_path = Path(save_dir)
    print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
    print("Teclas: s = salvar canvas, q/Esc = sair.")

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

        warped, markers = paper_module.warp_paper(frame)

        status = frame.copy()
        for (x, y) in markers.values():
            cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
        if warped is None:
            cv2.putText(
                status,
                f"Mostre a folha-modelo ({len(markers)}/4 marcadores)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Draw - camera", _fit(status, 700))
        else:
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
    parser = argparse.ArgumentParser(description="Draw v0 — papel -> digital")
    parser.add_argument(
        "--source",
        default="0",
        help="indice da webcam local (0, 1...) ou URL do stream "
        "(ex.: http://192.168.0.10:8080/video). Padrao: 0",
    )
    parser.add_argument("--save-dir", default="captures")
    args = parser.parse_args()
    run(source=args.source, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
