"""Loop principal: webcam -> folha retificada -> canvas digital ao vivo.

Uso: python main.py [--camera 0] [--save-dir captures]
Teclas: s = salvar o canvas como PNG, v = salvar traços vetoriais
(SVG + JSON), q/Esc = sair.
"""

import argparse
import time
from pathlib import Path

import cv2

from . import ink as ink_module
from . import paper as paper_module
from . import strokes as strokes_module


def _fit(image, max_h=700):
    h, w = image.shape[:2]
    if h <= max_h:
        return image
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h))


def run(camera_index=0, save_dir="captures"):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Nao consegui abrir a webcam {camera_index}. "
            "Confira se ela esta conectada e livre."
        )
    save_path = Path(save_dir)
    print("Draw v0 rodando. Mostre a folha-modelo para a webcam.")
    print("Teclas: s = salvar canvas, v = salvar tracos vetoriais, "
          "q/Esc = sair.")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

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
            if key == ord("v"):
                save_path.mkdir(parents=True, exist_ok=True)
                base = save_path / f"strokes_{int(time.time())}"
                strokes = strokes_module.extract_strokes(ink_mask)
                svg_path, json_path = strokes_module.export_strokes(
                    strokes, base
                )
                print(f"{len(strokes)} tracos salvos em {svg_path} "
                      f"e {json_path}")
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
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--save-dir", default="captures")
    args = parser.parse_args()
    run(camera_index=args.camera, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
