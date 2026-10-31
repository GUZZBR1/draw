#!/usr/bin/env python3
"""Compara captura/detecção da câmera com e sem o detector neural do Draw.

Não grava imagens ou vídeo. Salva tempos por quadro em CSV e um resumo JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from draw import capture as capture_module  # noqa: E402
from draw import neural as neural_module  # noqa: E402
from draw import paper as paper_module  # noqa: E402
from draw.app import _SheetTracker  # noqa: E402


DEFAULT_MODEL = (
    ROOT / ".worktrees/model-kaggle-real-pilot/models/"
    "kaggle-real-pilot-v1/model.onnx"
)
CSV_FIELDS = [
    "frame",
    "elapsed_s",
    "read_ms",
    "loop_ms",
    "detect_ran",
    "neural_ms",
    "classic_ms",
    "detect_ms",
    "warp_ms",
    "display_ms",
    "processing_ms",
    "paper_detected",
    "classic_fallback",
    "width",
    "height",
]


def _fourcc_text(value: float) -> str | None:
    try:
        code = int(value)
        text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
        return text if text.strip("\x00") else None
    except (TypeError, ValueError, OverflowError):
        return None


def _backend_name(capture) -> str | None:
    try:
        return capture.getBackendName()
    except Exception:
        return None


def _detector_for_frame(frame, tracker, frame_number, detector):
    neural_ms = 0.0
    classic_ms = 0.0
    detect_ran = tracker.should_detect(frame_number)
    used_fallback = False
    corners = verdict = None

    if detect_ran:
        if detector is not None:
            started = time.perf_counter_ns()
            corners, verdict = detector.detect_sheet_with_confidence(frame)
            neural_ms = (time.perf_counter_ns() - started) / 1_000_000
            if corners is None:
                used_fallback = True
                started = time.perf_counter_ns()
                corners, verdict = paper_module.detect_sheet_with_confidence(frame)
                classic_ms = (time.perf_counter_ns() - started) / 1_000_000
        else:
            started = time.perf_counter_ns()
            corners, verdict = paper_module.detect_sheet_with_confidence(frame)
            classic_ms = (time.perf_counter_ns() - started) / 1_000_000

    tracker.update(corners, verdict, detect_ran, time.monotonic())
    return neural_ms, classic_ms, detect_ran, used_fallback


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "max": None}
    data = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(data, 50)), 3),
        "p95": round(float(np.percentile(data, 95)), 3),
        "p99": round(float(np.percentile(data, 99)), 3),
        "max": round(float(np.max(data)), 3),
    }


def _summarize(rows: list[dict], duration_s: float, elapsed_s: float) -> dict:
    def values(field: str) -> list[float]:
        return [float(row[field]) for row in rows if row[field] is not None]

    return {
        "target_seconds": duration_s,
        "elapsed_seconds": round(elapsed_s, 3),
        "frames": len(rows),
        "fps_observed": round(len(rows) / elapsed_s, 3) if elapsed_s > 0 else 0,
        "detections_run": sum(bool(row["detect_ran"]) for row in rows),
        "paper_detected_frames": sum(bool(row["paper_detected"]) for row in rows),
        "classic_fallback_frames": sum(bool(row["classic_fallback"]) for row in rows),
        "timing_ms": {
            field: _percentiles(values(field))
            for field in (
                "read_ms", "loop_ms", "neural_ms", "classic_ms",
                "detect_ms", "warp_ms", "display_ms", "processing_ms",
            )
        },
        "invocation_timing_ms": {
            "detection": _percentiles([
                float(row["detect_ms"]) for row in rows if row["detect_ran"]
            ]),
            "neural": _percentiles([
                float(row["neural_ms"]) for row in rows if row["neural_ms"] > 0
            ]),
            "classic": _percentiles([
                float(row["classic_ms"]) for row in rows if row["classic_ms"] > 0
            ]),
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _measure_phase(
    args,
    phase: str,
    model_path: str | None,
    detector,
    output_dir: Path,
    model_load_ms: float | None = None,
) -> dict:
    print(f"\nFase: {phase}")
    print(f"Fonte: {args.source}; duração: {args.seconds:g}s")
    if args.display:
        print("Janela ao vivo ligada; pressione Ctrl+C para interromper.")
    else:
        print("Sem janela de vídeo; pressione Ctrl+C para interromper.")
    if sys.stdin.isatty() and not args.no_prompt:
        input("Prepare a folha/cena e pressione Enter para começar: ")

    capture = capture_module.open_capture(
        args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=capture_module.backend_from_name(args.backend),
        open_timeout_ms=args.open_timeout_ms,
    )
    tracker = _SheetTracker()
    frame_number = 0
    warmup_done = 0
    rows: list[dict] = []
    read_failures = 0
    start_ns = None
    previous_frame_end = None
    last_progress = 0
    duration_ns = int(args.seconds * 1_000_000_000)

    try:
        while True:
            read_start = time.perf_counter_ns()
            ok, frame = capture.read()
            read_end = time.perf_counter_ns()
            read_ms = (read_end - read_start) / 1_000_000
            if not ok or frame is None:
                read_failures += 1
                time.sleep(0.01)
                continue

            if warmup_done < args.warmup_frames:
                _detector_for_frame(frame, tracker, frame_number, detector)
                if tracker.corners is not None:
                    paper_module.warp_sheet(frame, tracker.corners)
                if args.display:
                    cv2.imshow(f"Camera latency: {phase}", frame)
                    cv2.waitKey(1)
                warmup_done += 1
                frame_number += 1
                continue

            if start_ns is None:
                start_ns = time.perf_counter_ns()
                previous_frame_end = start_ns
            now_ns = time.perf_counter_ns()
            elapsed_s = (now_ns - start_ns) / 1_000_000_000
            if now_ns - start_ns >= duration_ns:
                break

            neural_ms, classic_ms, detect_ran, used_fallback = _detector_for_frame(
                frame, tracker, frame_number, detector
            )
            detect_ms = neural_ms + classic_ms

            warp_ms = 0.0
            if tracker.corners is not None:
                stage_start = time.perf_counter_ns()
                paper_module.warp_sheet(frame, tracker.corners)
                warp_ms = (time.perf_counter_ns() - stage_start) / 1_000_000

            display_ms = 0.0
            if args.display:
                shown = frame.copy()
                if tracker.corners is not None:
                    outline = tracker.corners.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(shown, [outline], True, (0, 255, 0), 2)
                stage_start = time.perf_counter_ns()
                cv2.imshow(f"Camera latency: {phase}", shown)
                cv2.waitKey(1)
                display_ms = (time.perf_counter_ns() - stage_start) / 1_000_000

            end_ns = time.perf_counter_ns()
            processing_ms = detect_ms + warp_ms + display_ms
            rows.append({
                "frame": frame_number,
                "elapsed_s": round((end_ns - start_ns) / 1_000_000_000, 4),
                "read_ms": round(read_ms, 3),
                "loop_ms": round((end_ns - previous_frame_end) / 1_000_000, 3),
                "detect_ran": int(detect_ran),
                "neural_ms": round(neural_ms, 3),
                "classic_ms": round(classic_ms, 3),
                "detect_ms": round(detect_ms, 3),
                "warp_ms": round(warp_ms, 3),
                "display_ms": round(display_ms, 3),
                "processing_ms": round(processing_ms, 3),
                "paper_detected": int(tracker.corners is not None),
                "classic_fallback": int(used_fallback),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            })
            previous_frame_end = end_ns
            frame_number += 1

            progress_s = int((end_ns - start_ns) / 1_000_000_000)
            if progress_s >= last_progress + 10:
                last_progress = progress_s
                print(f"  {min(progress_s, int(args.seconds))}/{args.seconds:g}s — {len(rows)} quadros")

        end = time.perf_counter_ns()
        elapsed_s = ((end - start_ns) / 1_000_000_000) if start_ns else 0.0
        height = int(frame.shape[0]) if frame is not None else None
        width = int(frame.shape[1]) if frame is not None else None
        result = _summarize(rows, args.seconds, elapsed_s)
        result.update({
            "phase": phase,
            "source": args.source,
            "model_path": model_path,
            "model_load_ms": round(model_load_ms, 3) if model_load_ms is not None else None,
            "warmup_frames": warmup_done,
            "read_failures": read_failures,
            "actual_width": width,
            "actual_height": height,
            "reported_fps": float(capture.get(cv2.CAP_PROP_FPS) or 0),
            "reported_fourcc": _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC)),
            "backend": _backend_name(capture),
            "display_enabled": args.display,
            "jpeg_warning_count": None,
            "frame_age_measured": False,
        })
        safe_phase = phase.replace(" ", "_")
        _write_csv(output_dir / f"{safe_phase}.csv", rows)
        (output_dir / f"{safe_phase}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Fim: {result['frames']} quadros, {result['fps_observed']} FPS; "
            f"detecção p50/p95: "
            f"{result['timing_ms']['detect_ms']['p50']}/"
            f"{result['timing_ms']['detect_ms']['p95']} ms."
        )
        print(f"Dados: {output_dir / safe_phase}.csv e .json")
        return result
    finally:
        capture.release()
        if args.display:
            try:
                cv2.destroyWindow(f"Camera latency: {phase}")
            except cv2.error:
                pass


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Mede captura e detecção da folha com e sem ONNX por 60 s cada."
    )
    parser.add_argument("--source", default="0", help="índice, /dev/videoN ou URL (padrão: 0)")
    parser.add_argument(
        "--neural-model",
        default=os.environ.get("DRAW_NEURAL_MODEL") or (str(DEFAULT_MODEL) if DEFAULT_MODEL.is_file() else None),
        help="modelo ONNX; padrão: DRAW_NEURAL_MODEL ou modelo do piloto local",
    )
    parser.add_argument("--seconds", type=float, default=60, help="duração medida de cada fase")
    parser.add_argument("--warmup-frames", type=int, default=10, help="quadros de aquecimento não medidos")
    parser.add_argument("--width", type=int, default=None, help="largura a solicitar à câmera local")
    parser.add_argument("--height", type=int, default=None, help="altura a solicitar à câmera local")
    parser.add_argument("--fps", type=float, default=None, help="FPS a solicitar à câmera local")
    parser.add_argument("--backend", default=None, help="backend OpenCV; padrão: backend do sistema")
    parser.add_argument("--open-timeout-ms", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=None, help="pasta dos CSV/JSON")
    parser.add_argument("--display", action="store_true", help="mostra uma janela ao vivo nas duas fases")
    parser.add_argument("--no-prompt", action="store_true", help="não esperar Enter antes de cada fase")
    args = parser.parse_args()
    if args.seconds <= 0 or args.warmup_frames < 0:
        parser.error("--seconds deve ser positivo e --warmup-frames não pode ser negativo")
    if not args.neural_model:
        parser.error("passe --neural-model ou defina DRAW_NEURAL_MODEL")
    if not Path(args.neural_model).is_file():
        parser.error(f"modelo ONNX não encontrado: {args.neural_model}")
    return args


def main() -> int:
    args = _parse_args()
    try:
        model_detector = neural_module.NeuralSheetDetector(args.neural_model)
        model_load_start = time.perf_counter_ns()
        model_detector.load()
        model_load_ms = (time.perf_counter_ns() - model_load_start) / 1_000_000
        # Fail before opening the camera if the ONNX contract is incompatible.
        model_detector.detect_sheet_with_confidence(
            np.zeros((720, 1280, 3), dtype=np.uint8)
        )
    except (RuntimeError, FileNotFoundError, ValueError, cv2.error) as error:
        print(f"Não consegui preparar o modelo ONNX: {error}", file=sys.stderr)
        return 2
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or ROOT / "captures" / "latency" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    print("Medição local: captura + detector de folha; não grava vídeo ou frames.")
    print("A idade real do quadro não é exposta pelo VideoCapture usado aqui.")
    print(f"ONNX carregado e pré-validado em {model_load_ms:.1f} ms.")
    print(f"Saída: {output_dir}")
    results = []
    try:
        results.append(_measure_phase(args, "sem_modelo", None, None, output_dir))
        results.append(_measure_phase(
            args, "com_modelo", args.neural_model, model_detector, output_dir,
            model_load_ms=model_load_ms,
        ))
    except KeyboardInterrupt:
        print("\nMedição interrompida; os arquivos completos de fases anteriores foram preservados.")
        return 130
    except (capture_module.CaptureError, RuntimeError, FileNotFoundError, cv2.error) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2
    (output_dir / "comparison.json").write_text(
        json.dumps({"phases": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nComparação concluída: {output_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
