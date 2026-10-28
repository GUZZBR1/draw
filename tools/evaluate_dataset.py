"""Avalia o pipeline do Draw contra um dataset com resposta esperada.

O selftest e a verificação geométrica provam o pipeline em cenas
sintéticas. Este avaliador mede o mesmo pipeline em fotos reais (ou
em qualquer caso com verdade declarada): detecção da folha, erro de
canto, recall/precisão de tinta e contagem de traços, com as mesmas
métricas do selftest, caso a caso e em agregado.

Um caso é uma pasta `cases/<id>/` com `truth.json` e a foto indicada
nele. O manifesto (`dataset/manifest.json`) é o índice: lista os
casos e seus splits. Regras duras, herdadas do Senado:

- Foto sem `truth.json`, sem consentimento ou sem licença não entra:
  o manifesto é recusado, não avaliado pela metade.
- Split `test` é held-out: só mede, nunca calibra. O relatório separa
  `dev` e `test` sempre; nunca agrega os dois sem dizer qual é qual.
- A verdade nunca pode ser a saída do próprio pipeline (`extract_ink`
  ou `detect_sheet`): isso é vazamento de rótulo. Cantos são revisados
  por humano; tinta traçada, quando existe, vem de traçado manual ou
  de cena controlada; forma, quando declarada, vem de quem desenhou,
  com tolerância explícita.
- Sem webcam, sem janela, determinístico: a seed é fixada no início
  e nenhum passo usa aleatoriedade não semeada.

Saída: 0 quando todos os casos passam nos portões do `baseline` do
manifesto, 1 quando algum caso falha, 2 quando o manifesto ou um
`truth.json` é inválido. Um manifesto com zero casos é válido: o
relatório diz "0 casos" e a saída é 0, sem fingir medição.

Uso:
  python tools/evaluate_dataset.py dataset/manifest.json --split all
  python tools/evaluate_dataset.py dataset/manifest.json --split test --report out/relatorio.md
  python tools/evaluate_dataset.py dataset/manifest.json --synthetic
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config, ink, paper, strokes  # noqa: E402

MANIFEST_VERSION = 1

# Campos obrigatórios de truth.json. `sheet_corners` é obrigatório
# quando o caso espera folha; casos negativos usam expects_sheet=false
# e não precisam de cantos. A ordem dos cantos é TL, TR, BR, BL, a
# mesma ordem que detect_sheet devolve.
_REQUIRED_TRUTH_FIELDS = (
    "id",
    "split",
    "photo",
    "paper",
    "lighting",
    "consent",
    "license",
)
_VALID_SPLITS = ("dev", "test")


@dataclass
class CaseResult:
    id: str
    split: str
    expects_sheet: bool
    detected: bool
    detected_corners: list | None
    verdict: str | None
    corner_error_max_px: float | None
    corner_error_mean_px: float | None
    ink_recall: float | None
    ink_precision: float | None
    stroke_count: int | None
    status: str  # "pass", "fail" ou "error"
    failure_class: str  # "ok", "load", "detection", "corner", "ink"

    def as_dict(self):
        return asdict(self)


def validate_truth(truth, case_dir):
    """Retorna a lista de erros de um truth.json (vazia = válido)."""
    errors = []
    if not isinstance(truth, dict):
        return ["truth.json não é um objeto JSON"]
    for field in _REQUIRED_TRUTH_FIELDS:
        if field not in truth:
            errors.append(f"campo obrigatório ausente: {field}")
    if truth.get("split") not in _VALID_SPLITS:
        errors.append(f"split inválido: {truth.get('split')!r}")
    if truth.get("consent") is not True:
        errors.append("consent precisa ser true para o caso entrar")
    if not truth.get("license"):
        errors.append("license ausente: padrão é internal-only")
    expects_sheet = truth.get("expects_sheet")
    if expects_sheet is None:
        expects_sheet = truth.get("sheet_corners") is not None
    if expects_sheet and not truth.get("sheet_corners"):
        errors.append("caso espera folha mas não tem sheet_corners")
    if truth.get("sheet_corners") is not None:
        corners = truth["sheet_corners"]
        if len(corners) != 4 or any(len(p) != 2 for p in corners):
            errors.append("sheet_corners precisa ter 4 pontos [x, y]")
    photo = truth.get("photo")
    if photo and not (Path(case_dir) / photo).is_file():
        errors.append(f"foto não encontrada no caso: {photo}")
    ink_mask = truth.get("ink_mask")
    if ink_mask and not (Path(case_dir) / ink_mask).is_file():
        errors.append(f"ink_mask não encontrada no caso: {ink_mask}")
    return errors


def load_manifest(manifest_path):
    """Lê e valida um manifesto. Retorna (manifesto, casos resolvidos).

    Cada caso resolvido é (case_id, split, case_dir). Levanta
    ValueError com a lista de problemas quando o manifesto é inválido.
    """
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifesto ilegível: {exc}") from exc
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            f"manifest_version esperado: {MANIFEST_VERSION}, "
            f"encontrado: {manifest.get('manifest_version')!r}"
        )
    baseline = manifest.get("baseline") or {}
    for field in ("min_detection_rate", "max_corner_error_px",
                  "min_ink_recall"):
        if field not in baseline:
            raise ValueError(f"baseline sem o campo: {field}")
    root = manifest_path.parent
    cases = []
    errors = []
    seen_ids = set()
    for entry in manifest.get("cases", []):
        case_id = entry.get("id")
        case_dir = root / entry.get("dir", f"cases/{case_id}")
        if case_id in seen_ids:
            errors.append(f"caso duplicado no manifesto: {case_id}")
        seen_ids.add(case_id)
        truth_path = case_dir / "truth.json"
        if not truth_path.is_file():
            errors.append(f"caso {case_id}: truth.json ausente em {case_dir}")
            continue
        try:
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"caso {case_id}: truth.json inválido: {exc}")
            continue
        truth_errors = validate_truth(truth, case_dir)
        if truth_errors:
            errors.extend(f"caso {case_id}: {e}" for e in truth_errors)
            continue
        if truth["id"] != case_id:
            errors.append(
                f"caso {case_id}: id do truth.json é {truth['id']!r}"
            )
            continue
        if entry.get("split", truth["split"]) != truth["split"]:
            errors.append(f"caso {case_id}: split diverge do truth.json")
            continue
        cases.append((case_id, truth["split"], case_dir))
    if errors:
        raise ValueError("\n".join(errors))
    return manifest, cases


def evaluate_frame(frame, truth, max_corner_error_px=12.0,
                   min_ink_recall=None):
    """Avalia um frame em memória contra uma verdade declarada.

    `truth` usa o mesmo formato de truth.json, com duas extensões
    opcionais para uso em memória: `ink_mask_array` (máscara 850×1200
    no espaço da folha retificada) no lugar de `ink_mask` em arquivo.
    Não toca em disco, não abre janela. O portão de canto padrão é o
    do selftest (12 px); o manifesto pode apertá-lo via baseline.
    """
    case_id = truth.get("id", "?")
    split = truth.get("split", "dev")
    expects_sheet = truth.get("expects_sheet")
    if expects_sheet is None:
        expects_sheet = truth.get("sheet_corners") is not None
    if frame is None or frame.size == 0:
        return CaseResult(
            id=case_id, split=split, expects_sheet=bool(expects_sheet),
            detected=False, detected_corners=None, verdict=None,
            corner_error_max_px=None, corner_error_mean_px=None,
            ink_recall=None, ink_precision=None,
            stroke_count=None, status="error", failure_class="load",
        )
    corners, verdict = paper.detect_sheet_with_confidence(frame)
    detected = verdict == "paper"
    corner_max = corner_mean = None
    if detected and truth.get("sheet_corners") is not None:
        expected = np.asarray(truth["sheet_corners"], dtype=np.float64)
        errors = np.linalg.norm(
            np.asarray(corners, dtype=np.float64) - expected, axis=1
        )
        corner_max = float(errors.max())
        corner_mean = float(errors.mean())
    ink_recall = ink_precision = None
    stroke_count = None
    if detected:
        warped = paper.warp_sheet(frame, corners)
        recovered = ink.extract_ink(warped)
        stroke_count = len(strokes.extract_strokes(recovered))
        truth_mask = truth.get("ink_mask_array")
        if truth_mask is not None:
            truth_px = int((np.asarray(truth_mask) > 0).sum())
            recovered_px = int((recovered > 0).sum())
            hits = int(
                ((recovered > 0) & (np.asarray(truth_mask) > 0)).sum()
            )
            ink_recall = hits / max(truth_px, 1) if truth_px else None
            ink_precision = hits / max(recovered_px, 1)
    if not expects_sheet:
        status = "pass" if not detected else "fail"
        failure = "ok" if not detected else "detection"
    elif not detected:
        status, failure = "fail", "detection"
    elif corner_max is not None and corner_max > max_corner_error_px:
        status, failure = "fail", "corner"
    elif (min_ink_recall is not None and ink_recall is not None
          and ink_recall < min_ink_recall):
        status, failure = "fail", "ink"
    else:
        status, failure = "pass", "ok"
    return CaseResult(
        id=case_id, split=split, expects_sheet=bool(expects_sheet),
        detected=detected,
        detected_corners=(
            [[float(x), float(y)] for x, y in corners]
            if corners is not None else None
        ),
        verdict=verdict, corner_error_max_px=corner_max,
        corner_error_mean_px=corner_mean, ink_recall=ink_recall,
        ink_precision=ink_precision, stroke_count=stroke_count,
        status=status, failure_class=failure,
    )


def evaluate_case(case_dir, baseline=None):
    """Avalia um caso em disco (foto + truth.json)."""
    case_dir = Path(case_dir)
    truth = json.loads((case_dir / "truth.json").read_text(encoding="utf-8"))
    frame = cv2.imread(str(case_dir / truth["photo"]))
    if truth.get("ink_mask"):
        mask = cv2.imread(str(case_dir / truth["ink_mask"]),
                          cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape[:2] != (
            config.PAPER_H, config.PAPER_W
        ):
            # A verdade de tinta vive no espaço da folha retificada.
            mask = cv2.resize(mask, (config.PAPER_W, config.PAPER_H))
        truth = dict(truth, ink_mask_array=mask)
    if baseline is None:
        return evaluate_frame(frame, truth)
    return evaluate_frame(
        frame, truth,
        max_corner_error_px=baseline["max_corner_error_px"],
        min_ink_recall=baseline["min_ink_recall"],
    )


def _aggregate(results, baseline):
    detected_expected = [r for r in results if r.expects_sheet]
    detected_ok = [r for r in detected_expected if r.detected]
    detection_rate = (
        len(detected_ok) / len(detected_expected)
        if detected_expected else None
    )
    corner_errors = [
        r.corner_error_max_px for r in detected_ok
        if r.corner_error_max_px is not None
    ]
    recalls = [
        r.ink_recall for r in results if r.ink_recall is not None
    ]
    passed = all(r.status == "pass" for r in results)
    if detection_rate is not None and (
        detection_rate < baseline["min_detection_rate"]
    ):
        passed = False
    if corner_errors and max(corner_errors) > baseline["max_corner_error_px"]:
        passed = False
    if recalls and (
        sum(recalls) / len(recalls) < baseline["min_ink_recall"]
    ):
        passed = False
    return {
        "cases": len(results),
        "passed": sum(1 for r in results if r.status == "pass"),
        "failed": sum(1 for r in results if r.status == "fail"),
        "errors": sum(1 for r in results if r.status == "error"),
        "detection_rate": detection_rate,
        "corner_error_max_px": max(corner_errors) if corner_errors else None,
        "corner_error_mean_px": (
            sum(corner_errors) / len(corner_errors) if corner_errors else None
        ),
        "ink_recall_mean": (
            sum(recalls) / len(recalls) if recalls else None
        ),
        "pass": passed,
    }


def evaluate_manifest(manifest_path, split="all"):
    """Avalia os casos de um manifesto, filtrados por split."""
    manifest, cases = load_manifest(manifest_path)
    baseline = manifest["baseline"]
    results = []
    for case_id, case_split, case_dir in cases:
        if split != "all" and case_split != split:
            continue
        results.append(evaluate_case(case_dir, baseline))
    by_split = {}
    for name in _VALID_SPLITS:
        subset = [r for r in results if r.split == name]
        by_split[name] = _aggregate(subset, baseline)
    overall = _aggregate(results, baseline)
    return {
        "manifest": str(manifest_path),
        "split": split,
        "baseline": baseline,
        "results": [r.as_dict() for r in results],
        "by_split": by_split,
        "overall": overall,
    }


def _load_selftest_builders():
    """Carrega os builders sintéticos do selftest, sem rodá-lo."""
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "draw_selftest", root / "tests" / "selftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_cases():
    """Casos sintéticos que provam o avaliador antes da 1ª foto real.

    Reutiliza as cenas do selftest: folha sem marcadores e folha
    aninhada (com cantos e tinta esperados), bandeja branca (negativo:
    não é papel) e folha branca lisa (veredito blank_maybe, não papel).
    """
    builders = _load_selftest_builders()
    cases = []
    scene, gt_ink, gt_corners = builders.build_markerless_scene()
    cases.append(("synthetic-markerless", scene, {
        "id": "synthetic-markerless", "split": "dev",
        "sheet_corners": gt_corners.tolist(), "ink_mask_array": gt_ink,
    }))
    scene, gt_ink, gt_corners = builders.build_nested_scene()
    cases.append(("synthetic-nested", scene, {
        "id": "synthetic-nested", "split": "dev",
        "sheet_corners": gt_corners.tolist(), "ink_mask_array": gt_ink,
    }))
    negatives = builders.build_negative_scenes()
    cases.append(("synthetic-tray-negative", negatives["bandeja branca"], {
        "id": "synthetic-tray-negative", "split": "dev",
        "expects_sheet": False,
    }))
    cases.append(("synthetic-blank-sheet", builders.build_blank_sheet_scene(), {
        "id": "synthetic-blank-sheet", "split": "dev",
        "expects_sheet": False,
    }))
    return cases


def evaluate_synthetic(baseline=None):
    """Avalia as cenas sintéticas e devolve o mesmo formato de relatório."""
    baseline = baseline or {
        "min_detection_rate": 1.0,
        "max_corner_error_px": 12.0,
        "min_ink_recall": 0.90,
    }
    results = [
        evaluate_frame(
            frame, truth,
            max_corner_error_px=baseline["max_corner_error_px"],
            min_ink_recall=baseline["min_ink_recall"],
        )
        for _, frame, truth in synthetic_cases()
    ]
    return {
        "manifest": "synthetic",
        "split": "all",
        "baseline": baseline,
        "results": [r.as_dict() for r in results],
        "by_split": {"dev": _aggregate(results, baseline), "test": _aggregate([], baseline)},
        "overall": _aggregate(results, baseline),
    }


def render_report_markdown(report):
    """Relatório em Markdown: tabela por caso + agregados por split."""
    lines = ["# Relatório do dataset", ""]
    lines.append(f"Manifesto: `{report['manifest']}` · split: `{report['split']}`")
    lines.append("")
    baseline = report["baseline"]
    lines.append(
        f"Baseline: detecção ≥ {baseline['min_detection_rate']:.0%}, "
        f"erro de canto ≤ {baseline['max_corner_error_px']:.0f}px, "
        f"recall de tinta ≥ {baseline['min_ink_recall']:.0%}"
    )
    lines.append("")
    if not report["results"]:
        lines.append("0 casos neste split: nada a medir.")
        lines.append("")
    else:
        lines.append("| caso | split | folha? | veredito | erro canto (máx) | recall tinta | traços | status | falha |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in report["results"]:
            corner = (
                f"{r['corner_error_max_px']:.2f}px"
                if r["corner_error_max_px"] is not None else "—"
            )
            recall = (
                f"{r['ink_recall']:.2%}" if r["ink_recall"] is not None else "—"
            )
            strokes_n = r["stroke_count"] if r["stroke_count"] is not None else "—"
            lines.append(
                f"| {r['id']} | {r['split']} | "
                f"{'sim' if r['expects_sheet'] else 'não'} | "
                f"{r['verdict'] or '—'} | {corner} | {recall} | "
                f"{strokes_n} | {r['status']} | {r['failure_class']} |"
            )
        lines.append("")
    for name, agg in report["by_split"].items():
        rate = (
            f"{agg['detection_rate']:.0%}"
            if agg["detection_rate"] is not None else "—"
        )
        recall = (
            "—" if agg["ink_recall_mean"] is None
            else f"{agg['ink_recall_mean']:.2%}"
        )
        lines.append(
            f"- {name}: {agg['cases']} casos, {agg['passed']} passaram, "
            f"{agg['failed']} falharam, detecção {rate}, "
            f"recall médio de tinta {recall}"
        )
    lines.append("")
    lines.append(f"Resultado geral: {'PASS' if report['overall']['pass'] else 'FAIL'}")
    return "\n".join(lines) + "\n"


def _write_overlays(report, cases_frames, overlays_dir):
    """Overlays de revisão: verdade em azul, detecção em verde."""
    overlays_dir = Path(overlays_dir)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    for result, (frame, truth) in zip(report["results"], cases_frames):
        if frame is None:
            continue
        overlay = frame.copy()
        if truth.get("sheet_corners") is not None:
            outline = np.asarray(
                truth["sheet_corners"], dtype=np.int32
            ).reshape((-1, 1, 2))
            cv2.polylines(overlay, [outline], True, (255, 0, 0), 3)
        detected = result.get("detected_corners")
        if detected is not None:
            outline = np.asarray(
                detected, dtype=np.int32
            ).reshape((-1, 1, 2))
            cv2.polylines(overlay, [outline], True, (0, 255, 0), 3)
        cv2.imwrite(str(overlays_dir / f"{result['id']}.png"), overlay)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Avalia o pipeline contra um dataset com verdade"
    )
    parser.add_argument("manifest", nargs="?",
                        help="manifesto JSON do dataset")
    parser.add_argument("--split", choices=("dev", "test", "all"),
                        default="all", help="split a avaliar")
    parser.add_argument("--report", help="arquivo de saída do relatório")
    parser.add_argument("--format", choices=("markdown", "json"),
                        default="markdown")
    parser.add_argument("--synthetic", action="store_true",
                        help="avalia cenas sintéticas, sem manifesto")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed fixa para determinismo")
    parser.add_argument("--overlays-dir",
                        help="pasta de overlays de revisão (só verdade)")
    args = parser.parse_args(argv)
    np.random.seed(args.seed)

    if args.synthetic:
        report = evaluate_synthetic()
        frames = [(frame, truth) for _, frame, truth in synthetic_cases()]
    else:
        if not args.manifest:
            parser.error("manifesto é obrigatório sem --synthetic")
        try:
            report = evaluate_manifest(args.manifest, args.split)
        except ValueError as exc:
            print(f"Manifesto inválido:\n{exc}", file=sys.stderr)
            return 2
        frames = []
        if args.overlays_dir:
            manifest, cases = load_manifest(args.manifest)
            for case_id, case_split, case_dir in cases:
                if args.split != "all" and case_split != args.split:
                    continue
                truth = json.loads(
                    (case_dir / "truth.json").read_text(encoding="utf-8")
                )
                frame = cv2.imread(str(case_dir / truth["photo"]))
                frames.append((frame, truth))
    if args.format == "json":
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    else:
        text = render_report_markdown(report)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.overlays_dir and frames:
        _write_overlays(report, frames, args.overlays_dir)
    return 0 if report["overall"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
