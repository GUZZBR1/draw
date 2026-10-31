"""Testes da vigia (tools/runner/watch.py) sobre as corridas reais.

Os historicos abaixo sao os valores medidos das corridas da
flagship (pairs_metrics.json, draw-lab/ink-flagship): a v2 com o
bug de LR colapsou de verdade; a v3 corrigida aprendeu de verdade.
A vigia precisa separar as duas sem ver o futuro: o colapso da v2
ja era legivel nas tres primeiras epocas.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tools.runner.watch import WatchState, detect_collapse

LAB = os.path.expanduser("~/workspace/draw-lab/ink-flagship")


def _history(path):
    with open(path) as f:
        return json.load(f)["history"]


def test_v2_lrbug_collapses():
    history = _history(os.path.join(LAB, "v5-output-v2-lrbug",
                                    "pairs_metrics.json"))
    verdict = detect_collapse(history)
    assert verdict["collapsed"] is True


def test_v3_healthy_run_not_collapsed():
    history = _history(os.path.join(LAB, "v5-output-v3",
                                    "pairs_metrics.json"))
    verdict = detect_collapse(history)
    assert verdict["collapsed"] is False


def test_short_history_is_not_judged():
    verdict = detect_collapse([{"iou2": 0.001}, {"iou2": 0.001}])
    assert verdict["collapsed"] is False


def test_state_survives_reopen(tmp_path):
    path = str(tmp_path / "watch.json")
    state = WatchState(path)
    state.record("running")
    state.record("running", {"epoch": 7})
    reopened = WatchState(path)
    assert reopened.data["last_status"] == "running"
    assert len(reopened.data["polls"]) == 2
    assert reopened.data["polls"][1]["epoch"] == 7
