"""U1/U2/U3: acumulador de tinta com confirmacao entre passadas.

O defeito (draw-lab/page-turn-latency-2026-11-01): a uniao direta
de binarios limiarizados integrava o tremor de registro entre
passadas — pagina vazia parada acumulava FP de mobiliario
deslocado sem limite e traco real engrossava com o tempo. Estes
testes alimentam o acumulador com mapas de probabilidade
sinteticos (sem modelo) e afirmam o comportamento novo:

- FP que treme de posicao nao confirma nunca;
- traco estavel confirma (forte na 2a passada, lapis fraco na 3a);
- pagina parada assenta (trava U2) e escrita nova reabre;
- virada de pagina dispara o evento U3; borracha, nao.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402

H, W = 120, 160


def _prob(rects=(), value=0.95):
    prob = np.zeros((H, W), dtype=np.float32)
    for (y0, y1, x0, x1) in rects:
        prob[y0:y1, x0:x1] = value
    return prob


def _acc():
    return app_module._ConfirmedInkAccumulator(threshold=0.7)


def test_jittered_false_positive_never_confirms():
    acc = _acc()
    # FP forte mudando de posicao a cada passada, sem repetir
    # posicao dentro da janela de voto: nenhum pixel soma 1,4.
    offsets = [
        (0, 0), (4, 0), (0, 4), (4, 4), (-4, 0), (0, -4),
        (-4, -4), (4, -4), (-4, 4), (8, 0), (0, 8), (8, 8),
    ]
    for dy, dx in offsets:
        result = acc.add(_prob(rects=[(20 + dy, 23 + dy, 20 + dx, 23 + dx)]))
        assert result["new_px"] == 0
    assert int(np.count_nonzero(acc.canvas)) == 0


def test_stable_strong_trace_confirms_on_second_pass():
    acc = _acc()
    rect = (30, 40, 30, 60)
    first = acc.add(_prob(rects=[rect]))
    assert first["new_px"] == 0  # 1 passada: soma 0,95 < 1,4
    second = acc.add(_prob(rects=[rect]))
    assert second["new_px"] == 10 * 30
    third = acc.add(_prob(rects=[rect]))
    assert third["new_px"] == 0  # confirmado nao cresce


def test_weak_pencil_confirms_on_third_pass():
    acc = _acc()
    rect = (50, 54, 30, 90)
    acc.add(_prob(rects=[rect], value=0.55))
    second = acc.add(_prob(rects=[rect], value=0.55))
    assert second["new_px"] == 0  # soma 1,10 < 1,4
    third = acc.add(_prob(rects=[rect], value=0.55))
    assert third["new_px"] == 4 * 60  # soma 1,65: lapis entra


def test_settled_lock_after_quiet_passes():
    acc = _acc()
    rect = (30, 40, 30, 60)
    settled_at = None
    for i in range(10):
        result = acc.add(_prob(rects=[rect]))
        if result["settled"] and settled_at is None:
            settled_at = i
    assert settled_at is not None
    # Depois de assentar, mais passadas iguais nao mudam o canvas.
    before = acc.canvas
    for _ in range(5):
        acc.add(_prob(rects=[rect]))
    assert np.array_equal(acc.canvas, before)


def test_new_stroke_reopens_settled_page_and_enters():
    acc = _acc()
    rect = (30, 40, 30, 60)
    for _ in range(8):
        acc.add(_prob(rects=[rect]))
    assert acc.settled
    stroke = (70, 100, 100, 130)  # 900 px fortes novos
    reopened = acc.add(_prob(rects=[rect, stroke]))
    assert not reopened["settled"]
    entered = None
    for i in range(3):
        result = acc.add(_prob(rects=[rect, stroke]))
        if result["new_px"] >= 900:
            entered = i
            break
    assert entered is not None, "traco novo nao entrou apos reabrir"


def test_page_turn_fires_once_after_two_decorrelated_passes():
    acc = _acc()
    page_a = (10, 70, 10, 70)  # 3600 px confirmados
    for _ in range(4):
        acc.add(_prob(rects=[page_a]))
    assert int(np.count_nonzero(acc.canvas)) == 3600
    empty = _prob()
    first = acc.add(empty)
    assert not first["turn"]
    second = acc.add(empty)
    assert second["turn"]
    third = acc.add(empty)
    assert not third["turn"]  # uma vez por geracao


def test_eraser_does_not_fire_turn():
    acc = _acc()
    page = (10, 70, 10, 70)
    for _ in range(4):
        acc.add(_prob(rects=[page]))
    # Apaga 40% do desenho: a mascara segue cobrindo 60% do canvas,
    # muito acima do teto de 30% da virada.
    erased = _prob(rects=[(10, 70, 10, 46)])
    for _ in range(4):
        result = acc.add(erased)
        assert not result["turn"]


def test_rigid_shift_without_new_ink_does_not_grow_canvas():
    """Invariancia de posicao: o warp treme (deslocamento rigido
    aleatorio de +-3 px por passada, sem tinta nova). No desenho
    novo, a pagina assenta (U2) e a geometria congela no laco —
    daqui pra frente o warp e identico e o canvas nao anda. A
    serie abaixo alimenta deslocamentos ate assentar e depois
    congela (deslocamento 0), como o sistema faz. Medido nesta VM
    com a semente abaixo: assenta na passada 8 com +21% sobre o
    traco (o nucleo + a borda que o voto confirmou antes da
    trava); a uniao antiga, alimentada com os 30 deslocamentos,
    fecha em +90%. Tolerancia declarada: <= +25%."""
    rng = np.random.default_rng(20261101)
    rect = (50, 60, 60, 90)  # 300 px
    offsets = [
        (int(rng.integers(-3, 4)), int(rng.integers(-3, 4)))
        for _ in range(30)
    ]
    acc = _acc()
    union = None
    settled_at = None
    for i, (dy, dx) in enumerate(offsets):
        # Uniao antiga: os 30 deslocamentos, sem trava.
        old_prob = _prob(rects=[(
            rect[0] + dy, rect[1] + dy, rect[2] + dx, rect[3] + dx
        )])
        old_mask = (old_prob > 0.7).astype(np.uint8) * 255
        union = old_mask if union is None else np.maximum(union, old_mask)
        # Desenho novo: deslocamentos ate assentar; depois a
        # geometria congela e o warp passa a ser o de deslocamento 0.
        if settled_at is not None:
            dy = dx = 0
        prob = _prob(rects=[(
            rect[0] + dy, rect[1] + dy, rect[2] + dx, rect[3] + dx
        )])
        result = acc.add(prob)
        if result["settled"] and settled_at is None:
            settled_at = i
    assert settled_at is not None
    new_px = int(np.count_nonzero(acc.canvas))
    old_px = int(np.count_nonzero(union))
    assert old_px >= 1.8 * 300  # uniao antiga: halo inteiro entra
    assert new_px <= 1.25 * 300  # confirmado + trava: ~so o nucleo


def test_worker_settles_probes_and_reopens(monkeypatch):
    """Trabalhadora: assentada para de rodar passadas, sonda e
    escrita nova volta a entrar no canvas publicado."""
    monkeypatch.setattr(
        app_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.01
    )
    monkeypatch.setattr(app_module, "_SETTLED_PROBE_SECONDS", 0.05)

    class _FakeInk:
        threshold = 0.7

        def __init__(self):
            self.calls = 0
            self.extra = False

        def ink_prob(self, _warped):
            self.calls += 1
            rects = [(30, 40, 30, 60)]
            if self.extra:
                rects.append((70, 100, 100, 130))
            return _prob(rects=rects)

    fake = _FakeInk()
    worker = app_module._PageCleanWorker(fake).start()
    try:
        warp = np.zeros((H, W, 3), dtype=np.uint8)
        generation = worker.new_generation()
        deadline = time.monotonic() + 5.0
        while not worker.settled and time.monotonic() < deadline:
            worker.submit(warp, generation)
            time.sleep(0.01)
        assert worker.settled, "a pagina nunca assentou"
        calls_settled = fake.calls
        # Submits continuam chegando; passada nova, nao.
        for _ in range(10):
            worker.submit(warp, generation)
            time.sleep(0.01)
        # Sondas acontecem (1 a cada 50 ms aqui), mas sao sondas:
        # muito menos que os ~10 submits do periodo virariam.
        assert fake.calls - calls_settled <= 6
        got = worker.poll(generation, 0)
        assert got is not None
        _version, mask = got
        assert int(np.count_nonzero(mask)) == 300

        # Escrita nova: a sonda enxerga, destrava e o traco entra.
        fake.extra = True
        deadline = time.monotonic() + 5.0
        entered = False
        version = _version
        while time.monotonic() < deadline:
            worker.submit(warp, generation)
            got = worker.poll(generation, version)
            if got is not None:
                version, mask = got
                if int(np.count_nonzero(mask)) >= 1200:
                    entered = True
                    break
            time.sleep(0.01)
        assert entered, "traco novo nao entrou depois de assentar"
    finally:
        worker.stop()
