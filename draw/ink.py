"""Extração da tinta: separar o que foi desenhado do papel branco.

A separação tem duas camadas, medidas no replay do vídeo de desenho
(~/workspace/draw-lab/stroke-fix-2026-10-29/REPORT.md). A primeira é o
limiar adaptativo de sempre sobre a folha retificada, robusto a
variações de iluminação ao longo da página. A segunda é um veto de
fundo separável: um pixel só é tinta se estiver escuro o bastante
contra a mediana da sua linha inteira e da sua coluna inteira. A
pauta impressa ocupa a linha toda, então a mediana da linha já é o
brilho da pauta e ela não passa no veto; a lombada do caderno e as
bordas da folha ocupam a coluna toda e caem pelo mesmo caminho. O
traço de caneta fica dezenas de níveis abaixo das duas medianas e
passa. Um portão de cor (saturação ou preto absoluto) foi medido e
reprovado neste material: o traço real é cinza de baixa saturação,
igual ao papel nesses canais, e o portão apagava o desenho.

A tinta do canvas não é mais lida de um frame só: o InkAccumulator
junta as observações de uma janela de frames no espaço do papel. Um
pixel entra no canvas quando a fração de tinta entre as amostras
visíveis da janela passa de um limiar de entrada e sai quando cai
abaixo de um limiar de saída menor (histerese). Pixel coberto pela
mão é desconhecido: não conta como tinta nem como papel limpo, o
que preserva o que já foi desenhado embaixo da mão e também
preserva a borracha (apagou e ficou visível como papel limpo por
amostras bastantes, sai do canvas).

No modo com marcadores, as áreas dos marcadores são apagadas da
máscara para não virarem "tinta" no canvas. No modo sem marcadores
não há marcador nenhum para apagar: apagar as mesmas áreas
apagaria desenho de verdade perto dos cantos da página.
"""

from collections import deque
import time

import cv2
import numpy as np

from . import config

_VETO_DELTA = 6.0


def _marker_mask(shape):
    """Máscara (255 = ignorar) cobrindo as áreas dos marcadores."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    radius = config.MARKER_SIZE_PX // 2 + 15
    for center in config.MARKER_CENTERS.values():
        cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask


def _separable_veto(gray):
    """True onde o pixel é escuro contra a linha e a coluna inteiras.

    A mediana de uma linha da folha retificada é o brilho típico
    daquela linha (papel, ou pauta onde a linha é pautada); a da
    coluna idem (papel, ou lombada e bordas onde a coluna as cruza).
    Estruturas que ocupam a linha ou a coluna inteira têm delta
    perto de zero contra a própria mediana e são vetadas; o traço,
    local e bem mais escuro, passa.
    """
    gray_f = gray.astype(np.float32)
    bg_row = np.median(gray_f, axis=1, keepdims=True)
    bg_col = np.median(gray_f, axis=0, keepdims=True)
    return (gray_f <= bg_row - _VETO_DELTA) & (
        gray_f <= bg_col - _VETO_DELTA
    )


def extract_ink(warped, clear_markers=False):
    """Retorna a máscara de tinta (255 = tinta) da folha retificada.

    clear_markers apaga as áreas nominais dos marcadores ArUco dos
    cantos. Só faz sentido no modo com marcadores, onde a folha-modelo
    tem marcadores impressos de verdade; o chamador desse modo é quem
    pede a limpeza. Sem o pedido, nenhum pixel é apagado por posição.

    O veto de fundo separável vale só no caminho sem marcadores, que
    é onde mora o papel pautado de caderno. A folha-modelo impressa
    não tem pauta: nela o veto não derruba artefato nenhum e só custa
    recall (medido no selftest: 96,03% sem o veto, 93,84% com ele).
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=12,
    )
    ink = cv2.bitwise_not(binary)  # tinta escura vira branco (255)
    if clear_markers:
        ink[_marker_mask(warped.shape) > 0] = 0
    # Remove ruído pequeno (poeira, textura do papel).
    kernel = np.ones((2, 2), dtype=np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    if not clear_markers:
        ink[~_separable_veto(gray)] = 0
    return ink


# Amostras da pagina base e delta da tinta da sessao. A pagina base
# e a mediana dos primeiros quadros retificados depois que o
# usuario manda comecar (tecla C no app): pauta, margem impressa,
# lombada e o que ja estava desenhado entram nela e somem da
# observacao. So conta como tinta o que escurece DEPOIS da base
# fechada. O delta de 12 niveis fica logo abaixo do traco real
# medido contra o papel (~13 niveis, auditoria da tinta) e acima
# da cintilacao de luz; o resto quem limpa e o voto temporal do
# InkAccumulator sobre esta mascara. A observacao normaliza a
# iluminacao do quadro pela razao das medianas contra a base
# (ganho limitado), para uma mudanca global de luz nao virar
# tinta na pagina inteira.
_SESSION_BASE_SAMPLES = 12
_SESSION_INK_DELTA = 12.0


class SessionBaselineInk:
    """Tinta da sessao: o que escureceu em relacao a pagina base.

    Diferente de extract_ink, que julga cada quadro sozinho contra
    limiares absolutos e por isso entrega a estrutura do caderno
    (pauta, margem, lombada) como tinta, esta fonte compara o
    quadro com a pagina como ela estava quando a sessao abriu.
    Desenho preexistente vira fundo por construcao: a pergunta que
    ela responde e "o que foi desenhado nesta sessao".
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._samples = []
        self._baseline_min = None
        self.reference = None
        self._ref_median = None

    @property
    def ready(self):
        return self._baseline_min is not None

    def add_sample(self, warped):
        """Alimenta a base; a mediana fecha na decima segunda amostra."""
        if self.ready:
            return
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        self._samples.append(gray)
        if len(self._samples) >= _SESSION_BASE_SAMPLES:
            base = np.median(np.stack(self._samples), axis=0)
            base = cv2.GaussianBlur(
                base.astype(np.float32), (3, 3), 0)
            # A referencia cheia fica guardada: e contra ela que um
            # realinhamento futuro (ECC/fase) mede o deslocamento
            # do quadro atual quando o caderno se mexe.
            self.reference = base
            self._ref_median = float(np.median(base))
            # Minimo local 3x3: a estrutura escura da base ganha um
            # halo de 1 px que absorve o jitter residual do warp,
            # para a pauta nao virar tinta quando o quadro treme.
            self._baseline_min = cv2.erode(
                base, np.ones((3, 3), dtype=np.uint8))
            self._samples = []

    def observe(self, warped):
        """Mascara (255 = tinta) do que escureceu contra a base."""
        if not self.ready:
            return np.zeros(warped.shape[:2], dtype=np.uint8)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        cur_median = float(np.median(gray))
        gain = 1.0
        if cur_median > 1.0 and self._ref_median is not None:
            gain = float(np.clip(
                self._ref_median / cur_median, 0.85, 1.18))
        blurred = cv2.GaussianBlur(gray * gain, (3, 3), 0)
        delta = self._baseline_min - blurred
        return np.where(
            delta >= _SESSION_INK_DELTA, 255, 0).astype(np.uint8)


def render_canvas(ink_mask):
    """Renderiza o canvas digital: fundo branco, tinta preta."""
    canvas = np.full(
        (config.PAPER_H, config.PAPER_W, 3), 255, dtype=np.uint8
    )
    canvas[ink_mask > 0] = (0, 0, 0)
    return canvas


class InkAccumulator:
    """Canvas acumulativo no espaço do papel, com histerese temporal.

    Cada update recebe a observação de tinta de um frame (a máscara
    de extract_ink) e a máscara do que está visível nesse frame
    (False sob a mão; None quando não há máscara de mão, caso em que
    tudo conta como visível). A evidência de tinta é a observação
    dilatada de tolerance pixels, para absorver o jitter residual do
    warp entre frames vizinhos; evidência só conta onde o pixel está
    visível. Sobre a janela das últimas window amostras, cada pixel
    tem uma fração de tinta: votos de tinta dividido por votos
    visíveis. O pixel entra no canvas quando a fração passa de
    enter e sai quando cai a exit ou abaixo, julgado apenas onde a
    janela já viu o pixel ao menos min_visible vezes. Pixel sem
    julgamento suficiente congela no estado atual: é o que faz a
    região sob a mão não ganhar nem perder tinta.

    Os parâmetros foram medidos no replay do vídeo de desenho: a
    presença por pixel do traço real fica perto de 0,55 e a do
    artefato de pauta abaixo de 0,3, então enter=0,45 separa os
    dois; exit=0,20 segura o traço através da cintilação sem segurar
    o artefato, cuja fração desaba quando a luz muda.

    Além do canvas, o acumulador guarda dois carimbos por pixel,
    first_seen_ms e last_update_ms (uint32, 850×1200 cada, 0 =
    sem carimbo): quando o pixel entrou no canvas e quando a
    evidência dele se renovou pela última vez. É a ordem temporal
    dos traços que o raster sozinho perde, e é o que a vetorização
    consome para ordenar as polilinhas. A época dos carimbos é o
    relógio monotônico em milissegundos (time.monotonic), e o
    instante que vale é o de CAPTURA do frame, passado por quem
    chama em update(t_capture_ms=...): carimbo de processamento
    mediria a fila do pipeline, não o desenho. A manutenção dos
    carimbos só acontece em pixel visível no frame corrente:
    pixel sob a mão congela os dois carimbos junto com o estado.
    """

    def __init__(
        self,
        shape=(config.PAPER_H, config.PAPER_W),
        window=15,
        enter=0.45,
        exit=0.20,
        tolerance=1,
        min_visible=4,
    ):
        self._window = window
        self._enter = enter
        self._exit = exit
        self._min_visible = min_visible
        size = 2 * tolerance + 1
        self._kernel = np.ones((size, size), dtype=np.uint8)
        self._shape = shape
        self.reset()

    def reset(self):
        """Esquece a janela e esvazia o canvas (folha nova, sessão nova)."""
        self._ink_votes = deque()
        self._vis_votes = deque()
        self._ink_counts = np.zeros(self._shape, dtype=np.int16)
        self._vis_counts = np.zeros(self._shape, dtype=np.int16)
        self.state = np.zeros(self._shape, dtype=bool)
        self._first_seen_ms = np.zeros(self._shape, dtype=np.uint32)
        self._last_update_ms = np.zeros(self._shape, dtype=np.uint32)

    def first_seen_ms(self):
        """Cópia do plano de primeira vista (uint32, ms monotônico)."""
        return self._first_seen_ms.copy()

    def last_update_ms(self):
        """Cópia do plano de última atualização (uint32, ms monotônico)."""
        return self._last_update_ms.copy()

    def update(self, ink_mask, visible_mask=None, t_capture_ms=None):
        """Alimenta uma observação e retorna o canvas (255 = tinta).

        t_capture_ms é o instante de captura do frame, em ms do
        relógio monotônico. Se None, usa o instante de processamento
        (monotônico de agora) e os carimbos passam a medir
        processamento, não captura: o app deve passar o de captura.
        """
        if t_capture_ms is None:
            t_ms = int(time.monotonic() * 1000)
        else:
            t_ms = int(t_capture_ms)
        ink = ink_mask > 0
        if visible_mask is None:
            visible = np.ones(ink.shape, dtype=bool)
        elif visible_mask.dtype == bool:
            visible = visible_mask
        else:
            visible = visible_mask > 0
        evidence = (
            cv2.dilate(ink.astype(np.uint8), self._kernel).astype(bool)
            & visible
        )
        if len(self._ink_votes) == self._window:
            self._ink_counts -= self._ink_votes.popleft()
            self._vis_counts -= self._vis_votes.popleft()
        self._ink_votes.append(evidence)
        self._vis_votes.append(visible)
        self._ink_counts += evidence
        self._vis_counts += visible
        judged = self._vis_counts >= self._min_visible
        fraction = np.zeros(self._shape, dtype=np.float32)
        fraction[judged] = self._ink_counts[judged] / np.maximum(
            self._vis_counts[judged], 1
        )
        enter = judged & ~self.state & (fraction >= self._enter)
        clear = judged & self.state & (fraction <= self._exit)
        self.state[enter] = True
        self.state[clear] = False
        self._update_stamps(evidence, visible, t_ms)
        return (self.state * 255).astype(np.uint8)

    def _update_stamps(self, evidence, visible, t_ms):
        """Mantém first_seen/last_update sobre o estado já atualizado.

        Semântica, só em pixel visível neste frame: pixel de tinta
        sem carimbo (entrou agora, ou entrou coberto e só agora
        ficou visível) ganha first_seen = last_update = t; pixel de
        tinta cuja evidência (a mesma dilatada que alimenta os
        votos) se renova neste frame tem last_update = t; pixel fora
        do canvas zera os dois. Pixel não visível não entra em
        nenhum dos três casos: sob a mão, os carimbos congelam.
        """
        t_val = np.uint32(t_ms & 0xFFFFFFFF)
        ink_visible = self.state & visible
        fresh = ink_visible & (self._first_seen_ms == 0)
        self._first_seen_ms[fresh] = t_val
        self._last_update_ms[fresh] = t_val
        renewed = ink_visible & evidence
        self._last_update_ms[renewed] = t_val
        gone = visible & ~self.state
        self._first_seen_ms[gone] = 0
        self._last_update_ms[gone] = 0
