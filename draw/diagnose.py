"""Relatorio de diagnostico por fonte de camera, sem acao no sistema.

Este modulo so observa: abre a fonte, le alguns quadros e solta a
captura. Ele nao anexa USB, nao muda grupo, nao liga servico e nao
re-escreve o `python main.py --check`. O --check continua sendo o
diagnostico porta a porta do upstream; aqui o foco e outro: para
cada fonte, dizer se abriu, com qual backend, com quais props
negociadas, quantos quadros chegaram e, se falhou, qual e o tipo
do problema e o que fazer. URLs so sao testadas quando passadas
explicitamente; probe_all() so testa fontes locais vindas de
list_cameras().
"""

import time
from dataclasses import asdict, dataclass

import cv2

from . import capture as capture_module

_DEFAULT_DEADLINE_S = 5.0
_READ_SLEEP_S = 0.01

_BACKEND_ID_TO_NAME = {
    getattr(cv2, attr): attr
    for attr in capture_module._BACKEND_NAMES.values()
    if hasattr(cv2, attr)
}


@dataclass
class SourceReport:
    """Resultado de uma sonda em uma fonte de camera."""

    source: object
    kind: str
    opened: bool
    backend_used: object
    width: object
    height: object
    fps: object
    fourcc: object
    frames_ok: int
    latency_ms: object
    error_kind: object
    hint: str

    @property
    def ok(self):
        """Ok quando abriu e entregou os quadros pedidos."""
        return self.opened and self.error_kind is None

    def to_dict(self):
        return asdict(self)


def _source_kind(parsed):
    if isinstance(parsed, int):
        return "local"
    if isinstance(parsed, str) and parsed.startswith(
        ("http://", "https://", "rtsp://")
    ):
        return "network"
    return "file"


def _get_prop(capture, prop):
    try:
        value = capture.get(prop)
    except Exception:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number <= 0:
        return None
    return number


def _fourcc_str(value):
    if not value:
        return None
    try:
        raw = int(value).to_bytes(4, "little")
    except (OverflowError, ValueError):
        return None
    text = raw.decode("ascii", "ignore").strip("\x00 ").strip()
    if not text or not all(32 <= ord(char) < 127 for char in text):
        return None
    return text


def probe_source(source, read_frames=3, deadline=_DEFAULT_DEADLINE_S,
                 open_timeout_ms=None, backend=None, backends=None,
                 capture_factory=None, monotonic=None, sleep=None):
    """Sonda uma fonte e devolve um SourceReport, sem travar.

    deadline e o teto, em segundos de relogio de parede, para a
    leitura de quadros depois da abertura. Ele limita o loop de
    leitura; uma unica chamada read() bloqueante de um backend nao
    pode ser interrompida por ele, por isso a abertura tambem usa
    open_timeout_ms. A fabrica e injetavel para testes com mocks.
    """
    clock = monotonic or time.monotonic
    sleeper = sleep or time.sleep
    parsed = capture_module.parse_source(source)
    kind = _source_kind(parsed)
    start = clock()
    attempted = []

    def recording_factory(parsed_source, candidate=None, params=None):
        attempted.append(candidate)
        factory = capture_factory or cv2.VideoCapture
        # Ponte de integracao com a main: open_capture() da main chama
        # a fabrica com (fonte, backend, params de timeout). Fabricas
        # antigas (incluindo as de teste do senado) aceitam so 2
        # argumentos; nesse caso, tenta sem params em vez de deixar
        # a sonda de rede falhar com TypeError. Fabricas novas e o
        # cv2.VideoCapture real recebem os 3 argumentos.
        if params is None:
            return factory(parsed_source, candidate)
        try:
            return factory(parsed_source, candidate, params)
        except TypeError:
            return factory(parsed_source, candidate)

    open_kwargs = {}
    if open_timeout_ms is not None:
        open_kwargs["open_timeout_ms"] = open_timeout_ms
    if backend is not None:
        open_kwargs["backend"] = backend
    if backends is not None:
        open_kwargs["backends"] = backends

    try:
        capture = capture_module.open_capture(
            source, capture_factory=recording_factory, **open_kwargs
        )
    except capture_module.CaptureError as error:
        return SourceReport(
            source=source,
            kind=kind,
            opened=False,
            backend_used=None,
            width=None,
            height=None,
            fps=None,
            fourcc=None,
            frames_ok=0,
            latency_ms=None,
            error_kind=error.kind,
            hint=str(error),
        )
    except Exception as error:  # pragma: no cover - defesa, nao esperado
        return SourceReport(
            source=source,
            kind=kind,
            opened=False,
            backend_used=None,
            width=None,
            height=None,
            fps=None,
            fourcc=None,
            frames_ok=0,
            latency_ms=None,
            error_kind="busy",
            hint=(
                "Falha inesperada ao abrir a fonte. "
                f"Detalhe: {error}. Feche outros programas usando a "
                "camera e tente de novo."
            ),
        )

    backend_used = None
    if attempted:
        backend_used = _BACKEND_ID_TO_NAME.get(
            attempted[-1], attempted[-1]
        )
    try:
        width_value = _get_prop(capture, cv2.CAP_PROP_FRAME_WIDTH)
        height_value = _get_prop(capture, cv2.CAP_PROP_FRAME_HEIGHT)
        fps_value = _get_prop(capture, cv2.CAP_PROP_FPS)
        fourcc_value = _get_prop(capture, cv2.CAP_PROP_FOURCC)
        frames_ok = 0
        first_frame_at = None
        while frames_ok < read_frames:
            if clock() - start > deadline:
                break
            ok, frame = capture.read()
            if ok and frame is not None:
                if first_frame_at is None:
                    first_frame_at = clock()
                frames_ok += 1
            else:
                if clock() - start > deadline:
                    break
                sleeper(_READ_SLEEP_S)
        latency_ms = None
        if first_frame_at is not None:
            latency_ms = int((first_frame_at - start) * 1000)
        if frames_ok >= read_frames:
            error_kind = None
            hint = "Fonte abriu e entregou os quadros pedidos."
        elif frames_ok > 0:
            error_kind = "no_frames"
            hint = (
                "A fonte abriu, mas entregou menos quadros do que o "
                "pedido antes do prazo. A camera pode estar ocupada, "
                "com cabo solto ou travada; feche outros programas "
                "usando a camera, reconecte-a e tente de novo."
            )
        else:
            error_kind = "no_frames"
            hint = (
                "A fonte abriu, mas nenhum quadro chegou antes do "
                "prazo. A camera pode estar ocupada, com cabo solto "
                "ou travada; feche outros programas usando a camera, "
                "reconecte-a e tente de novo."
            )
        return SourceReport(
            source=source,
            kind=kind,
            opened=True,
            backend_used=backend_used,
            width=int(width_value) if width_value else None,
            height=int(height_value) if height_value else None,
            fps=float(fps_value) if fps_value else None,
            fourcc=_fourcc_str(fourcc_value),
            frames_ok=frames_ok,
            latency_ms=latency_ms,
            error_kind=error_kind,
            hint=hint,
        )
    finally:
        try:
            capture.release()
        except Exception:
            pass


def probe_all(read_frames=3, deadline=_DEFAULT_DEADLINE_S,
              include_urls=(), capture_factory=None, list_fn=None,
              open_timeout_ms=None):
    """Sonda as webcams locais abertas em list_cameras().

    URLs nunca entram sozinhas: so sao sondadas quando passadas em
    include_urls. Fontes locais que list_cameras() marcou como
    fechadas nao sao sondadas de novo aqui.
    """
    if list_fn is None:
        if capture_factory is not None:
            results = capture_module.list_cameras(
                capture_factory=capture_factory
            )
        else:
            results = capture_module.list_cameras()
    else:
        results = list_fn()
    reports = []
    for item in results:
        if not item.get("opened"):
            continue
        reports.append(probe_source(
            item["index"],
            read_frames=read_frames,
            deadline=deadline,
            capture_factory=capture_factory,
            open_timeout_ms=open_timeout_ms,
        ))
    for url in include_urls or ():
        reports.append(probe_source(
            url,
            read_frames=read_frames,
            deadline=deadline,
            capture_factory=capture_factory,
            open_timeout_ms=open_timeout_ms,
        ))
    return reports


def format_report(report):
    """Uma linha de texto para um relatorio, para o terminal."""
    if report.opened and report.error_kind is None:
        props = []
        if report.width and report.height:
            props.append(f"{report.width}x{report.height}")
        if report.fps:
            props.append(f"@{report.fps}fps")
        if report.fourcc:
            props.append(str(report.fourcc))
        if report.backend_used:
            props.append(str(report.backend_used))
        detail = f" ({', '.join(props)})" if props else ""
        latency = (
            f", {report.latency_ms}ms ate o primeiro quadro"
            if report.latency_ms is not None else ""
        )
        return (
            f"OK {report.source}: abriu{detail}, "
            f"{report.frames_ok} quadros{latency}."
        )
    reason = report.error_kind or "falha"
    return f"FALHOU {report.source} ({reason}): {report.hint}"
