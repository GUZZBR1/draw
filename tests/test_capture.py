"""Testes da camada de captura, so com mocks, sem camera fisica.

A fabrica de captura e injetada em cada teste: nenhum teste abre
um dispositivo real, le frame de hardware ou depende de /dev/video*.
"""

import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import app as app_module  # noqa: E402
from draw import capture as capture_module  # noqa: E402


class FakeCapture:
    """VideoCapture falso: aberto ou nao, com frames programaveis."""

    def __init__(self, source, backend=None, opened=True, frames=()):
        self.source = source
        self.backend = backend
        self._opened = opened
        self._frames = list(frames)
        self.settings = {}
        self.released = False

    def isOpened(self):
        return self._opened and not self.released

    def set(self, prop, value):
        self.settings[prop] = value
        return True

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True


def test_parse_source_index_path_url_and_file():
    assert capture_module.parse_source(None) == 0
    assert capture_module.parse_source(2) == 2
    assert capture_module.parse_source("1") == 1
    assert capture_module.parse_source(" 2 ") == 2
    assert capture_module.parse_source("/dev/video2") == 2
    assert capture_module.parse_source("http://192.168.0.10:8080/video") == (
        "http://192.168.0.10:8080/video"
    )
    assert capture_module.parse_source("rtsp://camera:554/stream") == (
        "rtsp://camera:554/stream"
    )
    assert capture_module.parse_source("video.mp4") == "video.mp4"
    assert app_module.parse_source("/dev/video1") == 1


def test_backends_order_by_platform():
    linux = capture_module.backends_for_platform("linux")
    assert linux[0] == cv2.CAP_V4L2
    assert cv2.CAP_ANY in linux
    windows = capture_module.backends_for_platform("win32")
    assert windows[0] == cv2.CAP_MSMF
    assert capture_module.backend_from_name("v4l2") == cv2.CAP_V4L2
    assert capture_module.backend_from_name(None) is None
    with pytest.raises(ValueError):
        capture_module.backend_from_name("backend-que-nao-existe")


def test_list_cameras_only_index_1_opens_and_releases():
    seen = []

    def factory(index, backend=None):
        capture = FakeCapture(index, backend, opened=(index == 1))
        seen.append(capture)
        return capture

    results = capture_module.list_cameras(
        max_index=2, capture_factory=factory, nodes=[]
    )
    assert [item["index"] for item in results] == [0, 1, 2]
    assert [item["opened"] for item in results] == [False, True, False]
    assert all(capture.released for capture in seen)
    assert "1" in capture_module.format_camera_list(results)


def test_list_cameras_includes_video_node_above_max_index():
    results = capture_module.list_cameras(
        max_index=0,
        capture_factory=lambda index, backend=None: FakeCapture(index, backend, opened=False),
        nodes=["/dev/video4"],
    )
    assert [item["index"] for item in results] == [0, 4]


def test_open_capture_tries_backends_and_applies_local_settings():
    attempts = []

    def factory(source, backend=None):
        attempts.append(backend)
        return FakeCapture(source, backend, opened=(backend == cv2.CAP_ANY))

    capture = capture_module.open_capture(
        0, capture_factory=factory, backends=[cv2.CAP_V4L2, cv2.CAP_ANY]
    )
    assert capture.isOpened()
    assert attempts == [cv2.CAP_V4L2, cv2.CAP_ANY]
    assert capture.settings[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert capture.settings[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert capture.settings[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC] == 3000


def test_open_capture_custom_size_and_specific_backend():
    def factory(source, backend=None):
        return FakeCapture(source, backend, opened=True)

    capture = capture_module.open_capture(
        1, width=640, height=480, fps=30, backend=cv2.CAP_V4L2,
        capture_factory=factory,
    )
    assert capture.settings[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert capture.settings[cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert capture.settings[cv2.CAP_PROP_FPS] == 30
    assert capture.backend == cv2.CAP_V4L2


def test_open_capture_url_does_not_apply_local_settings():
    def factory(source, backend=None):
        return FakeCapture(source, backend, opened=True)

    capture = capture_module.open_capture(
        "http://192.168.0.10:8080/video", capture_factory=factory
    )
    assert capture.isOpened()
    assert cv2.CAP_PROP_FRAME_WIDTH not in capture.settings


def test_open_capture_local_missing_node_reports_not_found(monkeypatch):
    monkeypatch.setattr(capture_module.sys, "platform", "linux")
    monkeypatch.setattr(capture_module, "is_wsl", lambda: False)
    with pytest.raises(capture_module.CaptureError) as excinfo:
        capture_module.open_capture(
            7,
            capture_factory=lambda source, backend=None: FakeCapture(
                source, backend, opened=False
            ),
        )
    assert excinfo.value.kind == "not_found"
    assert "/dev/video7" in str(excinfo.value)


def test_open_capture_network_error_kind():
    with pytest.raises(capture_module.CaptureError) as excinfo:
        capture_module.open_capture(
            "http://192.168.0.10:8080/video",
            capture_factory=lambda source, backend=None: FakeCapture(
                source, backend, opened=False
            ),
        )
    assert excinfo.value.kind == "network"


def test_diagnose_local_permission_and_busy(monkeypatch):
    monkeypatch.setattr(capture_module.sys, "platform", "linux")
    kind, hint = capture_module.diagnose_local(
        0,
        exists=lambda path: True,
        access=lambda path, mode: False,
        wsl=False,
    )
    assert kind == "permission"
    assert "video" in hint
    kind, hint = capture_module.diagnose_local(
        0,
        exists=lambda path: True,
        access=lambda path, mode: True,
        wsl=False,
    )
    assert kind == "busy"
    assert "Outro programa" in hint


def test_reconnect_succeeds_on_second_attempt_and_fails_cleanly():
    calls = []

    def open_ok(source, **kwargs):
        calls.append(source)
        if len(calls) == 1:
            raise capture_module.CaptureError(source, "busy", "ocupada")
        return FakeCapture(source, opened=True)

    reconnect = capture_module.ReconnectingCapture(
        0, open_func=open_ok, attempts=2, delay_s=0, sleep=lambda seconds: None
    )
    assert reconnect.try_reconnect() is True
    assert reconnect.reconnects == 1
    assert reconnect.isOpened()

    def open_never(source, **kwargs):
        raise capture_module.CaptureError(source, "not_found", "sumiu")

    never = capture_module.ReconnectingCapture(
        0, open_func=open_never, attempts=2, delay_s=0, sleep=lambda seconds: None
    )
    assert never.try_reconnect() is False
    assert never.last_error.kind == "not_found"

    off = capture_module.ReconnectingCapture(
        0, open_func=open_never, attempts=0, delay_s=0, sleep=lambda seconds: None
    )
    assert off.try_reconnect() is False


def test_app_list_cameras_cli_exits_with_status(monkeypatch, capsys):
    monkeypatch.setattr(
        capture_module,
        "list_cameras",
        lambda: [
            {"index": 0, "node": "/dev/video0", "opened": False},
            {"index": 1, "node": "/dev/video1", "opened": True},
        ],
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--list-cameras"])
    with pytest.raises(SystemExit) as excinfo:
        app_module.main()
    assert excinfo.value.code == 0
    assert "1 (/dev/video1): ok" in capsys.readouterr().out
