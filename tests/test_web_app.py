"""Web API: visitor isolation, input handling, and shared Vietnamese composition."""
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from web_app import ClientError, ClientSession, Registry, decode_frame, handler_for


@pytest.fixture
def api():
    registry = Registry(lambda lm: ("O", 0.99), None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(registry))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def call(path, body=b"{}", key="", headers=None):
        req = Request(f"http://127.0.0.1:{server.server_port}" + path,
                      data=body, headers={"X-Session": key, **(headers or {})})
        try:
            with urlopen(req, timeout=5) as result:
                return result.status, json.load(result)
        except HTTPError as exc:
            return exc.code, json.load(exc)
    yield registry, call
    server.shutdown()
    server.server_close()
    thread.join()
    registry.close()


def test_api_isolates_visitors_and_preserves_composer(api):
    registry, call = api
    _, a = call("/api/session")
    _, b = call("/api/session")
    assert a["session"] != b["session"]
    registry.get(a["session"]).core._commit_static("O")
    for action, expected in [("mu", "ô"), ("sac", "ố"), ("backspace", "ô")]:
        status, data = call("/api/action", json.dumps({"action": action}).encode(), a["session"])
        assert status == 200 and data["text"] == expected
    assert registry.get(b["session"]).core.text == ""
    status, data = call("/api/action", b'{"action":"clear"}', a["session"])
    assert status == 200 and data["text"] == ""


def test_api_rejects_invalid_requests_and_closed_sessions(api):
    _, call = api
    assert call("/api/action")[0] == 401
    _, data = call("/api/session")
    key = data["session"]
    assert call("/api/action", b"[]", key)[0] == 400
    assert call("/api/action", b"broken json", key)[0] == 400
    assert call("/api/frame", b"not a jpeg", key)[0] == 400
    assert call("/api/action", b'{"action":"__dict__"}', key)[0] == 400
    assert call("/api/session", headers={"Origin":"https://other.example"})[0] == 403
    assert call("/api/close", key=key)[0] == 200
    assert call("/api/action", key=key)[0] == 401


def test_registry_releases_expired_visitors():
    registry = Registry(lambda lm: ("A", .99), None, capacity=1)
    key, old = registry.create()
    with pytest.raises(ClientError):
        registry.create()
    old.last_seen -= 601
    new_key, _ = registry.create()
    assert new_key != key and old.closed
    with pytest.raises(ClientError):
        registry.get(key)
    registry.close()


def test_pause_resets_tracking_without_losing_text():
    session = ClientSession(lambda lm: ("O", .99), None)
    session.core._commit_static("O")
    session.core.disarmed = True
    session.core.block_until = 999999999
    session.action("pause")
    assert session.core.text == "o"
    assert not session.core.disarmed and session.core.block_until == 0
    assert session.core.stab.candidate is None


def test_frame_decoder_rejects_huge_image_and_accepts_browser_size():
    image = np.zeros((480, 640, 3), np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    assert decode_frame(encoded.tobytes()).shape == image.shape
    _, encoded = cv2.imencode(".jpg", np.zeros((16, 1281, 3), np.uint8))
    with pytest.raises(ClientError):
        decode_frame(encoded.tobytes())


def sample(t, x=0.0):
    lm = np.zeros((21, 3), np.float32)
    lm[:, 0] = x
    lm[9, 1] = -.1
    return {"t": t, "landmarks": lm.tolist()}


def test_browser_samples_hold_by_capture_time_not_network_time(monkeypatch):
    session = ClientSession(lambda lm: ("O", .99), None)
    session.landmarks({"frames": [sample(i/20) for i in range(10)]})
    assert session.core.text == ""
    monkeypatch.setattr("web_app.time.monotonic", lambda: 1000000)
    # Ten seconds of network delay must not count as holding the sign.
    session.landmarks({"frames": [sample(.5)]})
    assert session.core.text == ""
    session.landmarks({"frames": [sample(i/20) for i in range(11, 27)]})
    assert session.core.text == "o"
    assert session.hands is None


def test_browser_tone_survives_delayed_batched_delivery(monkeypatch):
    predictions = []
    def tone(clip):
        predictions.append(clip)
        return "sac", .99
    session = ClientSession(lambda lm: ("O", .99), tone)
    session.core.tokens = ["o"]
    for packet in range(3):
        monkeypatch.setattr("web_app.time.monotonic", lambda p=packet: 100000+p*10)
        data = session.landmarks({"frames": [sample(i/30, i/30*.12) for i in range(packet*20,(packet+1)*20)]})
    assert data["text"] == "ó"
    assert len(predictions) >= 2 and predictions[0].shape == (30,21,3)


def test_browser_rejects_invalid_batch_without_partial_mutation():
    session = ClientSession(lambda lm: ("O", .99), None)
    for frames in ([sample(.1),sample(.1)], [sample(.1),{"t":.2,"landmarks":[[0,0,0]]}],
                   [sample(float("nan"))], [sample(.1)]*61):
        with pytest.raises(ClientError):
            session.landmarks({"frames": frames})
        assert session.last_sample is None
        assert not session.core.hist and session.core.text == ""


def test_actual_capture_gap_resets_hold_and_pause_keeps_text():
    session = ClientSession(lambda lm: ("O", .99), None)
    session.landmarks({"frames": [sample(0)]})
    session.landmarks({"frames": [sample(2)]})
    assert session.core.text == ""
    session.landmarks({"frames": [sample(2+i/20) for i in range(1,27)]})
    assert session.core.text == "o"
    session.action("pause")
    assert session.core.text == "o" and session.source is None


def test_landmark_api_accepts_sequence_and_rejects_out_of_order(api):
    _, call = api
    _, data = call("/api/session")
    key = data["session"]
    code, data = call("/api/landmarks", json.dumps({"frames":[sample(i/20) for i in range(27)]}).encode(), key)
    assert code == 200 and data["text"] == "o" and data["processed_frames"] == 27
    assert call("/api/landmarks", json.dumps({"frames":[sample(.1)]}).encode(),key)[0] == 409
