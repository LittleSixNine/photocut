import base64
import contextlib
import io
import json
import socket
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from photocut.confirmation.controller import (
    ConfirmationAction,
    ConfirmationEntryController,
    ConfirmationSessionController,
    ConfirmationSessionError,
    LoadedConfirmationItem,
)
from photocut.confirmation.web.server import LocalConfirmationServer, WriterLease


SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' blob:; connect-src 'self'; "
        "object-src 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
OPEN_RESPONSES = []


class FakeSession:
    def __init__(self):
        self.actions = []
        self.details_calls = 0
        self.wait_calls = 0
        self.results = {}

    def snapshot(self):
        return {
            "revision": len(self.actions),
            "session": {"status": "active", "readonly": False},
        }

    def technical_details(self):
        self.details_calls += 1
        return {"path": "/current/only.jpg", "candidate_audit": []}

    def dispatch(self, action):
        if not isinstance(action, ConfirmationAction):
            raise AssertionError("server must parse actions into ConfirmationAction")
        previous = self.results.get(action.action_id)
        if previous is not None:
            return previous
        self.actions.append(action)
        result = self.snapshot()
        self.results[action.action_id] = result
        return result

    def wait_until_stopped(self):
        self.wait_calls += 1


def fake_session():
    return FakeSession()


def request(url, method="GET", body=None, headers=None, raw_body=None):
    if raw_body is not None:
        value = raw_body
    else:
        value = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=value, method=method, headers=headers or {}
    )
    try:
        response = urllib.request.urlopen(req, timeout=2)
    except urllib.error.HTTPError as exc:
        response = exc
    OPEN_RESPONSES.append(response)
    return response


def post_response(server, suffix, body, headers=None, raw_body=None):
    request_headers = {
        "Content-Type": "application/json",
        "Origin": server.origin,
    }
    request_headers.update(headers or {})
    return request(
        server.url + "api/" + suffix.removeprefix("api/"),
        method="POST",
        body=body,
        headers=request_headers,
        raw_body=raw_body,
    )


def post_json(server, suffix, body, headers=None):
    response = post_response(server, suffix, body, headers=headers)
    return response, json.loads(response.read())


def action_body(client_id="client-a", action_id="action-1"):
    return {
        "client_id": client_id,
        "action_id": action_id,
        "expected_revision": 0,
        "kind": "reset",
        "payload": {},
    }


def raw_request(server, payload):
    connection = socket.create_connection(server.httpd.server_address, timeout=2)
    try:
        connection.settimeout(2)
        connection.sendall(payload)
        connection.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        connection.close()


def assert_raw_response(testcase, response, status):
    head, separator, body = response.partition(b"\r\n\r\n")
    testcase.assertEqual(b"\r\n\r\n", separator)
    lines = head.decode("iso-8859-1").split("\r\n")
    testcase.assertEqual(f"HTTP/1.0 {status}", " ".join(lines[0].split()[:2]))
    headers = {}
    for line in lines[1:]:
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    for name, value in SECURITY_HEADERS.items():
        testcase.assertEqual(value, headers.get(name.lower()), name)
    testcase.assertNotIn("access-control-allow-origin", headers)
    return body


class MemoryControllerBackend:
    def __init__(self):
        entry = {
            "filename": "real.jpg",
            "detector": "v5.2",
            "detector_used": "v5.2",
            "corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
            "boundary_corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
            "confirmed": False,
        }
        self.item = LoadedConfirmationItem(
            entry=entry,
            image=np.zeros((80, 100, 3), dtype=np.uint8),
            image_token="real-image",
            display_path="/fixtures/real.jpg",
            editor=ConfirmationEntryController(entry, (100, 80)),
            finalized_evidence=None,
            previous_annotation=None,
            started_at=0.0,
            is_revision=False,
        )

    def count(self):
        return 1

    def load(self, index):
        if index != 0:
            raise IndexError(index)
        return self.item

    def load_preview(self, index):
        raise IndexError(index)

    def checkpoint(self, item):
        return None

    def commit(self, item, duration_ms, gui_version):
        return {"gui_version": gui_version}

    def skip(self, item, reason):
        return None


class WriterLeaseTests(unittest.TestCase):
    def test_lease_uses_monotonic_ttl_and_does_not_transfer_before_expiry(self):
        now = [10.0]
        lease = WriterLease(ttl=15.0, clock=lambda: now[0])

        self.assertTrue(lease.acquire("client-a"))
        now[0] = 24.999
        self.assertFalse(lease.acquire("client-b"))
        self.assertTrue(lease.is_writer("client-a"))

        now[0] = 25.0
        self.assertTrue(lease.acquire("client-b"))
        self.assertFalse(lease.is_writer("client-a"))

    def test_same_client_heartbeat_renews_lease(self):
        now = [1.0]
        lease = WriterLease(ttl=15.0, clock=lambda: now[0])
        self.assertTrue(lease.acquire("client-a"))

        now[0] = 10.0
        self.assertTrue(lease.heartbeat("client-a"))
        now[0] = 16.1

        self.assertTrue(lease.is_writer("client-a"))
        self.assertFalse(lease.heartbeat("client-b"))

    def test_writer_check_and_dispatch_are_one_fenced_operation(self):
        now = [0.0]
        lease = WriterLease(ttl=15.0, clock=lambda: now[0])
        self.assertTrue(lease.acquire("client-a"))

        calls = []
        self.assertEqual(
            "done",
            lease.execute_if_writer("client-a", lambda: calls.append("a") or "done"),
        )
        now[0] = 15.0
        self.assertTrue(lease.acquire("client-b"))
        with self.assertRaisesRegex(RuntimeError, "writer"):
            lease.execute_if_writer("client-a", lambda: calls.append("stale"))

        self.assertEqual(["a"], calls)


class LocalConfirmationServerTests(unittest.TestCase):
    def tearDown(self):
        while OPEN_RESPONSES:
            OPEN_RESPONSES.pop().close()

    def assert_security_headers(self, response):
        for name, value in SECURITY_HEADERS.items():
            self.assertEqual(value, response.headers.get(name), name)
        self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

    def acquire(self, server, client_id="client-a"):
        response, payload = post_json(
            server, "lease", {"client_id": client_id}
        )
        self.assertEqual(200, response.status)
        self.assert_security_headers(response)
        return payload

    def test_response_ignores_client_disconnect_during_body_write(self):
        class DisconnectedWriter:
            def write(self, payload):
                raise BrokenPipeError("browser cancelled request")

        handler_type = LocalConfirmationServer._handler_type(None)
        handler = object.__new__(handler_type)
        handler.command = "GET"
        handler.close_connection = False
        handler.wfile = DisconnectedWriter()
        handler.send_response = lambda status: None
        handler.send_header = lambda name, value: None
        handler.end_headers = lambda: None

        handler._response(200, b"payload")

        self.assertTrue(handler.close_connection)

    def test_server_binds_loopback_and_requires_32_byte_session_token(self):
        with LocalConfirmationServer(fake_session()) as server:
            self.assertEqual("127.0.0.1", server.httpd.server_address[0])
            self.assertRegex(
                server.url,
                r"^http://127\.0\.0\.1:\d+/session/[A-Za-z0-9_-]+/$",
            )
            decoded = base64.urlsafe_b64decode(
                server.token + "=" * (-len(server.token) % 4)
            )
            self.assertEqual(32, len(decoded))

            wrong = request(server.origin + "/session/wrong/api/state?client_id=x")
            missing = request(server.origin + "/api/state?client_id=x")
            self.assertEqual(404, wrong.status)
            self.assertEqual(404, missing.status)
            self.assertEqual(b"", wrong.read())
            self.assertEqual(b"", missing.read())
            self.assert_security_headers(wrong)
            self.assert_security_headers(missing)

    def test_only_first_live_client_gets_writer_lease_and_refresh_keeps_it(self):
        with LocalConfirmationServer(fake_session()) as server:
            first = self.acquire(server, "client-a")
            second = self.acquire(server, "client-b")
            refresh = self.acquire(server, "client-a")
            _, heartbeat = post_json(
                server, "heartbeat", {"client_id": "client-a"}
            )

            self.assertTrue(first["writer"])
            self.assertFalse(second["writer"])
            self.assertTrue(refresh["writer"])
            self.assertTrue(heartbeat["writer"])

    def test_state_is_readonly_for_non_writer_and_validates_client_id(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            self.acquire(server, "client-a")
            writer = request(server.url + "api/state?client_id=client-a")
            reader = request(server.url + "api/state?client_id=client-b")
            invalid = request(
                server.url
                + "api/state?client_id="
                + urllib.parse.quote("../private/file", safe="")
            )
            too_long = request(
                server.url + "api/state?client_id=" + "a" * 129
            )

            self.assertFalse(json.loads(writer.read())["session"]["readonly"])
            self.assertTrue(json.loads(reader.read())["session"]["readonly"])
            self.assertEqual(400, invalid.status)
            self.assertEqual(400, too_long.status)

    def test_details_requires_bounded_client_id_and_calls_current_session_only(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            missing = request(server.url + "api/details")
            duplicate = request(
                server.url + "api/details?client_id=a&client_id=b"
            )
            valid = request(server.url + "api/details?client_id=client-a")

            self.assertEqual(400, missing.status)
            self.assertEqual(400, duplicate.status)
            self.assertEqual("/current/only.jpg", json.loads(valid.read())["path"])
            self.assertEqual(1, session.details_calls)

    def test_post_requires_exact_origin_json_object_and_known_fields(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            wrong_origin = post_response(
                server,
                "lease",
                {"client_id": "client-a"},
                headers={"Origin": server.origin + "/"},
            )
            wrong_type = post_response(
                server,
                "lease",
                {"client_id": "client-a"},
                headers={"Content-Type": "text/plain"},
            )
            array_body = post_response(server, "lease", ["client-a"])
            extra = post_response(
                server,
                "lease",
                {"client_id": "client-a", "path": "/tmp/private.jpg"},
            )
            malformed = post_response(
                server, "lease", None, raw_body=b"{not-json"
            )

            for response in (
                wrong_origin,
                wrong_type,
                array_body,
                extra,
                malformed,
            ):
                with self.subTest(status=response.status):
                    self.assertEqual(400, response.status)
                    self.assert_security_headers(response)
            self.assertEqual([], session.actions)

    def test_post_rejects_body_larger_than_one_mib_without_dispatch(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            response = post_response(
                server,
                "action",
                None,
                headers={"Content-Length": str(1024 * 1024 + 1)},
                raw_body=b"",
            )

            self.assertEqual(400, response.status)
            self.assertEqual([], session.actions)

    def test_action_requires_writer_and_is_parsed_as_confirmation_action(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            self.acquire(server, "client-a")
            rejected, rejected_body = post_json(
                server, "action", action_body("client-b")
            )
            accepted, accepted_body = post_json(
                server, "action", action_body("client-a")
            )

            self.assertEqual(409, rejected.status)
            self.assertEqual("writer_required", rejected_body["error"]["code"])
            self.assertEqual(200, accepted.status)
            self.assertEqual(1, accepted_body["revision"])
            self.assertEqual(1, len(session.actions))
            action = session.actions[0]
            self.assertIsInstance(action, ConfirmationAction)
            self.assertEqual("action-1", action.action_id)
            self.assertEqual("reset", action.kind)

    def test_duplicate_action_id_returns_identical_json_without_second_dispatch_effect(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            self.acquire(server)
            first, first_payload = post_json(server, "action", action_body())
            second, second_payload = post_json(server, "action", action_body())

            self.assertEqual(200, first.status)
            self.assertEqual(200, second.status)
            self.assertEqual(first_payload, second_payload)
            self.assertEqual(1, len(session.actions))

    def test_http_replay_and_conflict_use_real_session_controller(self):
        session = ConfirmationSessionController(
            MemoryControllerBackend(), gui_version="2.0"
        )
        with LocalConfirmationServer(session) as server:
            self.acquire(server)
            body = {
                "client_id": "client-a",
                "action_id": "real-action",
                "expected_revision": 0,
                "kind": "select_corner",
                "payload": {"index": 0},
            }
            first, first_payload = post_json(server, "action", body)
            replay, replay_payload = post_json(server, "action", body)
            conflict_body = dict(body)
            conflict_body.update(
                expected_revision=1, kind="reset", payload={}
            )
            conflict, conflict_payload = post_json(
                server, "action", conflict_body
            )

            self.assertEqual(200, first.status)
            self.assertEqual(200, replay.status)
            self.assertEqual(first_payload, replay_payload)
            self.assertEqual(1, session.revision)
            self.assertEqual(409, conflict.status)
            self.assertEqual(
                "action_id_conflict", conflict_payload["error"]["code"]
            )
            self.assertEqual(1, session.revision)

    def test_expired_queued_writer_is_fenced_before_canonical_dispatch(self):
        now = [0.0]
        old_waiting = threading.Event()
        release_old = threading.Event()

        class PausingLease(WriterLease):
            def _pause_old(self, client_id):
                if client_id == "client-a":
                    old_waiting.set()
                    self.assert_release()

            def assert_release(self):
                if not release_old.wait(2):
                    raise AssertionError("old writer release timed out")

            def is_writer(self, client_id):
                result = super().is_writer(client_id)
                if result:
                    self._pause_old(client_id)
                return result

            def execute_if_writer(self, client_id, callback):
                self._pause_old(client_id)
                return super().execute_if_writer(client_id, callback)

        session = fake_session()
        server = LocalConfirmationServer(session)
        server.writer_lease = PausingLease(clock=lambda: now[0])
        server.start()
        try:
            self.acquire(server, "client-a")
            old_result = {}

            def send_old_action():
                old_result["response"], old_result["body"] = post_json(
                    server, "action", action_body("client-a", "old-action")
                )

            old_thread = threading.Thread(target=send_old_action)
            old_thread.start()
            self.assertTrue(old_waiting.wait(2))

            now[0] = 15.0
            self.assertTrue(self.acquire(server, "client-b")["writer"])
            release_old.set()
            old_thread.join(2)

            self.assertFalse(old_thread.is_alive())
            self.assertEqual(409, old_result["response"].status)
            self.assertEqual("writer_required", old_result["body"]["error"]["code"])
            self.assertEqual([], session.actions)

            current, current_body = post_json(
                server, "action", action_body("client-b", "current-action")
            )
            self.assertEqual(200, current.status)
            self.assertEqual(1, current_body["revision"])
            self.assertEqual(1, len(session.actions))
        finally:
            release_old.set()
            server.close()

    def test_writer_transfer_waits_for_inflight_canonical_dispatch(self):
        now = [0.0]
        dispatch_started = threading.Event()
        release_dispatch = threading.Event()
        transfer_attempted = threading.Event()

        class BlockingSession(FakeSession):
            def dispatch(self, action):
                dispatch_started.set()
                if not release_dispatch.wait(2):
                    raise AssertionError("dispatch release timed out")
                return super().dispatch(action)

        class ObservedLease(WriterLease):
            def acquire(self, client_id):
                if client_id == "client-b":
                    transfer_attempted.set()
                return super().acquire(client_id)

        session = BlockingSession()
        server = LocalConfirmationServer(session)
        server.writer_lease = ObservedLease(clock=lambda: now[0])
        server.start()
        try:
            self.acquire(server, "client-a")
            action_result = {}
            transfer_result = {}

            def send_action():
                action_result["response"], action_result["body"] = post_json(
                    server, "action", action_body("client-a", "inflight")
                )

            def transfer_writer():
                transfer_result["lease"] = self.acquire(server, "client-b")

            action_thread = threading.Thread(target=send_action)
            action_thread.start()
            self.assertTrue(dispatch_started.wait(2))
            now[0] = 15.0
            transfer_thread = threading.Thread(target=transfer_writer)
            transfer_thread.start()
            self.assertTrue(transfer_attempted.wait(2))
            transfer_thread.join(0.1)
            self.assertTrue(transfer_thread.is_alive())

            release_dispatch.set()
            action_thread.join(2)
            transfer_thread.join(2)

            self.assertFalse(action_thread.is_alive())
            self.assertFalse(transfer_thread.is_alive())
            self.assertEqual(200, action_result["response"].status)
            self.assertTrue(transfer_result["lease"]["writer"])
            next_body = action_body("client-b", "after-transfer")
            next_body["expected_revision"] = 1
            next_response, _ = post_json(server, "action", next_body)
            self.assertEqual(200, next_response.status)
            self.assertEqual(2, len(session.actions))
        finally:
            release_dispatch.set()
            server.close()

    def test_session_errors_are_409_and_keep_structured_scope(self):
        class RejectedSession(FakeSession):
            def dispatch(self, action):
                raise ConfirmationSessionError(
                    "stale_revision", "stale confirmation revision", "action"
                )

        with LocalConfirmationServer(RejectedSession()) as server:
            self.acquire(server)
            response, payload = post_json(server, "action", action_body())

            self.assertEqual(409, response.status)
            self.assertEqual(
                {
                    "code": "stale_revision",
                    "message": "stale confirmation revision",
                    "scope": "action",
                },
                payload["error"],
            )

    def test_unexpected_errors_are_generic_500_without_path_or_traceback(self):
        class BrokenSession(FakeSession):
            def snapshot(self):
                raise RuntimeError("failed at /private/path/secret.jpg")

        with LocalConfirmationServer(BrokenSession()) as server:
            with self.assertLogs("photocut.confirmation.web.server", level="ERROR"):
                response = request(server.url + "api/state?client_id=client-a")
            raw = response.read().decode("utf-8")

            self.assertEqual(500, response.status)
            self.assertNotIn("/private/path", raw)
            self.assertNotIn("Traceback", raw)
            self.assertEqual(
                {"error": {"code": "internal_error"}}, json.loads(raw)
            )
            self.assert_security_headers(response)

    def test_unknown_routes_methods_and_wrong_tokens_have_no_details_or_request_log(self):
        with LocalConfirmationServer(fake_session()) as server:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                unknown = request(server.url + "api/not-a-route")
                options = request(server.url + "api/state", method="OPTIONS")
                trace = request(server.url + "api/state", method="TRACE")
                wrong = request(
                    server.origin
                    + "/session/"
                    + server.token
                    + "-wrong/api/state?client_id=a"
                )

            for response in (unknown, options, trace, wrong):
                self.assertEqual(404, response.status)
                self.assertEqual(b"", response.read())
                self.assert_security_headers(response)
            self.assertNotIn(server.token, stderr.getvalue())

    def test_parser_errors_are_http10_400_with_security_headers(self):
        with LocalConfirmationServer(fake_session()) as server:
            target = f"/session/{server.token}/api/state?client_id=client-a"
            requests = (
                f"GET {target}\r\n\r\n".encode("ascii"),
                f"GET {target} HTTP/2.0\r\nHost: 127.0.0.1\r\n\r\n".encode(
                    "ascii"
                ),
                (
                    f"GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    + "".join(f"X-{index}: x\r\n" for index in range(101))
                    + "\r\n"
                ).encode("ascii"),
            )

            for payload in requests:
                with self.subTest(request_line=payload.split(b"\r\n", 1)[0]):
                    response = raw_request(server, payload)
                    body = assert_raw_response(self, response, 400)
                    self.assertEqual(
                        {"error": {"code": "invalid_request"}},
                        json.loads(body),
                    )

    def test_raw_unsupported_method_is_detail_free_404_with_security_headers(self):
        with LocalConfirmationServer(fake_session()) as server:
            target = f"/session/{server.token}/api/state?client_id=client-a"
            response = raw_request(
                server,
                f"BREW {target} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".encode(
                    "ascii"
                ),
            )

            body = assert_raw_response(self, response, 404)
            self.assertEqual(b"", body)

    def test_request_target_must_be_canonical_origin_form(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            canonical = f"/session/{server.token}/api/state?client_id=client-a"
            noncanonical = (
                f"{server.origin}{canonical}",
                canonical + "#fragment",
                "//" + canonical.lstrip("/"),
                "http://[::1",
            )
            for target in noncanonical:
                with self.subTest(target=target):
                    response = raw_request(
                        server,
                        (
                            f"GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                            "Connection: close\r\n\r\n"
                        ).encode("ascii"),
                    )
                    self.assertEqual(
                        b"", assert_raw_response(self, response, 404)
                    )

            accepted = raw_request(
                server,
                f"GET {canonical} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".encode(
                    "ascii"
                ),
            )
            accepted_body = assert_raw_response(self, accepted, 200)
            self.assertEqual(0, json.loads(accepted_body)["revision"])

    def test_post_target_rejects_any_query_marker_before_reading_body(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            target = f"/session/{server.token}/api/lease?"
            payload = b'{"client_id":"client-a"}'
            response = raw_request(
                server,
                (
                    f"POST {target} HTTP/1.1\r\n"
                    f"Host: 127.0.0.1\r\nOrigin: {server.origin}\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                ).encode("ascii")
                + payload,
            )

            self.assertEqual(b"", assert_raw_response(self, response, 404))
            self.assertFalse(server.writer_lease.is_writer("client-a"))

    def test_raw_duplicate_headers_and_chunked_requests_are_400(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            target = f"/session/{server.token}/api/lease"
            payload = b'{"client_id":"client-a"}'
            variants = (
                (
                    f"Origin: {server.origin}\r\nOrigin: {server.origin}\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\n"
                ),
                (
                    f"Origin: {server.origin}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\nContent-Length: {len(payload)}\r\n"
                ),
                (
                    f"Origin: {server.origin}\r\nContent-Type: application/json\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\n"
                ),
                (
                    f"Origin: {server.origin}\r\nContent-Type: application/json\r\n"
                    "Transfer-Encoding: chunked\r\n"
                ),
            )
            for headers in variants:
                with self.subTest(headers=headers):
                    response = raw_request(
                        server,
                        (
                            f"POST {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                            + headers
                            + "Connection: close\r\n\r\n"
                        ).encode("ascii")
                        + payload,
                    )
                    assert_raw_response(self, response, 400)
            self.assertFalse(server.writer_lease.is_writer("client-a"))

    def test_nonfinite_json_constants_never_reach_session_dispatch(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            self.acquire(server)
            target = f"/session/{server.token}/api/action"
            for constant in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(constant=constant):
                    payload = (
                        "{"
                        '"client_id":"client-a",'
                        f'"action_id":"constant-{constant}",'
                        '"expected_revision":0,"kind":"move",'
                        f'"payload":{{"dx":{constant},"dy":0}}'
                        "}"
                    ).encode("ascii")
                    response = raw_request(
                        server,
                        (
                            f"POST {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                            f"Origin: {server.origin}\r\nContent-Type: application/json\r\n"
                            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                        ).encode("ascii")
                        + payload,
                    )
                    assert_raw_response(self, response, 400)
            self.assertEqual([], session.actions)

    def test_content_length_and_json_are_strict(self):
        session = fake_session()
        with LocalConfirmationServer(session) as server:
            target = f"/session/{server.token}/api/lease"
            duplicate_key = b'{"client_id":"client-a","client_id":"x"}'
            nonfinite = b'{"client_id":"client-a","value":NaN}'
            invalid_bodies = (
                ("+24", b'{"client_id":"client-a"}'),
                ("2_4", b'{"client_id":"client-a"}'),
                ("2 4", b'{"client_id":"client-a"}'),
                ("٢٤", b'{"client_id":"client-a"}'),
                (str(len(duplicate_key)), duplicate_key),
                (str(len(nonfinite)), nonfinite),
            )
            for content_length, payload in invalid_bodies:
                with self.subTest(content_length=content_length, payload=payload):
                    response = raw_request(
                        server,
                        (
                            f"POST {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                            f"Origin: {server.origin}\r\nContent-Type: application/json\r\n"
                            f"Content-Length: {content_length}\r\nConnection: close\r\n\r\n"
                        ).encode("utf-8")
                        + payload,
                    )
                    assert_raw_response(self, response, 400)
            self.assertFalse(server.writer_lease.is_writer("client-a"))

    def test_close_is_idempotent_stops_thread_and_releases_port(self):
        server = LocalConfirmationServer(fake_session())
        server.close()
        server.close()

        running = LocalConfirmationServer(fake_session())
        url = running.start()
        self.assertEqual(url, running.start())
        host, port = running.httpd.server_address
        thread = running.thread
        running.close()
        running.close()

        self.assertFalse(thread.is_alive())
        rebound = ThreadingHTTPServer((host, port), BaseHTTPRequestHandler)
        try:
            self.assertEqual((host, port), rebound.server_address)
        finally:
            rebound.server_close()

    def test_concurrent_close_waits_for_active_handler_quiescence(self):
        dispatch_started = threading.Event()
        release_dispatch = threading.Event()

        class BlockingSession(FakeSession):
            def dispatch(self, action):
                dispatch_started.set()
                if not release_dispatch.wait(2):
                    raise AssertionError("dispatch release timed out")
                return super().dispatch(action)

        session = BlockingSession()
        server = LocalConfirmationServer(session)
        server.start()
        self.acquire(server)
        action_result = {}
        close_done = [threading.Event(), threading.Event()]
        close_errors = []

        def send_action():
            action_result["response"], action_result["body"] = post_json(
                server, "action", action_body(action_id="blocking")
            )

        def close_server(index):
            try:
                server.close()
            except Exception as exc:
                close_errors.append(exc)
            finally:
                close_done[index].set()

        action_thread = threading.Thread(target=send_action)
        action_thread.start()
        self.assertTrue(dispatch_started.wait(2))
        host, port = server.httpd.server_address
        closing_threads = [
            threading.Thread(target=close_server, args=(index,))
            for index in range(2)
        ]
        for thread in closing_threads:
            thread.start()
        try:
            self.assertFalse(close_done[0].wait(0.1))
            self.assertFalse(close_done[1].wait(0.1))
        finally:
            release_dispatch.set()
        action_thread.join(2)
        for thread in closing_threads:
            thread.join(2)

        self.assertEqual([], close_errors)
        self.assertFalse(action_thread.is_alive())
        self.assertTrue(all(event.is_set() for event in close_done))
        self.assertFalse(server.thread.is_alive())
        self.assertEqual(0, server.active_handler_count)
        self.assertEqual(1, len(session.actions))
        self.assertEqual(200, action_result["response"].status)
        rebound = ThreadingHTTPServer((host, port), BaseHTTPRequestHandler)
        try:
            self.assertEqual((host, port), rebound.server_address)
        finally:
            rebound.server_close()

    def test_context_manager_always_closes_and_wait_delegates_without_swallowing_interrupt(self):
        session = fake_session()
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with LocalConfirmationServer(session) as server:
                thread = server.thread
                server.wait_until_session_stops()
                raise RuntimeError("body failed")
        self.assertEqual(1, session.wait_calls)
        self.assertFalse(thread.is_alive())

        class InterruptedSession(FakeSession):
            def wait_until_stopped(self):
                raise KeyboardInterrupt

        server = LocalConfirmationServer(InterruptedSession())
        try:
            with self.assertRaises(KeyboardInterrupt):
                server.wait_until_session_stops()
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
