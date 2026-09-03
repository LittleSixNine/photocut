"""Restricted loopback HTTP transport for one confirmation session."""
from __future__ import annotations

import copy
import json
import logging
import re
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from photocut.confirmation.controller import (
    ConfirmationAction,
    ConfirmationSessionController,
    ConfirmationSessionError,
)
from photocut.confirmation.web.media import (
    ConfirmationMediaService,
    EncodedImage,
    PreviewEncodingError,
    StaleImageToken,
)


LOGGER = logging.getLogger(__name__)
MAX_JSON_BODY = 1024 * 1024
WRITER_LEASE_SECONDS = 15.0
CLIENT_ID_PATTERN = re.compile(r"\A[A-Za-z0-9_-]{1,128}\Z")
CONTENT_LENGTH_PATTERN = re.compile(r"\A[0-9]+\Z")
MEDIA_TOKEN_PATTERN = re.compile(r"\A[A-Za-z0-9_-]{1,128}\Z")
MEDIA_INTEGER_PATTERN = re.compile(r"\A(?:0|[1-9][0-9]{0,3})\Z")
MEDIA_DPR_PATTERN = re.compile(
    r"\A(?:0\.[0-9]+|[1-3](?:\.[0-9]+)?|4(?:\.0+)?)\Z"
)
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' blob:; connect-src 'self'; "
    "object-src 'none'; frame-ancestors 'none'"
)
SECURITY_HEADERS = (
    ("Cache-Control", "no-store"),
    ("Content-Security-Policy", CONTENT_SECURITY_POLICY),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
)
STATIC_ROOT = Path(__file__).with_name("static")
STATIC_ASSETS = {
    "": (STATIC_ROOT / "index.html", "text/html; charset=utf-8"),
    "app.css": (STATIC_ROOT / "app.css", "text/css; charset=utf-8"),
    "app.js": (STATIC_ROOT / "app.js", "text/javascript; charset=utf-8"),
    "tokens.css": (STATIC_ROOT / "tokens.css", "text/css; charset=utf-8"),
}


class WriterLease:
    """Thread-safe, monotonic single-writer lease."""

    def __init__(
        self,
        ttl: float = WRITER_LEASE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if ttl <= 0:
            raise ValueError("writer lease TTL must be positive")
        self.ttl = float(ttl)
        self._clock = clock
        self._lock = threading.Lock()
        self._client_id: str | None = None
        self._expires_at = 0.0

    def _expire(self, now: float) -> None:
        if self._client_id is not None and now >= self._expires_at:
            self._client_id = None
            self._expires_at = 0.0

    def acquire(self, client_id: str) -> bool:
        with self._lock:
            now = self._clock()
            self._expire(now)
            if self._client_id not in (None, client_id):
                return False
            self._client_id = client_id
            self._expires_at = now + self.ttl
            return True

    def heartbeat(self, client_id: str) -> bool:
        return self.acquire(client_id)

    def is_writer(self, client_id: str) -> bool:
        with self._lock:
            now = self._clock()
            self._expire(now)
            return self._client_id == client_id

    def execute_if_writer(self, client_id: str, callback: Callable[[], object]):
        """Recheck and hold the lease fence through canonical dispatch."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            if self._client_id != client_id:
                raise _WriterRequiredError("writer lease is not held")
            return callback()


class _RequestInputError(ValueError):
    pass


class _WriterRequiredError(RuntimeError):
    pass


class _ServerClosingError(RuntimeError):
    pass


class _TrackedThreadingHTTPServer(ThreadingHTTPServer):
    """Count accepted handler threads before they can be scheduled."""

    def process_request(self, request, client_address):
        self.confirmation_service._handler_started()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.confirmation_service._handler_finished()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.confirmation_service._handler_finished()


class LocalConfirmationServer:
    """A short-lived, tokenized HTTP server bound only to IPv4 loopback."""

    def __init__(self, session: ConfirmationSessionController):
        self.session = session
        self.media = ConfirmationMediaService(session)
        self.token = secrets.token_urlsafe(32)
        self.writer_lease = WriterLease()
        self._action_gate = threading.Lock()
        self._lifecycle = threading.Condition()
        self._closing = False
        self._cleanup_complete = False
        self._cleanup_error: BaseException | None = None
        self._active_handlers = 0
        self.thread: threading.Thread | None = None
        self.httpd = _TrackedThreadingHTTPServer(
            ("127.0.0.1", 0), self._handler_type()
        )
        self.httpd.confirmation_service = self
        host, port = self.httpd.server_address
        self.origin = f"http://{host}:{port}"
        self.url = f"{self.origin}/session/{self.token}/"
        self._route_prefix = f"/session/{self.token}/"

    def _handler_type(self):
        class ConfirmationRequestHandler(BaseHTTPRequestHandler):
            server_version = "PhotoCutConfirmation"
            sys_version = ""

            @property
            def service(self) -> "LocalConfirmationServer":
                return self.server.confirmation_service

            def log_message(self, format, *args):
                # The default request log includes the tokenized path.
                return None

            def send_error(self, code, message=None, explain=None):
                # Parser errors are input errors; unsupported methods retain
                # the fixed-route, detail-free not-found response.
                if self.request_version in ("", "HTTP/0.9"):
                    self.request_version = "HTTP/1.0"
                if int(code) in {
                    HTTPStatus.BAD_REQUEST,
                    HTTPStatus.REQUEST_URI_TOO_LONG,
                    HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE,
                    HTTPStatus.HTTP_VERSION_NOT_SUPPORTED,
                }:
                    self._json_response(
                        400, {"error": {"code": "invalid_request"}}
                    )
                else:
                    self._not_found()

            def parse_request(self):
                requestline = self.raw_requestline.decode(
                    "iso-8859-1", errors="replace"
                ).rstrip("\r\n")
                words = requestline.split()
                self.request_target = words[1] if len(words) >= 2 else ""
                parsed = super().parse_request()
                if parsed and self.request_version == "HTTP/0.9":
                    self.request_version = "HTTP/1.0"
                    self.close_connection = True
                    self._json_response(
                        400, {"error": {"code": "invalid_request"}}
                    )
                    return False
                return parsed

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_HEAD(self):
                self._handle("HEAD")

            def do_OPTIONS(self):
                self._handle("OPTIONS")

            def do_PUT(self):
                self._handle("PUT")

            def do_PATCH(self):
                self._handle("PATCH")

            def do_DELETE(self):
                self._handle("DELETE")

            def _handle(self, method: str) -> None:
                try:
                    if not self.service._request_allowed():
                        raise _ServerClosingError
                    self.service._dispatch_request(self, method)
                except _RequestInputError:
                    self._json_response(
                        400, {"error": {"code": "invalid_request"}}
                    )
                except StaleImageToken:
                    self._json_response(
                        409,
                        {
                            "error": {
                                "code": "stale_image_token",
                                "scope": "image",
                            }
                        },
                    )
                except PreviewEncodingError as exc:
                    self._json_response(
                        500,
                        {"error": {"code": exc.code, "scope": "image"}},
                    )
                except _WriterRequiredError:
                    self._json_response(
                        409, {"error": {"code": "writer_required"}}
                    )
                except _ServerClosingError:
                    self._json_response(
                        503, {"error": {"code": "server_closing"}}
                    )
                except ConfirmationSessionError as exc:
                    self._json_response(
                        409,
                        {
                            "error": {
                                "code": exc.code,
                                "message": str(exc),
                                "scope": exc.scope,
                            }
                        },
                    )
                except Exception:
                    LOGGER.exception("unexpected local confirmation request failure")
                    self._json_response(
                        500, {"error": {"code": "internal_error"}}
                    )

            def _json_response(self, status: int, value: object) -> None:
                payload = json.dumps(
                    value, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self._response(
                    status,
                    payload,
                    content_type="application/json; charset=utf-8",
                )

            def _not_found(self) -> None:
                self._response(404, b"")

            def _binary_response(self, value: EncodedImage) -> None:
                if MEDIA_TOKEN_PATTERN.fullmatch(value.image_token) is None:
                    raise RuntimeError("unsafe image token")
                transform = json.dumps(
                    dict(value.transform),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                self._response(
                    200,
                    value.body,
                    content_type=value.content_type,
                    extra_headers=(
                        ("X-PhotoCut-Image-Token", value.image_token),
                        ("X-PhotoCut-Transform", transform),
                    ),
                )

            def _response(
                self,
                status: int,
                payload: bytes,
                *,
                content_type: str | None = None,
                extra_headers: tuple[tuple[str, str], ...] = (),
            ) -> None:
                try:
                    self.send_response(status)
                    for name, value in SECURITY_HEADERS:
                        self.send_header(name, value)
                    if content_type is not None:
                        self.send_header("Content-Type", content_type)
                    for name, value in extra_headers:
                        self.send_header(name, value)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    if payload and self.command != "HEAD":
                        self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

        return ConfirmationRequestHandler

    def _dispatch_request(
        self, handler: BaseHTTPRequestHandler, method: str
    ) -> None:
        target = handler.request_target
        if not target.startswith("/") or target.startswith("//") or "#" in target:
            handler._not_found()
            return
        try:
            parsed = urlsplit(target)
        except ValueError:
            handler._not_found()
            return
        if parsed.scheme or parsed.netloc or parsed.fragment:
            handler._not_found()
            return
        if method == "GET":
            if not parsed.query and parsed.path.startswith(self._route_prefix):
                asset_name = parsed.path[len(self._route_prefix) :]
                asset = STATIC_ASSETS.get(asset_name)
                if asset is not None:
                    path, content_type = asset
                    handler._response(
                        200,
                        path.read_bytes(),
                        content_type=content_type,
                    )
                    return
            json_routes = {
                self._route_prefix + "api/state": self._get_state,
                self._route_prefix + "api/details": self._get_details,
            }
            route = json_routes.get(parsed.path)
            if route is not None:
                client_id = self._client_id_from_query(parsed.query)
                handler._json_response(200, route(client_id))
                return
            media_routes = {
                self._route_prefix + "api/preview": (
                    {"image_token", "css_width", "css_height", "dpr"},
                    self._get_preview,
                ),
                self._route_prefix + "api/prefetch-next": (
                    {"css_width", "css_height", "dpr"},
                    self._get_prefetch_next,
                ),
                self._route_prefix + "api/magnifier": (
                    {
                        "image_token", "corner_index", "zoom", "size",
                        "center_x", "center_y",
                    },
                    self._get_magnifier,
                ),
            }
            media_route = media_routes.get(parsed.path)
            if media_route is not None:
                expected_keys, callback = media_route
                values = self._strict_media_query(parsed.query, expected_keys)
                try:
                    result = callback(values)
                except StaleImageToken:
                    raise
                except PreviewEncodingError:
                    raise
                except ValueError as exc:
                    raise _RequestInputError from exc
                handler._binary_response(result)
                return
            handler._not_found()
            return

        if method == "POST":
            routes = {
                self._route_prefix + "api/lease": (
                    {"client_id"},
                    self._post_lease,
                ),
                self._route_prefix + "api/heartbeat": (
                    {"client_id"},
                    self._post_heartbeat,
                ),
                self._route_prefix + "api/action": (
                    {
                        "client_id",
                        "action_id",
                        "expected_revision",
                        "kind",
                        "payload",
                    },
                    self._post_action,
                ),
            }
            route = routes.get(parsed.path)
            if route is None or "?" in target:
                handler._not_found()
                return
            expected_keys, callback = route
            body = self._read_json_body(handler, expected_keys)
            handler._json_response(200, callback(body))
            return

        handler._not_found()

    def _read_json_body(
        self,
        handler: BaseHTTPRequestHandler,
        expected_keys: set[str],
    ) -> dict:
        if handler.headers.get_all("Origin", []) != [self.origin]:
            raise _RequestInputError
        if handler.headers.get_all("Content-Type", []) != ["application/json"]:
            raise _RequestInputError
        if handler.headers.get("Transfer-Encoding") is not None:
            raise _RequestInputError
        lengths = handler.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            raise _RequestInputError
        if CONTENT_LENGTH_PATTERN.fullmatch(lengths[0]) is None:
            raise _RequestInputError
        try:
            length = int(lengths[0])
        except ValueError as exc:
            raise _RequestInputError from exc
        if not 0 <= length <= MAX_JSON_BODY:
            raise _RequestInputError
        raw = handler.rfile.read(length)
        if len(raw) != length:
            raise _RequestInputError
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=self._unique_json_object,
                parse_constant=self._reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _RequestInputError from exc
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise _RequestInputError
        return value

    @staticmethod
    def _unique_json_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise _RequestInputError
            value[key] = item
        return value

    @staticmethod
    def _reject_json_constant(value):
        del value
        raise _RequestInputError

    @staticmethod
    def _validate_client_id(value: object) -> str:
        if not isinstance(value, str) or CLIENT_ID_PATTERN.fullmatch(value) is None:
            raise _RequestInputError
        return value

    def _client_id_from_query(self, query: str) -> str:
        try:
            values = parse_qs(
                query,
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=2,
            )
        except ValueError as exc:
            raise _RequestInputError from exc
        if set(values) != {"client_id"} or len(values["client_id"]) != 1:
            raise _RequestInputError
        return self._validate_client_id(values["client_id"][0])

    @staticmethod
    def _strict_media_query(query: str, expected_keys: set[str]) -> dict[str, str]:
        if not query or any(character in query for character in "%+;#"):
            raise _RequestInputError
        pairs = query.split("&")
        if len(pairs) != len(expected_keys):
            raise _RequestInputError
        values = {}
        for pair in pairs:
            if pair.count("=") != 1:
                raise _RequestInputError
            key, value = pair.split("=", 1)
            if key not in expected_keys or key in values or not value:
                raise _RequestInputError
            try:
                key.encode("ascii")
                value.encode("ascii")
            except UnicodeEncodeError as exc:
                raise _RequestInputError from exc
            values[key] = value
        if set(values) != expected_keys:
            raise _RequestInputError
        return values

    @staticmethod
    def _media_integer(value: str) -> int:
        if MEDIA_INTEGER_PATTERN.fullmatch(value) is None:
            raise _RequestInputError
        return int(value)

    @staticmethod
    def _media_dpr(value: str) -> float:
        if MEDIA_DPR_PATTERN.fullmatch(value) is None:
            raise _RequestInputError
        result = float(value)
        if not result.is_integer() and str(result) != value.rstrip("0").rstrip("."):
            # Multiple decimal spellings are harmless numerically but keeping
            # one canonical form prevents cache-key/query ambiguity.
            raise _RequestInputError
        return result

    def _get_preview(self, values: dict[str, str]) -> EncodedImage:
        token = values["image_token"]
        if MEDIA_TOKEN_PATTERN.fullmatch(token) is None:
            raise _RequestInputError
        return self.media.preview(
            token,
            self._media_integer(values["css_width"]),
            self._media_integer(values["css_height"]),
            self._media_dpr(values["dpr"]),
        )

    def _get_prefetch_next(self, values: dict[str, str]) -> EncodedImage:
        return self.media.prefetch_next(
            self._media_integer(values["css_width"]),
            self._media_integer(values["css_height"]),
            self._media_dpr(values["dpr"]),
        )

    def _get_magnifier(self, values: dict[str, str]) -> EncodedImage:
        token = values["image_token"]
        if MEDIA_TOKEN_PATTERN.fullmatch(token) is None:
            raise _RequestInputError
        return self.media.magnifier(
            token,
            self._media_integer(values["corner_index"]),
            self._media_integer(values["zoom"]),
            self._media_integer(values["size"]),
            center=(
                self._media_integer(values["center_x"]),
                self._media_integer(values["center_y"]),
            ),
        )

    def _get_state(self, client_id: str) -> dict:
        snapshot = copy.deepcopy(self.session.snapshot())
        if not isinstance(snapshot, dict) or not isinstance(
            snapshot.get("session"), dict
        ):
            raise TypeError("invalid confirmation snapshot")
        snapshot["session"]["readonly"] = not self.writer_lease.is_writer(
            client_id
        )
        return snapshot

    def _get_details(self, client_id: str) -> dict:
        del client_id
        details = self.session.technical_details()
        if not isinstance(details, dict):
            raise TypeError("invalid confirmation details")
        return details

    def _post_lease(self, body: dict) -> dict:
        client_id = self._validate_client_id(body["client_id"])
        return {"writer": self.writer_lease.acquire(client_id)}

    def _post_heartbeat(self, body: dict) -> dict:
        client_id = self._validate_client_id(body["client_id"])
        return {"writer": self.writer_lease.heartbeat(client_id)}

    def _post_action(self, body: dict) -> dict:
        client_id = self._validate_client_id(body["client_id"])
        action = ConfirmationAction(
            action_id=body["action_id"],
            expected_revision=body["expected_revision"],
            kind=body["kind"],
            payload=body["payload"],
        )
        with self._action_gate:
            with self._lifecycle:
                if self._closing:
                    raise _ServerClosingError
            result = self.writer_lease.execute_if_writer(
                client_id, lambda: self.session.dispatch(action)
            )
            if self.media.has_observed_current:
                self.media.sync_navigation()
            return result

    def _handler_started(self) -> None:
        with self._lifecycle:
            self._active_handlers += 1

    def _handler_finished(self) -> None:
        with self._lifecycle:
            self._active_handlers -= 1
            self._lifecycle.notify_all()

    def _request_allowed(self) -> bool:
        with self._lifecycle:
            return not self._closing

    @property
    def active_handler_count(self) -> int:
        with self._lifecycle:
            return self._active_handlers

    def start(self) -> str:
        with self._lifecycle:
            if self._closing or self._cleanup_complete:
                raise RuntimeError("confirmation server is closed")
            if self.thread is not None:
                if not self.thread.is_alive():
                    raise RuntimeError("confirmation server stopped unexpectedly")
                return self.url
            self.thread = threading.Thread(
                target=self.httpd.serve_forever,
                name="photocut-confirmation-server",
                daemon=False,
            )
            self.thread.start()
            return self.url

    def wait_until_session_stops(self) -> None:
        self.session.wait_until_stopped()

    def close(self) -> None:
        with self._lifecycle:
            if self._cleanup_complete:
                error = self._cleanup_error
                if error is not None:
                    raise error
                return
            if self._closing:
                while not self._cleanup_complete:
                    self._lifecycle.wait()
                error = self._cleanup_error
                if error is not None:
                    raise error
                return
            self._closing = True
            thread = self.thread

        deadline = time.monotonic() + 5.0
        error = None
        try:
            if thread is not None and thread.is_alive():
                self.httpd.shutdown()
            self.httpd.server_close()
            if thread is not None:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if thread.is_alive():
                    raise RuntimeError("confirmation server did not stop")
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._action_gate.acquire(
                timeout=max(0.0, remaining)
            ):
                raise RuntimeError("confirmation server dispatch did not stop")
            self._action_gate.release()
            with self._lifecycle:
                while self._active_handlers:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(
                            "confirmation server handlers did not stop"
                        )
                    self._lifecycle.wait(remaining)
        except BaseException as exc:
            error = exc
        finally:
            with self._lifecycle:
                self._cleanup_error = error
                self._cleanup_complete = True
                self._lifecycle.notify_all()
        if error is not None:
            raise error

    def __enter__(self) -> "LocalConfirmationServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["LocalConfirmationServer", "WriterLease"]
