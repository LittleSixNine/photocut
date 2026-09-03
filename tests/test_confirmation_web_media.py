import concurrent.futures
import json
import math
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import numpy as np

from photocut.confirmation.controller import ConfirmationAction, ConfirmationSessionError
from photocut.confirmation.web.media import (
    ByteBudgetCache,
    ConfirmationMediaService,
    EncodedImage,
    PreviewEncodingError,
    StaleImageToken,
)
from photocut.confirmation.web.server import LocalConfirmationServer


class FakeMediaSession:
    def __init__(self):
        y, x = np.mgrid[0:80, 0:100]
        self.images = [
            np.dstack((x, y, (x + y) % 255)).astype(np.uint8),
            np.dstack((x + 1, y + 2, (x + y + 3) % 255)).astype(np.uint8),
            np.dstack((x + 4, y + 5, (x + y + 6) % 255)).astype(np.uint8),
        ]
        self.tokens = ["img-1", "img-2", "img-3"]
        self.index = 0
        self.corners = [[10, 10], [90, 10], [90, 70], [10, 70]]
        self.actions = []
        self.lock = threading.RLock()

    def current_image(self):
        with self.lock:
            return self.images[self.index], self.tokens[self.index]

    def next_preview_source(self):
        with self.lock:
            if self.index + 1 >= len(self.images):
                raise RuntimeError("no next image")
            return self.images[self.index + 1].copy(), self.tokens[self.index + 1]

    def snapshot(self):
        with self.lock:
            return {
                "revision": len(self.actions),
                "session": {"status": "active", "readonly": False},
                "editor": {"corners": [point[:] for point in self.corners]},
            }

    def technical_details(self):
        return {"path": "/must/not/leak/source.jpg"}

    def toggle_candidate_without_navigation(self):
        return None

    def dispatch(self, action):
        if not isinstance(action, ConfirmationAction):
            raise AssertionError("canonical action required")
        with self.lock:
            self.actions.append(action)
            if action.kind == "next":
                self.index += 1
            return self.snapshot()

    def wait_until_stopped(self):
        return None


def get_response(url):
    try:
        return urllib.request.urlopen(url, timeout=2)
    except urllib.error.HTTPError as exc:
        return exc


def post_json(server, path, body):
    request = urllib.request.Request(
        server.url + path,
        method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Origin": server.origin},
    )
    try:
        return urllib.request.urlopen(request, timeout=2)
    except urllib.error.HTTPError as exc:
        return exc


class ConfirmationWebMediaTests(unittest.TestCase):
    def test_preview_is_binary_and_candidate_changes_do_not_reencode(self):
        session = FakeMediaSession()
        service = ConfirmationMediaService(session, preview_budget=64 * 1024 * 1024)

        first = service.preview("img-1", 1200, 800, 2.0)
        session.toggle_candidate_without_navigation()
        second = service.preview("img-1", 1200, 800, 2.0)

        self.assertEqual("image/jpeg", first.content_type)
        self.assertTrue(first.body.startswith(b"\xff\xd8"))
        self.assertEqual(first, second)
        self.assertEqual(1, service.preview_encode_count)

    def test_byte_budget_is_thread_safe_lru_and_oversize_entry_is_not_retained(self):
        cache = ByteBudgetCache(max_bytes=10)
        cache.put("a", b"123456")
        cache.put("b", b"abcdef")
        self.assertIsNone(cache.get("a"))
        self.assertEqual(b"abcdef", cache.get("b"))
        cache.put("c", b"1234")
        self.assertEqual(b"abcdef", cache.get("b"))
        cache.put("d", b"5678")
        self.assertIsNone(cache.get("c"))
        self.assertEqual(b"abcdef", cache.get("b"))
        self.assertEqual(b"5678", cache.get("d"))

        cache.put("oversize", b"x" * 11)
        self.assertIsNone(cache.get("oversize"))
        self.assertLessEqual(cache.total_bytes, cache.max_bytes)

        def write(index):
            cache.put(str(index), bytes([index % 255]) * ((index % 7) + 1))
            cache.get(str(index - 1))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write, range(500)))
        self.assertGreaterEqual(cache.total_bytes, 0)
        self.assertLessEqual(cache.total_bytes, cache.max_bytes)
        self.assertEqual(
            cache.total_bytes,
            sum(len(value) for value in cache.snapshot().values()),
        )

    def test_encoded_image_is_deeply_immutable_and_budget_counts_body_only(self):
        encoded = EncodedImage(
            body=b"1234",
            content_type="image/jpeg",
            image_token="img-1",
            width=10,
            height=8,
            transform={"scale": 1.0},
        )
        cache = ByteBudgetCache(4)
        cache.put("image", encoded)

        with self.assertRaises(TypeError):
            encoded.transform["scale"] = 2.0
        with self.assertRaises(Exception):
            encoded.width = 20
        self.assertEqual(4, cache.total_bytes)

    def test_preview_rejects_bool_nonfinite_and_out_of_range_dimensions(self):
        service = ConfirmationMediaService(FakeMediaSession())
        invalid = (
            (True, 800, 1.0),
            (1200, False, 1.0),
            (0, 800, 1.0),
            (4097, 800, 1.0),
            (1200, 800, True),
            (1200, 800, math.nan),
            (1200, 800, math.inf),
            (1200, 800, 0.49),
            (1200, 800, 4.01),
        )
        for width, height, dpr in invalid:
            with self.subTest(width=width, height=height, dpr=dpr):
                with self.assertRaises(ValueError):
                    service.preview("img-1", width, height, dpr)

    def test_stale_token_fails_closed_before_cache_or_encoding(self):
        service = ConfirmationMediaService(FakeMediaSession())
        with self.assertRaises(StaleImageToken):
            service.preview("img-2", 1200, 800, 1.0)
        with self.assertRaises(StaleImageToken):
            service.magnifier("img-2", 0, 4, 400)
        self.assertEqual(0, service.preview_encode_count)
        self.assertEqual(0, service.preview_cache.total_bytes)
        self.assertEqual(0, service.magnifier_cache.total_bytes)

    def test_navigation_during_preview_encoding_fails_closed_and_drops_payload(self):
        session = FakeMediaSession()
        service = ConfirmationMediaService(session)
        from photocut.confirmation.web import media as media_module

        real_imencode = media_module.cv2.imencode

        def navigate_after_encoding(*args, **kwargs):
            result = real_imencode(*args, **kwargs)
            session.index = 1
            return result

        with patch("photocut.confirmation.web.media.cv2.imencode", side_effect=navigate_after_encoding):
            with self.assertRaises(StaleImageToken):
                service.preview("img-1", 1200, 800, 1.0)

        self.assertNotIn("img-1", service.preview_cache_tokens)

    def test_next_prefetch_keeps_only_one_encoded_payload_and_no_full_array(self):
        session = FakeMediaSession()
        service = ConfirmationMediaService(session, preview_budget=64 * 1024 * 1024)

        current = service.preview("img-1", 1200, 800, 1.0)
        next_image = service.prefetch_next(1200, 800, 1.0)

        self.assertEqual("image/jpeg", next_image.content_type)
        self.assertEqual("img-2", next_image.image_token)
        self.assertIsNone(service.prefetched_full_image)
        self.assertEqual({"img-1", "img-2"}, service.preview_cache_tokens)
        self.assertLessEqual(len(service.preview_cache), 2)
        self.assertEqual(2, service.preview_encode_count)
        self.assertEqual(current, service.preview("img-1", 1200, 800, 1.0))

    def test_navigation_clears_old_magnifier_and_unrelated_preview_entries(self):
        session = FakeMediaSession()
        service = ConfirmationMediaService(session)
        service.preview("img-1", 1200, 800, 1.0)
        service.prefetch_next(1200, 800, 1.0)
        service.magnifier("img-1", 0, 4, 400)

        session.index = 1
        service.sync_navigation()

        self.assertEqual({"img-2"}, service.preview_cache_tokens)
        self.assertEqual(0, len(service.magnifier_cache))
        service.prefetch_next(1200, 800, 1.0)
        self.assertEqual({"img-2", "img-3"}, service.preview_cache_tokens)

    def test_magnifier_key_tracks_canonical_corner_and_encoding_failures_are_structured(self):
        session = FakeMediaSession()
        service = ConfirmationMediaService(session)
        first = service.magnifier("img-1", 0, 4, 400)
        again = service.magnifier("img-1", 0, 4, 400)
        self.assertEqual(first, again)
        self.assertEqual(1, service.magnifier_encode_count)

        session.corners[0] = [11, 10]
        changed = service.magnifier("img-1", 0, 4, 400)
        self.assertNotEqual(first.body, changed.body)
        self.assertEqual(2, service.magnifier_encode_count)

        with patch("photocut.confirmation.web.media.cv2.imencode", return_value=(False, None)):
            with self.assertRaisesRegex(PreviewEncodingError, "preview") as preview:
                service.preview("img-1", 1000, 700, 1.0)
            with self.assertRaisesRegex(PreviewEncodingError, "magnifier") as magnifier:
                service.magnifier("img-1", 1, 8, 400)
        self.assertEqual("preview_encoding_failed", preview.exception.code)
        self.assertEqual("magnifier_encoding_failed", magnifier.exception.code)

    def test_transient_magnifier_center_is_bounded_and_does_not_mutate_session(self):
        session = FakeMediaSession()
        service = ConfirmationMediaService(session)
        before = [point[:] for point in session.corners]

        canonical = service.magnifier("img-1", 0, 2, 768)
        transient = service.magnifier(
            "img-1", 0, 2, 768, center=(20, 20)
        )

        self.assertNotEqual(canonical.body, transient.body)
        self.assertEqual((384, 384), (transient.width, transient.height))
        self.assertEqual(before, session.corners)
        for center in ((-1, 10), (100, 10), (10, 80), (True, 10), (10,)):
            with self.subTest(center=center):
                with self.assertRaises(ValueError):
                    service.magnifier("img-1", 0, 2, 768, center=center)

    def test_magnifier_selection_is_strict_and_rejects_bool(self):
        service = ConfirmationMediaService(FakeMediaSession())
        self.assertEqual("image/png", service.magnifier("img-1", 0, 2, 400).content_type)
        invalid = (
            (True, 4, 400),
            (4, 4, 400),
            (0, True, 400),
            (0, 4.0, 400),
            (0, 16, 400),
            (0, 4, True),
            (0, 4, 63),
            (0, 4, 513),
            (0, 16, 402),
        )
        for corner, zoom, size in invalid:
            with self.subTest(corner=corner, zoom=zoom, size=size):
                with self.assertRaises(ValueError):
                    service.magnifier("img-1", corner, zoom, size)

    def test_same_key_concurrent_requests_encode_once_and_signal_waiters(self):
        session = FakeMediaSession()
        service = ConfirmationMediaService(session)
        from photocut.confirmation.web import media as media_module

        real_imencode = media_module.cv2.imencode
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def blocking_encode(*args, **kwargs):
            calls.append(1)
            entered.set()
            if not release.wait(2):
                raise AssertionError("encode release timed out")
            return real_imencode(*args, **kwargs)

        with patch("photocut.confirmation.web.media.cv2.imencode", side_effect=blocking_encode):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(service.preview, "img-1", 1200, 800, 1.0)
                self.assertTrue(entered.wait(1))
                second = executor.submit(service.preview, "img-1", 1200, 800, 1.0)
                release.set()
                self.assertEqual(first.result(2), second.result(2))

        self.assertEqual(1, len(calls))
        self.assertEqual(0, service.inflight_count)

    def test_singleflight_exception_wakes_waiter_and_allows_retry(self):
        service = ConfirmationMediaService(FakeMediaSession())
        entered = threading.Event()
        release = threading.Event()

        def failed_encode(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise AssertionError("encode release timed out")
            return False, None

        with patch("photocut.confirmation.web.media.cv2.imencode", side_effect=failed_encode):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(service.preview, "img-1", 1200, 800, 1.0)
                self.assertTrue(entered.wait(1))
                second = executor.submit(service.preview, "img-1", 1200, 800, 1.0)
                release.set()
                for future in (first, second):
                    with self.assertRaises(PreviewEncodingError):
                        future.result(2)

        self.assertEqual(0, service.inflight_count)
        self.assertEqual("image/jpeg", service.preview("img-1", 1200, 800, 1.0).content_type)

    def test_slow_prefetch_does_not_block_unrelated_current_preview(self):
        class SlowPrefetchSession(FakeMediaSession):
            def __init__(self):
                super().__init__()
                self.prefetch_started = threading.Event()
                self.prefetch_release = threading.Event()

            def next_preview_source(self):
                self.prefetch_started.set()
                if not self.prefetch_release.wait(2):
                    raise AssertionError("prefetch release timed out")
                return self.images[1].copy(), self.tokens[1]

        session = SlowPrefetchSession()
        service = ConfirmationMediaService(session)
        result = {}

        def prefetch():
            result["prefetch"] = service.prefetch_next(1200, 800, 1.0)

        thread = threading.Thread(target=prefetch)
        thread.start()
        self.assertTrue(session.prefetch_started.wait(1))
        started = time.perf_counter()
        current = service.preview("img-1", 1200, 800, 1.0)
        elapsed = time.perf_counter() - started
        session.prefetch_release.set()
        thread.join(2)

        self.assertLess(elapsed, 0.25)
        self.assertEqual("img-1", current.image_token)
        self.assertFalse(thread.is_alive())
        self.assertEqual("img-2", result["prefetch"].image_token)

    def test_navigation_during_prefetch_discards_stale_payload_without_caching(self):
        class StalePrefetchSession(FakeMediaSession):
            def __init__(self):
                super().__init__()
                self.prefetch_started = threading.Event()
                self.prefetch_release = threading.Event()

            def next_preview_source(self):
                with self.lock:
                    source_index = self.index
                    source_token = self.tokens[self.index]
                self.prefetch_started.set()
                if not self.prefetch_release.wait(2):
                    raise AssertionError("prefetch release timed out")
                with self.lock:
                    if self.index != source_index or self.tokens[self.index] != source_token:
                        raise ConfirmationSessionError(
                            "stale_preview",
                            "confirmation image changed during next preview load",
                            "image",
                        )
                    return self.images[source_index + 1].copy(), self.tokens[source_index + 1]

        session = StalePrefetchSession()
        service = ConfirmationMediaService(session)
        result = {}

        def prefetch():
            try:
                result["value"] = service.prefetch_next(1200, 800, 1.0)
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=prefetch)
        thread.start()
        self.assertTrue(session.prefetch_started.wait(1))
        session.index = 1
        session.prefetch_release.set()
        thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertNotIn("value", result)
        self.assertIsInstance(result.get("error"), ConfirmationSessionError)
        self.assertEqual("stale_preview", result["error"].code)
        self.assertNotIn("img-2", service.preview_cache_tokens)
        self.assertEqual(0, service.preview_encode_count)
        self.assertEqual(0, service.inflight_count)

    def test_magnifier_cache_hit_is_fenced_against_navigation_during_lookup(self):
        session = FakeMediaSession()
        service = ConfirmationMediaService(session)
        service.magnifier("img-1", 0, 4, 400)
        real_get = service.magnifier_cache.get

        def navigate_then_get(key):
            value = real_get(key)
            session.index = 1
            return value

        with patch.object(service.magnifier_cache, "get", side_effect=navigate_then_get):
            with self.assertRaises(StaleImageToken):
                service.magnifier("img-1", 0, 4, 400)

        self.assertEqual(0, len(service.magnifier_cache))


class ConfirmationMediaRouteTests(unittest.TestCase):
    def assert_security_headers(self, response):
        self.assertEqual("no-store", response.headers.get("Cache-Control"))
        self.assertEqual("nosniff", response.headers.get("X-Content-Type-Options"))
        self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

    def test_binary_routes_have_canonical_headers_and_no_path_disclosure(self):
        session = FakeMediaSession()
        with LocalConfirmationServer(session) as server:
            urls = (
                server.url
                + "api/preview?image_token=img-1&css_width=1200&css_height=800&dpr=1",
                server.url
                + "api/prefetch-next?css_width=1200&css_height=800&dpr=1",
                server.url
                + "api/magnifier?image_token=img-1&corner_index=0&zoom=4&size=400&center_x=10&center_y=10",
            )
            for url, mime, token in zip(
                urls,
                ("image/jpeg", "image/jpeg", "image/png"),
                ("img-1", "img-2", "img-1"),
            ):
                with self.subTest(url=url):
                    response = get_response(url)
                    body = response.read()
                    self.assertEqual(200, response.status)
                    self.assertEqual(mime, response.headers["Content-Type"])
                    self.assertEqual(str(len(body)), response.headers["Content-Length"])
                    self.assertEqual(token, response.headers["X-PhotoCut-Image-Token"])
                    transform = response.headers["X-PhotoCut-Transform"]
                    json.loads(transform)
                    self.assertNotIn("/", transform)
                    self.assertNotIn("source", transform.lower())
                    self.assert_security_headers(response)

    def test_media_queries_reject_duplicates_unknown_percent_and_noncanonical_values(self):
        with LocalConfirmationServer(FakeMediaSession()) as server:
            invalid_queries = (
                "image_token=img-1&image_token=img-1&css_width=1200&css_height=800&dpr=1",
                "image_token=img-1&css_width=1200&css_height=800&dpr=1&extra=1",
                "image_token=img%2D1&css_width=1200&css_height=800&dpr=1",
                "image_token=img-1&css_width=%31%32%30%30&css_height=800&dpr=1",
                "image_token=img-1&css_width=01200&css_height=800&dpr=1",
                "image_token=img-1&css_width=1200&css_height=800&dpr=NaN",
                "image_token=img-1&css_width=1200&css_height=800&dpr=inf",
                "image_token=img-1&css_width=1200&css_height=800&dpr=1e0",
            )
            for query in invalid_queries:
                with self.subTest(query=query):
                    response = get_response(server.url + "api/preview?" + query)
                    self.assertEqual(400, response.status)
                    self.assertEqual(
                        {"error": {"code": "invalid_request"}},
                        json.loads(response.read()),
                    )
                    self.assert_security_headers(response)

            invalid_magnifiers = (
                "image_token=img-1&corner_index=0&zoom=2&size=768",
                "image_token=img-1&corner_index=0&zoom=2&size=768&center_x=10",
                "image_token=img-1&corner_index=0&zoom=2&size=768&center_x=-1&center_y=10",
                "image_token=img-1&corner_index=0&zoom=2&size=768&center_x=100&center_y=10",
            )
            for query in invalid_magnifiers:
                with self.subTest(query=query):
                    response = get_response(server.url + "api/magnifier?" + query)
                    self.assertEqual(400, response.status)

    def test_stale_and_encoding_failures_map_to_safe_structured_errors(self):
        with LocalConfirmationServer(FakeMediaSession()) as server:
            stale = get_response(
                server.url
                + "api/preview?image_token=img-2&css_width=1200&css_height=800&dpr=1"
            )
            self.assertEqual(409, stale.status)
            self.assertEqual(
                {"error": {"code": "stale_image_token", "scope": "image"}},
                json.loads(stale.read()),
            )

            with patch("photocut.confirmation.web.media.cv2.imencode", return_value=(False, None)):
                failed = get_response(
                    server.url
                    + "api/preview?image_token=img-1&css_width=1200&css_height=800&dpr=1"
                )
            self.assertEqual(500, failed.status)
            payload = failed.read().decode("utf-8")
            self.assertEqual(
                {
                    "error": {
                        "code": "preview_encoding_failed",
                        "scope": "image",
                    }
                },
                json.loads(payload),
            )
            self.assertNotIn("/must/not/leak", payload)

    def test_navigation_action_immediately_prunes_media_cache(self):
        session = FakeMediaSession()
        with LocalConfirmationServer(session) as server:
            get_response(
                server.url
                + "api/preview?image_token=img-1&css_width=1200&css_height=800&dpr=1"
            ).read()
            get_response(
                server.url
                + "api/prefetch-next?css_width=1200&css_height=800&dpr=1"
            ).read()
            get_response(
                server.url
                + "api/magnifier?image_token=img-1&corner_index=0&zoom=4&size=400&center_x=10&center_y=10"
            ).read()
            lease = post_json(server, "api/lease", {"client_id": "client-a"})
            lease.read()
            action = post_json(
                server,
                "api/action",
                {
                    "client_id": "client-a",
                    "action_id": "next-1",
                    "expected_revision": 0,
                    "kind": "next",
                    "payload": {},
                },
            )
            self.assertEqual(200, action.status)
            action.read()

            self.assertEqual({"img-2"}, server.media.preview_cache_tokens)
            self.assertEqual(0, len(server.media.magnifier_cache))


if __name__ == "__main__":
    unittest.main()
