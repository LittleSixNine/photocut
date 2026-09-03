import threading
import time
import unittest

from photocut.algorithms.v7.coordinator import DetectionCoordinator, RequestIdentity


class CoordinatorTests(unittest.TestCase):
    def identity(self, request="r1", mode="safe"):
        return RequestIdentity(request, "sha256:image", "exif_1", "7.0", "a" * 64, mode)

    def test_only_exact_current_identity_is_delivered(self):
        delivered = []
        coordinator = DetectionCoordinator(lambda image, **kwargs: {"image": image}, on_result=lambda *args: delivered.append(args))
        try:
            current = self.identity()
            coordinator.set_current(current)
            self.assertTrue(coordinator.accept_result(self.identity(), {"ok": True}))
            self.assertFalse(coordinator.accept_result(self.identity("stale"), {"ok": False}))
            self.assertEqual(1, len(delivered))
            self.assertEqual(1, coordinator.cache_size)
        finally:
            coordinator.close()

    def test_queue_is_bounded_and_drain_for_test_is_deterministic(self):
        gate = threading.Event()
        started = threading.Event()
        def detector(image, **kwargs):
            started.set()
            gate.wait(1.0)
            return image
        coordinator = DetectionCoordinator(detector, worker_count=1, max_queue=1)
        try:
            self.assertTrue(coordinator.submit(self.identity("r1"), 1))
            started.wait(1.0)
            self.assertTrue(coordinator.submit(self.identity("r2"), 2))
            self.assertFalse(coordinator.submit(self.identity("r3"), 3))
            gate.set()
            self.assertTrue(coordinator.drain_for_test(2.0))
        finally:
            coordinator.close()

    def test_cancel_and_close_do_not_deliver_cancelled_work(self):
        gate = threading.Event()
        def detector(image, **kwargs):
            gate.wait(1.0)
            return image
        delivered = []
        coordinator = DetectionCoordinator(detector, on_result=lambda *args: delivered.append(args))
        identity = self.identity()
        try:
            coordinator.set_current(identity)
            self.assertTrue(coordinator.submit(identity, "image"))
            coordinator.cancel("user_cancelled")
            gate.set()
            coordinator.drain_for_test(2.0)
            self.assertEqual([], delivered)
            self.assertTrue(any(item["cancellation_reason"] == "user_cancelled" for item in coordinator.metrics()))
        finally:
            coordinator.close()

    def test_navigate_supersedes_previous_and_mode_identity_is_strict(self):
        coordinator = DetectionCoordinator(lambda image, **kwargs: image)
        try:
            first = self.identity("r1")
            second = self.identity("r2", mode="aggressive")
            coordinator.navigate(first, "first")
            coordinator.navigate(second, "second")
            self.assertEqual(second, coordinator.current_identity)
            self.assertFalse(coordinator.accept_result(first, "old"))
            self.assertTrue(coordinator.accept_result(second, "new"))
        finally:
            coordinator.close()


if __name__ == "__main__":
    unittest.main()
