import time
import unittest
import hashlib

import numpy as np


class LegacyWorkerTests(unittest.TestCase):
    def setUp(self):
        from photocut.algorithms.v7.legacy_worker import SourceSnapshot

        image = np.zeros((20, 30, 3), dtype=np.uint8)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        self.snapshot = SourceSnapshot(image, digest, (30, 20), (30, 20), "identity")

    def test_worker_returns_snapshot_identity(self):
        from photocut.algorithms.v7.legacy_worker import run_legacy_worker

        result = run_legacy_worker(self.snapshot, timeout_s=1.0,
                                   detector=lambda snapshot: {"success": True, "corners": []})
        self.assertEqual("ok", result["status"])
        self.assertEqual(self.snapshot.source_sha256, result["source_sha256"])

    def test_worker_timeout_terminates_slow_legacy(self):
        from photocut.algorithms.v7.legacy_worker import run_legacy_worker

        def slow(_snapshot):
            time.sleep(0.4)
            return {"success": True}

        result = run_legacy_worker(self.snapshot, timeout_s=0.05, detector=slow)
        self.assertEqual("timeout", result["status"])

    def test_worker_cancel_does_not_return_legacy_result(self):
        from photocut.algorithms.v7.legacy_worker import run_legacy_worker

        class Cancel:
            is_set = True

        result = run_legacy_worker(self.snapshot, timeout_s=1.0, cancellation_token=Cancel(),
                                   detector=lambda snapshot: {"success": True})
        self.assertEqual("cancelled", result["status"])


if __name__ == "__main__":
    unittest.main()
