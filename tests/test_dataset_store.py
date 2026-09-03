import fcntl
import hashlib
import json
import os
import stat
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from photocut.data import dataset_store
from photocut.data.dataset_store import (
    ARCHIVE_SPACE_RESERVE_BYTES,
    BatchLock,
    DatasetStore,
    append_jsonl,
    append_jsonl_validated,
    atomic_write_json,
    load_jsonl,
    sha256_file,
)


class AtomicPersistenceTests(unittest.TestCase):
    def test_atomic_json_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "state.json"

            atomic_write_json(path, {"version": 1})
            atomic_write_json(path, {"version": 2, "complete": True})

            self.assertEqual(
                {"version": 2, "complete": True},
                json.loads(path.read_text(encoding="utf-8")),
            )
            self.assertEqual([], list(path.parent.glob(f".{path.name}.*.tmp")))

    def test_replace_failure_preserves_previous_document_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "state.json"
            atomic_write_json(path, {"version": 1})

            with patch("photocut.data.dataset_store.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    atomic_write_json(path, {"version": 2})

            self.assertEqual({"version": 1}, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual([], list(path.parent.glob(f".{path.name}.*.tmp")))

    def test_jsonl_appends_one_event_per_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"

            append_jsonl(path, {"id": 1})
            append_jsonl(path, {"id": 2})

            self.assertEqual([{"id": 1}, {"id": 2}], load_jsonl(path))

    def test_jsonl_round_trips_unicode_line_separators_inside_fields(self):
        for separator in ("\u0085", "\u2028", "\u2029"):
            with self.subTest(separator=hex(ord(separator))):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir).resolve() / "events.jsonl"
                    event = {"id": 1, "value": f"before{separator}after"}

                    append_jsonl(path, event)

                    self.assertEqual(1, path.read_bytes().count(b"\n"))
                    self.assertEqual([event], load_jsonl(path))

    def test_jsonl_crlf_blank_lines_and_terminal_lf_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"
            path.write_bytes(b'{"id":1}\r\n\r\n')

            self.assertEqual([{"id": 1}], load_jsonl(path))
            append_jsonl(path, {"id": 2})
            self.assertEqual(b'{"id":1}\r\n\r\n{"id":2}\n', path.read_bytes())

            truncated = path.parent / "truncated.jsonl"
            original = b'{"id":1}\r\n\r\n{"id":2}'
            truncated.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "terminal newline"):
                append_jsonl(truncated, {"id": 3})
            self.assertEqual(original, truncated.read_bytes())

            invalid = path.parent / "invalid.jsonl"
            invalid.write_bytes(b'{"id":1}\r\n\r\nnot-json\n')
            with self.assertRaisesRegex(ValueError, "line 3"):
                load_jsonl(invalid)

    def test_jsonl_partial_write_restores_existing_bytes_and_fsyncs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"
            append_jsonl(path, {"id": 1})
            original = path.read_bytes()
            real_write = os.write
            real_fsync = os.fsync
            write_calls = 0

            def partial_write_then_fail(descriptor, data):
                nonlocal write_calls
                write_calls += 1
                if write_calls == 1:
                    return real_write(descriptor, data[:5])
                raise OSError("partial write failed")

            with patch(
                "photocut.data.dataset_store.os.write", side_effect=partial_write_then_fail
            ):
                with patch(
                    "photocut.data.dataset_store.os.fsync", wraps=real_fsync
                ) as fsync:
                    with self.assertRaisesRegex(OSError, "partial write failed"):
                        append_jsonl(path, {"id": 2})

            self.assertEqual(original, path.read_bytes())
            self.assertGreaterEqual(fsync.call_count, 1)
            self.assertEqual([{"id": 1}], load_jsonl(path))

    def test_jsonl_partial_write_removes_new_file_after_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"
            real_write = os.write
            write_calls = 0

            def partial_write_then_fail(descriptor, data):
                nonlocal write_calls
                write_calls += 1
                if write_calls == 1:
                    return real_write(descriptor, data[:5])
                raise OSError("partial write failed")

            with patch(
                "photocut.data.dataset_store.os.write", side_effect=partial_write_then_fail
            ):
                with self.assertRaisesRegex(OSError, "partial write failed"):
                    append_jsonl(path, {"id": 1})

            self.assertFalse(path.exists())
            self.assertEqual([], list(path.parent.glob(f".{path.name}.*.tmp")))

    def test_jsonl_validated_append_rejects_unexpected_existing_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"
            append_jsonl(path, {"id": "unrelated"})
            original = path.read_bytes()

            def require_empty_history(events):
                if events:
                    raise ValueError("history changed before append")

            with self.assertRaisesRegex(ValueError, "history changed"):
                append_jsonl_validated(
                    path, {"id": 1}, require_empty_history
                )

            self.assertEqual(original, path.read_bytes())

    def test_jsonl_new_file_rollback_never_unlinks_a_replacement_leaf(self):
        for replacement_kind in ("file", "symlink"):
            with self.subTest(replacement_kind=replacement_kind):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir).resolve()
                    path = root / "events.jsonl"
                    moved = root / "failed-append.jsonl"
                    unrelated = root / "unrelated.jsonl"
                    unrelated.write_bytes(b"unrelated bytes")
                    original_unrelated = unrelated.read_bytes()
                    real_write = os.write
                    write_calls = 0

                    def partial_write_then_replace(descriptor, data):
                        nonlocal write_calls
                        write_calls += 1
                        if write_calls == 1:
                            return real_write(descriptor, data[:5])
                        os.replace(path, moved)
                        if replacement_kind == "file":
                            path.write_bytes(b"replacement bytes")
                        else:
                            path.symlink_to(unrelated)
                        raise OSError("append failed after replacement")

                    with patch(
                        "photocut.data.dataset_store.os.write",
                        side_effect=partial_write_then_replace,
                    ):
                        try:
                            append_jsonl(path, {"id": 1})
                        except RuntimeError as exc:
                            caught = exc
                        except OSError:
                            self.fail(
                                "replacement rollback was not reported as "
                                "a consistency failure"
                            )
                        else:
                            self.fail("partial append unexpectedly succeeded")

                    self.assertIn("consistency", str(caught))
                    self.assertIsInstance(caught.__cause__, OSError)
                    self.assertIn(
                        "append failed after replacement", str(caught.__cause__)
                    )
                    self.assertEqual(b"", moved.read_bytes())
                    if replacement_kind == "file":
                        self.assertEqual(b"replacement bytes", path.read_bytes())
                    else:
                        self.assertTrue(path.is_symlink())
                        self.assertEqual(unrelated, path.resolve())
                    self.assertEqual(original_unrelated, unrelated.read_bytes())

    def test_jsonl_write_detects_path_replacement_without_polluting_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            path = root / "events.jsonl"
            moved = root / "moved.jsonl"
            unrelated = root / "unrelated.jsonl"
            append_jsonl(path, {"id": 1})
            original = path.read_bytes()
            unrelated.write_bytes(b"unrelated bytes")
            unrelated_bytes = unrelated.read_bytes()
            real_write = os.write
            swapped = False

            def write_then_replace(descriptor, data):
                nonlocal swapped
                written = real_write(descriptor, data)
                if not swapped:
                    swapped = True
                    os.replace(path, moved)
                    path.symlink_to(unrelated)
                return written

            with patch("photocut.data.dataset_store.os.write", side_effect=write_then_replace):
                try:
                    append_jsonl(path, {"id": 2})
                except RuntimeError as exc:
                    caught = exc
                else:
                    self.fail("append did not detect pathname replacement")

            self.assertIn("consistency", str(caught))
            self.assertEqual(original, moved.read_bytes())
            self.assertTrue(path.is_symlink())
            self.assertEqual(unrelated_bytes, unrelated.read_bytes())

    def test_jsonl_append_uses_verified_fd_after_final_prewrite_check(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            path = root / "events.jsonl"
            moved = root / "moved.jsonl"
            unrelated = root / "unrelated.jsonl"
            append_jsonl(path, {"id": 1})
            original = path.read_bytes()
            unrelated.write_bytes(b"unrelated bytes")
            unrelated_bytes = unrelated.read_bytes()
            real_validate = dataset_store._validate_jsonl_identity
            validation_calls = 0

            def replace_after_final_prewrite_check(target, descriptor):
                nonlocal validation_calls
                identity = real_validate(target, descriptor)
                validation_calls += 1
                if validation_calls == 4:
                    os.replace(path, moved)
                    path.symlink_to(unrelated)
                return identity

            with patch(
                "photocut.data.dataset_store._validate_jsonl_identity",
                side_effect=replace_after_final_prewrite_check,
            ):
                with self.assertRaisesRegex(RuntimeError, "consistency"):
                    append_jsonl(path, {"id": 2})

            self.assertEqual(4, validation_calls)
            self.assertEqual(original, moved.read_bytes())
            self.assertTrue(path.is_symlink())
            self.assertEqual(unrelated_bytes, unrelated.read_bytes())

    def test_jsonl_retries_short_writes_until_the_complete_line_is_durable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"
            real_write = os.write
            write_calls = 0

            def short_write(descriptor, data):
                nonlocal write_calls
                write_calls += 1
                size = max(1, len(data) // 2)
                return real_write(descriptor, data[:size])

            with patch("photocut.data.dataset_store.os.write", side_effect=short_write):
                append_jsonl(path, {"id": 1, "value": "long enough"})

            self.assertGreater(write_calls, 1)
            self.assertEqual(
                [{"id": 1, "value": "long enough"}], load_jsonl(path)
            )

    def test_jsonl_file_lock_serializes_complete_append_transactions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"
            real_write = os.write
            active = 0
            maximum_active = 0
            guard = threading.Lock()
            errors = []

            def slow_write(descriptor, data):
                nonlocal active, maximum_active
                with guard:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.03)
                try:
                    return real_write(descriptor, data)
                finally:
                    with guard:
                        active -= 1

            def append(event):
                try:
                    append_jsonl(path, event)
                except BaseException as exc:
                    errors.append(exc)

            with patch("photocut.data.dataset_store.os.write", side_effect=slow_write):
                threads = [
                    threading.Thread(target=append, args=({"id": identifier},))
                    for identifier in (1, 2)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertEqual([], errors)
            self.assertEqual(1, maximum_active)
            self.assertEqual([1, 2], sorted(event["id"] for event in load_jsonl(path)))

    def test_jsonl_new_file_rollback_does_not_lose_waiting_writer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"
            real_open = os.open
            real_write = os.write
            first_file_open = threading.Event()
            second_file_open = threading.Event()
            first_write_started = threading.Event()
            writer_id = threading.local()
            first_write_calls = 0
            errors = {}

            def coordinated_open(target, flags, *arguments):
                if Path(target) == path and writer_id.value == 2:
                    first_file_open.wait()
                descriptor = real_open(target, flags, *arguments)
                if Path(target) == path:
                    if writer_id.value == 1:
                        first_file_open.set()
                        second_file_open.wait()
                    else:
                        second_file_open.set()
                        first_write_started.wait()
                return descriptor

            def fail_first_event(descriptor, data):
                nonlocal first_write_calls
                payload = bytes(data)
                if writer_id.value == 1:
                    first_write_calls += 1
                    if first_write_calls == 1:
                        first_write_started.set()
                        return real_write(descriptor, payload[:5])
                    raise OSError("first writer failed")
                return real_write(descriptor, data)

            def append(identifier):
                writer_id.value = identifier
                try:
                    append_jsonl(path, {"id": identifier})
                except BaseException as exc:
                    errors[identifier] = exc

            with patch("photocut.data.dataset_store.os.open", side_effect=coordinated_open), patch(
                "photocut.data.dataset_store.os.write", side_effect=fail_first_event
            ):
                threads = [
                    threading.Thread(target=append, args=(identifier,))
                    for identifier in (1, 2)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertIsInstance(errors.get(1), OSError)
            self.assertNotIn(2, errors)
            self.assertEqual([{"id": 2}], load_jsonl(path))

    def test_jsonl_read_rejects_open_time_aba_even_when_original_path_is_restored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            path = root / "events.jsonl"
            parked = root / "parked.jsonl"
            alternate = root / "alternate.jsonl"
            append_jsonl(path, {"id": "original"})
            append_jsonl(alternate, {"id": "alternate"})
            original = path.read_bytes()
            real_open = os.open
            injected = False

            def aba_open(target, flags, *arguments):
                nonlocal injected
                if (
                    Path(target) == path
                    and flags & os.O_ACCMODE == os.O_RDONLY
                    and not injected
                ):
                    injected = True
                    path.rename(parked)
                    alternate.rename(path)
                    descriptor = real_open(target, flags, *arguments)
                    path.rename(alternate)
                    parked.rename(path)
                    return descriptor
                return real_open(target, flags, *arguments)

            with patch("photocut.data.dataset_store.os.open", side_effect=aba_open):
                with self.assertRaisesRegex(ValueError, "changed|identity"):
                    load_jsonl(path)

            self.assertTrue(injected)
            self.assertEqual(original, path.read_bytes())

    def test_jsonl_append_rejects_path_replacement_immediately_after_flock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            path = root / "events.jsonl"
            displaced = root / "displaced.jsonl"
            replacement = root / "replacement.jsonl"
            append_jsonl(path, {"id": 1})
            original = path.read_bytes()
            replacement.write_bytes(b"replacement")
            real_flock = fcntl.flock
            injected = False

            def replace_after_flock(descriptor, operation):
                nonlocal injected
                result = real_flock(descriptor, operation)
                if not injected and operation & fcntl.LOCK_EX:
                    injected = True
                    path.rename(displaced)
                    replacement.rename(path)
                return result

            with patch("photocut.data.dataset_store.fcntl.flock", side_effect=replace_after_flock):
                with self.assertRaisesRegex(ValueError, "changed|identity"):
                    append_jsonl(path, {"id": 2})

            self.assertTrue(injected)
            self.assertEqual(original, displaced.read_bytes())
            self.assertEqual(b"replacement", path.read_bytes())

    def test_jsonl_append_rolls_back_open_fd_without_touching_post_write_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            path = root / "events.jsonl"
            displaced = root / "displaced.jsonl"
            replacement = root / "replacement.jsonl"
            append_jsonl(path, {"id": 1})
            original = path.read_bytes()
            replacement.write_bytes(b"replacement")
            real_fsync = os.fsync
            injected = False

            def replace_after_file_fsync(descriptor):
                nonlocal injected
                result = real_fsync(descriptor)
                info = os.fstat(descriptor)
                if not injected and stat.S_ISREG(info.st_mode):
                    injected = True
                    path.rename(displaced)
                    replacement.rename(path)
                return result

            with patch("photocut.data.dataset_store.os.fsync", side_effect=replace_after_file_fsync):
                with self.assertRaisesRegex(RuntimeError, "consistency"):
                    append_jsonl(path, {"id": 2})

            self.assertTrue(injected)
            self.assertEqual(original, displaced.read_bytes())
            self.assertEqual(b"replacement", path.read_bytes())

    def test_jsonl_failed_new_file_cleanup_never_deletes_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            path = root / "events.jsonl"
            attacker = root / "attacker.jsonl"
            saved_created = root / "saved-created.jsonl"
            attacker.write_bytes(b"attacker replacement")
            real_write = os.write
            real_unlink = Path.unlink
            real_replace = os.replace
            write_calls = 0
            injected = False

            def partial_write_then_fail(descriptor, data):
                nonlocal write_calls
                write_calls += 1
                if write_calls == 1:
                    return real_write(descriptor, data[:5])
                raise OSError("primary append failure")

            def inject_replacement():
                nonlocal injected
                if not injected:
                    injected = True
                    path.rename(saved_created)
                    attacker.rename(path)

            def guarded_unlink(target, *arguments, **kwargs):
                if Path(target) == path:
                    inject_replacement()
                return real_unlink(target, *arguments, **kwargs)

            def guarded_replace(source, destination, *arguments, **kwargs):
                if Path(source) == path:
                    inject_replacement()
                return real_replace(source, destination, *arguments, **kwargs)

            with patch(
                "photocut.data.dataset_store.os.write", side_effect=partial_write_then_fail
            ), patch.object(Path, "unlink", new=guarded_unlink), patch(
                "photocut.data.dataset_store.os.replace", side_effect=guarded_replace
            ):
                with self.assertRaisesRegex(RuntimeError, "consistency") as caught:
                    append_jsonl(path, {"id": 1})

            self.assertTrue(injected)
            self.assertIsInstance(caught.exception.__cause__, OSError)
            self.assertIn(
                "primary append failure", str(caught.exception.__cause__)
            )
            self.assertTrue(path.exists())
            self.assertEqual(b"attacker replacement", path.read_bytes())

    def test_jsonl_rollback_failure_reports_consistency_and_chains_append_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"
            append_jsonl(path, {"id": 1})
            original = path.read_bytes()
            real_write = os.write
            write_calls = 0

            def partial_write_then_fail(descriptor, data):
                nonlocal write_calls
                write_calls += 1
                if write_calls == 1:
                    return real_write(descriptor, data[:5])
                raise OSError("partial write failed")

            with patch(
                "photocut.data.dataset_store.os.write", side_effect=partial_write_then_fail
            ):
                with patch(
                    "photocut.data.dataset_store.os.ftruncate",
                    side_effect=OSError("rollback failed"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "consistency.*rollback failed"
                    ) as caught:
                        append_jsonl(path, {"id": 2})

            self.assertIsInstance(caught.exception.__cause__, OSError)
            self.assertIn("partial write failed", str(caught.exception.__cause__))
            self.assertNotEqual(original, path.read_bytes())

    def test_jsonl_rejects_truncated_non_empty_last_line_with_line_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"
            path.write_text('{"id": 1}\n{"id":', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                load_jsonl(path)

    def test_jsonl_rejects_non_object_values_with_their_line_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "events.jsonl"
            path.write_text('{"id": 1}\n42\n["not", "an event"]\nnull\n', encoding="utf-8")

            for line_number in (2, 3, 4):
                with self.subTest(line_number=line_number):
                    prefix = '{"id": 1}\n' * (line_number - 1)
                    values = {2: "42", 3: '["not", "an event"]', 4: "null"}
                    path.write_text(prefix + values[line_number] + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, f"line {line_number}"):
                        load_jsonl(path)

    def test_atomic_json_fsyncs_new_nested_directories_and_existing_ancestor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            existing_parent = root / "existing"
            existing_parent.mkdir()
            path = existing_parent / "new-one" / "new-two" / "state.json"

            with patch("photocut.data.dataset_store._fsync_directory") as fsync_directory:
                atomic_write_json(path, {"complete": True})

            synced = {call.args[0] for call in fsync_directory.call_args_list}
            self.assertTrue(
                {existing_parent, existing_parent / "new-one", path.parent}.issubset(synced)
            )

    def test_jsonl_fsyncs_new_nested_directories_and_existing_ancestor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            existing_parent = root / "existing"
            existing_parent.mkdir()
            path = existing_parent / "new-one" / "new-two" / "events.jsonl"

            with patch("photocut.data.dataset_store._fsync_directory") as fsync_directory:
                append_jsonl(path, {"id": 1})

            synced = {call.args[0] for call in fsync_directory.call_args_list}
            self.assertTrue(
                {existing_parent, existing_parent / "new-one", path.parent}.issubset(synced)
            )

    def test_fdopen_failure_closes_descriptor_cleans_temp_and_preserves_old_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "state.json"
            atomic_write_json(path, {"version": 1})
            real_mkstemp = tempfile.mkstemp
            captured = {}

            def capture_mkstemp(*args, **kwargs):
                descriptor, temp_name = real_mkstemp(*args, **kwargs)
                captured["descriptor"] = descriptor
                captured["temp_path"] = Path(temp_name)
                return descriptor, temp_name

            with patch("photocut.data.dataset_store.tempfile.mkstemp", side_effect=capture_mkstemp):
                with patch("photocut.data.dataset_store.os.fdopen", side_effect=OSError("stream open failed")):
                    with self.assertRaisesRegex(OSError, "stream open failed"):
                        atomic_write_json(path, {"version": 2})

            self.assertEqual({"version": 1}, json.loads(path.read_text(encoding="utf-8")))
            self.assertFalse(captured["temp_path"].exists())
            with self.assertRaises(OSError):
                os.fstat(captured["descriptor"])

    def test_fdopen_failure_does_not_mask_original_error_when_temp_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir).resolve() / "state.json"

            with patch("photocut.data.dataset_store.os.fdopen", side_effect=OSError("stream open failed")):
                with patch.object(Path, "unlink", side_effect=OSError("cleanup failed")):
                    with self.assertRaisesRegex(OSError, "stream open failed"):
                        atomic_write_json(path, {"version": 2})


class ObjectArchiveTests(unittest.TestCase):
    JPEG_BYTES = b"\xff\xd8\xff\xe0fixture-jpeg\xff\xd9"

    def _object_path(self, store: DatasetStore, source: Path) -> Path:
        digest = sha256_file(source)
        return store.objects_dir / digest[:2] / f"{digest}.jpg"

    def test_identical_bytes_share_one_read_only_object_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            first = root / "first.jpg"
            second = root / "renamed.jpeg"
            first.write_bytes(self.JPEG_BYTES)
            second.write_bytes(self.JPEG_BYTES)
            source_modes = [path.stat().st_mode for path in (first, second)]
            store = DatasetStore(root / ".photocut" / "internal" / "datasets")

            archived_first = store.archive_image(first, image_size=(20, 20))
            archived_second = store.archive_image(second, image_size=(20, 20))

            self.assertEqual(archived_first.image_id, archived_second.image_id)
            self.assertEqual(archived_first.object_path, archived_second.object_path)
            self.assertEqual(1, len(list(store.objects_dir.glob("*/*"))))
            self.assertEqual(self.JPEG_BYTES, archived_first.object_path.read_bytes())
            self.assertEqual(0, archived_first.object_path.stat().st_mode & 0o222)
            self.assertEqual(self.JPEG_BYTES, first.read_bytes())
            self.assertEqual(self.JPEG_BYTES, second.read_bytes())
            self.assertEqual(source_modes, [path.stat().st_mode for path in (first, second)])

    def test_hash_verification_failure_does_not_publish_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "scan.jpg"
            source.write_bytes(self.JPEG_BYTES)
            store = DatasetStore(root / "data")
            real_hash = hashlib.sha256(self.JPEG_BYTES).hexdigest()

            with patch(
                "photocut.data.dataset_store.sha256_file", side_effect=[real_hash, "0" * 64]
            ):
                with self.assertRaisesRegex(IOError, "hash verification"):
                    store.archive_image(source, image_size=(20, 20))

            self.assertEqual([], list(store.objects_dir.glob("*/*")))

    def test_existing_verified_object_is_reused_without_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "scan.jpg"
            source.write_bytes(self.JPEG_BYTES)
            store = DatasetStore(root / "data")
            digest = sha256_file(source)
            object_path = store.objects_dir / digest[:2] / f"{digest}.jpg"
            object_path.parent.mkdir(parents=True)
            object_path.write_bytes(self.JPEG_BYTES)
            os.chmod(object_path, 0o444)
            before = object_path.stat().st_ino

            archived = store.archive_image(source, image_size=(20, 20))

            self.assertEqual(object_path, archived.object_path)
            self.assertEqual(before, object_path.stat().st_ino)
            self.assertEqual(self.JPEG_BYTES, object_path.read_bytes())

    def test_existing_object_must_be_read_only_regular_file_for_archive_and_preflight(self):
        for object_kind in ("symlink", "directory", "writable"):
            for operation in ("archive", "preflight"):
                with self.subTest(object_kind=object_kind, operation=operation):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        root = Path(temp_dir).resolve()
                        source = root / "scan.jpg"
                        source.write_bytes(self.JPEG_BYTES)
                        store = DatasetStore(root / "data")
                        object_path = self._object_path(store, source)
                        object_path.parent.mkdir(parents=True)
                        if object_kind == "symlink":
                            target = root / "target.jpg"
                            target.write_bytes(self.JPEG_BYTES)
                            object_path.symlink_to(target)
                        elif object_kind == "directory":
                            object_path.mkdir()
                        else:
                            object_path.write_bytes(self.JPEG_BYTES)
                            os.chmod(object_path, 0o644)

                        with self.assertRaisesRegex(IOError, "invalid archived object"):
                            if operation == "archive":
                                store.archive_image(source, image_size=(20, 20))
                            else:
                                store.preflight_sources([source])

    def test_existing_object_with_wrong_hash_is_not_reused(self):
        for operation in ("archive", "preflight"):
            with self.subTest(operation=operation):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir).resolve()
                    source = root / "scan.jpg"
                    source.write_bytes(self.JPEG_BYTES)
                    store = DatasetStore(root / "data")
                    object_path = self._object_path(store, source)
                    object_path.parent.mkdir(parents=True)
                    object_path.write_bytes(self.JPEG_BYTES + b"different")
                    os.chmod(object_path, 0o444)

                    with self.assertRaisesRegex(IOError, "hash verification"):
                        if operation == "archive":
                            store.archive_image(source, image_size=(20, 20))
                        else:
                            store.preflight_sources([source])

    def test_existing_object_replaced_after_hash_read_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "scan.jpg"
            source.write_bytes(self.JPEG_BYTES)
            store = DatasetStore(root / "data")
            object_path = self._object_path(store, source)
            object_path.parent.mkdir(parents=True)
            object_path.write_bytes(self.JPEG_BYTES)
            os.chmod(object_path, 0o444)
            real_read = os.read
            replaced = False

            def read_then_replace(descriptor, size):
                nonlocal replaced
                chunk = real_read(descriptor, size)
                if not chunk and not replaced:
                    replacement = root / "replacement.jpg"
                    replacement.write_bytes(self.JPEG_BYTES)
                    os.chmod(replacement, 0o444)
                    os.replace(replacement, object_path)
                    replaced = True
                return chunk

            with patch("photocut.data.dataset_store.os.read", side_effect=read_then_replace):
                with self.assertRaisesRegex(IOError, "invalid archived object"):
                    store.archive_image(source, image_size=(20, 20))

            self.assertTrue(replaced)

    def test_archived_metadata_uses_verified_object_when_source_changes_after_publish(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "scan.jpg"
            source.write_bytes(self.JPEG_BYTES)
            store = DatasetStore(root / "data")
            real_publish = store._publish_object

            def publish_then_change_source(*args):
                object_size = real_publish(*args)
                source.write_bytes(self.JPEG_BYTES + b"source-changed-after-publish")
                return object_size

            with patch.object(store, "_publish_object", side_effect=publish_then_change_source):
                archived = store.archive_image(source, image_size=(20, 20))

            self.assertEqual(sha256_file(archived.object_path), archived.sha256)
            self.assertEqual(archived.object_path.stat().st_size, archived.byte_size)
            self.assertNotEqual(source.stat().st_size, archived.byte_size)

    def test_preflight_rejects_when_free_space_equals_required_plus_reserve(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "scan.jpg"
            source.write_bytes(self.JPEG_BYTES)
            store = DatasetStore(root / "data")
            required = source.stat().st_size

            for free_space in (required, required + ARCHIVE_SPACE_RESERVE_BYTES):
                with self.subTest(free_space=free_space):
                    with patch(
                        "photocut.data.dataset_store.shutil.disk_usage",
                        return_value=MagicMock(free=free_space),
                    ):
                        with self.assertRaisesRegex(OSError, "insufficient disk space"):
                            store.preflight_sources([source])

            with patch(
                "photocut.data.dataset_store.shutil.disk_usage",
                return_value=MagicMock(
                    free=required + ARCHIVE_SPACE_RESERVE_BYTES + 1
                ),
            ):
                store.preflight_sources([source])

    def test_published_object_survives_reported_temp_cleanup_failure_and_retry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "scan.jpg"
            source.write_bytes(self.JPEG_BYTES)
            store = DatasetStore(root / "data")

            with patch.object(Path, "unlink", side_effect=OSError("temp cleanup failed")):
                with self.assertRaisesRegex(OSError, "temp cleanup failed"):
                    store.archive_image(source, image_size=(20, 20))

            object_path = self._object_path(store, source)
            self.assertTrue(object_path.exists())
            retried = store.archive_image(source, image_size=(20, 20))
            self.assertEqual(object_path, retried.object_path)
            self.assertEqual(self.JPEG_BYTES, retried.object_path.read_bytes())

    def test_hash_failure_is_not_masked_when_temp_cleanup_also_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "scan.jpg"
            source.write_bytes(self.JPEG_BYTES)
            store = DatasetStore(root / "data")
            real_hash = hashlib.sha256(self.JPEG_BYTES).hexdigest()

            with patch(
                "photocut.data.dataset_store.sha256_file", side_effect=[real_hash, "0" * 64]
            ):
                with patch.object(Path, "unlink", side_effect=OSError("cleanup failed")):
                    with self.assertRaisesRegex(IOError, "hash verification"):
                        store.archive_image(source, image_size=(20, 20))

    def test_preflight_rejects_insufficient_free_space_before_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "scan.jpg"
            source.write_bytes(self.JPEG_BYTES)
            store = DatasetStore(root / "data")
            fake_usage = MagicMock(free=len(self.JPEG_BYTES) - 1)

            with patch("photocut.data.dataset_store.shutil.disk_usage", return_value=fake_usage):
                with self.assertRaisesRegex(OSError, "insufficient disk space"):
                    store.preflight_sources([source])

    def test_preflight_empty_generator_is_a_no_op_without_store_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store = DatasetStore(root / "data")

            with patch("photocut.data.dataset_store.shutil.disk_usage") as disk_usage:
                store.preflight_sources(source for source in ())

            self.assertFalse(store.root.exists())
            disk_usage.assert_not_called()

    def test_preflight_reuses_verified_object_without_disk_check(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "scan.jpg"
            source.write_bytes(self.JPEG_BYTES)
            store = DatasetStore(root / "data")
            archived = store.archive_image(source, image_size=(20, 20))

            with patch(
                "photocut.data.dataset_store.shutil.disk_usage", return_value=MagicMock(free=0)
            ) as disk_usage:
                store.preflight_sources([source])

            self.assertTrue(archived.object_path.exists())
            disk_usage.assert_not_called()


class BatchLifecycleTests(unittest.TestCase):
    def _archive_fixture(self, root: Path):
        source = root / "scan.jpg"
        source.write_bytes(ObjectArchiveTests.JPEG_BYTES)
        store = DatasetStore(root / "data")
        return store, store.archive_image(source, image_size=(20, 20))

    def test_batch_run_is_progressive_then_atomically_finalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, archived = self._archive_fixture(root)
            paths = store.start_batch(
                runtime={"algorithm_version": "5.2", "parameters": {"inset": None}}
            )

            self.assertRegex(paths.batch_id, r"^\d{8}-\d{6}-[0-9a-f]{8}$")
            self.assertTrue(paths.manifest.exists())
            self.assertTrue(paths.run_in_progress.exists())
            self.assertFalse(paths.production_run.exists())
            in_progress = json.loads(paths.run_in_progress.read_text(encoding="utf-8"))
            self.assertEqual("in_progress", in_progress["status"])
            self.assertEqual("5.2", in_progress["runtime"]["algorithm_version"])
            self.assertEqual({"inset": None}, in_progress["runtime"]["parameters"])
            self.assertTrue(in_progress["runtime"]["git"])
            self.assertIn("python", in_progress["runtime"]["environment"])

            with BatchLock(paths):
                store.register_archived_image(paths, archived)
                store.update_run(
                    paths,
                    {
                        "image_id": archived.image_id,
                        "algorithm_boundary_corners": [[1, 1], [9, 1], [9, 9], [1, 9]],
                        "confidences": [0.8, 0.8, 0.8, 0.8],
                        "success": True,
                    },
                )
                self.assertEqual(
                    [archived.image_id],
                    [entry["image_id"] for entry in json.loads(paths.manifest.read_text())["images"]],
                )
                self.assertTrue(paths.run_in_progress.exists())
                self.assertFalse(paths.production_run.exists())
                store.finalize_run(paths)

            self.assertFalse(paths.run_in_progress.exists())
            self.assertTrue(paths.production_run.exists())
            finalized = json.loads(paths.production_run.read_text(encoding="utf-8"))
            self.assertEqual("complete", finalized["status"])
            self.assertIn("completed_at", finalized)
            self.assertEqual([archived.image_id], [entry["image_id"] for entry in finalized["images"]])

    def test_manifest_register_is_idempotent_and_rejects_conflicting_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, archived = self._archive_fixture(root)
            paths = store.start_batch(runtime={"parameters": {}})

            with BatchLock(paths):
                store.register_archived_image(paths, archived)
                store.register_archived_image(paths, archived)
            self.assertEqual(1, len(json.loads(paths.manifest.read_text())["images"]))

            renamed = root / "renamed.jpg"
            renamed.write_bytes(ObjectArchiveTests.JPEG_BYTES)
            conflicting = store.archive_image(renamed, image_size=(20, 20))
            with BatchLock(paths):
                with self.assertRaisesRegex(ValueError, "conflicting metadata"):
                    store.register_archived_image(paths, conflicting)

    def test_manifest_source_ids_allow_aliases_for_one_content_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, first = self._archive_fixture(root)
            second_path = root / "renamed.jpg"
            second_path.write_bytes(ObjectArchiveTests.JPEG_BYTES)
            second = store.archive_image(second_path, image_size=(20, 20))
            paths = store.start_batch(runtime={"parameters": {}})

            with BatchLock(paths):
                store.register_archived_image(paths, first, source_id="source:first")
                store.register_archived_image(paths, second, source_id="source:second")

            images = json.loads(paths.manifest.read_text())["images"]
            self.assertEqual(1, len(images))
            self.assertEqual(
                ["source:first", "source:second"],
                [source["source_id"] for source in images[0]["sources"]],
            )

    def test_manifest_source_id_cannot_move_to_a_different_content_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, first = self._archive_fixture(root)
            second_path = root / "second.jpg"
            second_path.write_bytes(b"\xff\xd8\xff\xe0different-jpeg\xff\xd9")
            second = store.archive_image(second_path, image_size=(20, 20))
            self.assertNotEqual(first.image_id, second.image_id)
            paths = store.start_batch(runtime={"parameters": {}})

            with BatchLock(paths):
                store.register_archived_image(paths, first, source_id="source:shared")
                original_manifest = paths.manifest.read_bytes()
                with self.assertRaisesRegex(ValueError, "conflicting metadata for source"):
                    store.register_archived_image(
                        paths, second, source_id="source:shared"
                    )

            self.assertEqual(original_manifest, paths.manifest.read_bytes())

    def test_manifest_source_registration_is_idempotent_only_for_full_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, archived = self._archive_fixture(root)
            alias_path = root / "alias.jpg"
            alias_path.write_bytes(ObjectArchiveTests.JPEG_BYTES)
            alias = store.archive_image(alias_path, image_size=(20, 20))
            paths = store.start_batch(runtime={"parameters": {}})

            with BatchLock(paths):
                store.register_archived_image(paths, archived, source_id="source:one")
                original_manifest = paths.manifest.read_bytes()
                store.register_archived_image(paths, archived, source_id="source:one")
                self.assertEqual(original_manifest, paths.manifest.read_bytes())
                with self.assertRaisesRegex(ValueError, "conflicting metadata for source"):
                    store.register_archived_image(paths, alias, source_id="source:one")

            self.assertEqual(original_manifest, paths.manifest.read_bytes())

    def test_manifest_rejects_malformed_sources_before_registering_another_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, first = self._archive_fixture(root)
            second_path = root / "second.jpg"
            second_path.write_bytes(b"\xff\xd8\xff\xe0different-jpeg\xff\xd9")
            second = store.archive_image(second_path, image_size=(20, 20))
            paths = store.start_batch(runtime={"parameters": {}})

            with BatchLock(paths):
                store.register_archived_image(paths, first, source_id="source:first")
                manifest = json.loads(paths.manifest.read_text())
                manifest["images"][0]["sources"] = {"source_id": "source:first"}
                paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
                malformed_manifest = paths.manifest.read_bytes()
                with self.assertRaisesRegex(ValueError, "sources must be a list"):
                    store.register_archived_image(
                        paths, second, source_id="source:second"
                    )

            self.assertEqual(malformed_manifest, paths.manifest.read_bytes())

    def test_manifest_rejects_blank_source_metadata_before_mutation(self):
        cases = (
            ("source_id", ""),
            ("source_id", "  "),
            ("source_filename", ""),
            ("source_filename", "   "),
            ("source_path", ""),
            ("source_path", "\t"),
        )
        for field, value in cases:
            with self.subTest(
                field=field, value=repr(value)
            ), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir).resolve()
                store, first = self._archive_fixture(root)
                alias_path = root / "alias.jpg"
                alias_path.write_bytes(ObjectArchiveTests.JPEG_BYTES)
                alias = store.archive_image(alias_path, image_size=(20, 20))
                paths = store.start_batch(runtime={"parameters": {}})

                with BatchLock(paths):
                    store.register_archived_image(
                        paths, first, source_id="source:first"
                    )
                    manifest = json.loads(paths.manifest.read_text())
                    manifest["images"][0]["sources"][0][field] = value
                    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
                    malformed_manifest = paths.manifest.read_bytes()

                    with self.assertRaisesRegex(
                        ValueError, "invalid source record"
                    ):
                        store.register_archived_image(
                            paths, alias, source_id="source:alias"
                        )

                self.assertEqual(malformed_manifest, paths.manifest.read_bytes())

    def test_source_id_keys_run_results_while_legacy_results_still_key_by_image_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, archived = self._archive_fixture(root)
            paths = store.start_batch(runtime={"parameters": {}})
            with BatchLock(paths):
                store.update_run(paths, {"image_id": archived.image_id, "source_id": "source:first", "success": True})
                store.update_run(paths, {"image_id": archived.image_id, "source_id": "source:second", "success": True})
                with self.assertRaisesRegex(ValueError, "duplicate source_id"):
                    store.update_run(paths, {"image_id": archived.image_id, "source_id": "source:first", "success": True})

            legacy = store.start_batch(runtime={"parameters": {}})
            with BatchLock(legacy):
                store.update_run(legacy, {"image_id": archived.image_id, "success": True})
                with self.assertRaisesRegex(ValueError, "duplicate image_id"):
                    store.update_run(legacy, {"image_id": archived.image_id, "success": True})

    def test_archive_unknown_dimensions_keeps_positive_dimension_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "broken.jpg"
            source.write_bytes(ObjectArchiveTests.JPEG_BYTES)
            store = DatasetStore(root / "data")

            archived = store.archive_image(source, image_size=None)
            self.assertIsNone(archived.width)
            self.assertIsNone(archived.height)
            with self.assertRaisesRegex(ValueError, "positive"):
                store.archive_image(source, image_size=(0, 20))

    def test_manifest_rejects_object_path_outside_dataset_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, archived = self._archive_fixture(root)
            paths = store.start_batch(runtime={"parameters": {}})
            escaped = type(archived)(
                image_id=archived.image_id,
                sha256=archived.sha256,
                object_path=root / "outside.jpg",
                source_path=archived.source_path,
                byte_size=archived.byte_size,
                width=archived.width,
                height=archived.height,
            )

            with BatchLock(paths):
                with self.assertRaisesRegex(ValueError, "outside dataset root"):
                    store.register_archived_image(paths, escaped)

    def test_manifest_revalidates_untrusted_archived_image_identity_and_object_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, archived = self._archive_fixture(root)
            paths = store.start_batch(runtime={"parameters": {}})
            invalid_identity = replace(archived, image_id="sha256:" + "0" * 64)
            invalid_size = replace(archived, byte_size=archived.byte_size + 1)
            with BatchLock(paths):
                with self.assertRaisesRegex(ValueError, "identity"):
                    store.register_archived_image(paths, invalid_identity)
                with self.assertRaisesRegex(ValueError, "byte size"):
                    store.register_archived_image(paths, invalid_size)

    def test_duplicate_run_image_is_rejected_and_final_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, archived = self._archive_fixture(root)
            paths = store.start_batch(runtime={"parameters": {}})
            result = {"image_id": archived.image_id, "success": True}

            with BatchLock(paths):
                store.update_run(paths, result)
                with self.assertRaisesRegex(ValueError, "duplicate image_id"):
                    store.update_run(paths, result)
                store.finalize_run(paths)
            original = paths.production_run.read_bytes()
            paths.run_in_progress.write_text('{"status": "in_progress", "images": []}', encoding="utf-8")

            with BatchLock(paths):
                with self.assertRaisesRegex(RuntimeError, "conflicts"):
                    store.finalize_run(paths)
            self.assertEqual(original, paths.production_run.read_bytes())
            self.assertTrue(paths.run_in_progress.exists())

    def test_batch_ids_do_not_collide_when_created_in_the_same_second(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DatasetStore(Path(temp_dir).resolve() / "data")
            first = store.start_batch(runtime={"parameters": {}})
            second = store.start_batch(runtime={"parameters": {}})

            self.assertNotEqual(first.batch_id, second.batch_id)
            self.assertTrue(first.batch_dir.exists())
            self.assertTrue(second.batch_dir.exists())

    def test_start_batch_never_publishes_partial_batch_on_initialization_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DatasetStore(Path(temp_dir).resolve() / "data")
            real_write = atomic_write_json

            for failing_call in (1, 2):
                calls = 0

                def fail_at_call(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == failing_call:
                        raise OSError("interrupted initialization")
                    return real_write(*args, **kwargs)

                with patch("photocut.data.dataset_store.atomic_write_json", side_effect=fail_at_call):
                    with self.assertRaisesRegex(OSError, "interrupted initialization"):
                        store.start_batch(runtime={"parameters": {}})
                finals = [path for path in store.batches_dir.iterdir() if not path.name.startswith(".")]
                self.assertEqual([], finals)

    def test_start_batch_never_publishes_partial_batch_when_directory_publish_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DatasetStore(Path(temp_dir).resolve() / "data")
            real_replace = os.replace

            def fail_directory_publish(source, destination):
                if Path(source).is_dir():
                    raise OSError("directory publish interrupted")
                return real_replace(source, destination)

            with patch("photocut.data.dataset_store.os.replace", side_effect=fail_directory_publish):
                with self.assertRaisesRegex(OSError, "directory publish interrupted"):
                    store.start_batch(runtime={"parameters": {}})
            finals = [path for path in store.batches_dir.iterdir() if not path.name.startswith(".")]
            self.assertEqual([], finals)

    def test_lifecycle_rejects_forged_paths_and_unlocked_writer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, archived = self._archive_fixture(root)
            paths = store.start_batch(runtime={"parameters": {}})
            forged = replace(paths, manifest=root / "outside.json")
            with self.assertRaisesRegex(ValueError, "invalid batch paths"):
                store.register_archived_image(forged, archived)
            with self.assertRaisesRegex(RuntimeError, "BatchLock"):
                store.register_archived_image(paths, archived)

    def test_non_owner_thread_cannot_mutate_while_lock_is_held(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, archived = self._archive_fixture(root)
            paths = store.start_batch(runtime={"parameters": {}})
            errors = []
            with BatchLock(paths):
                thread = threading.Thread(
                    target=lambda: self._capture_error(
                        errors, store.register_archived_image, paths, archived
                    )
                )
                thread.start()
                thread.join()
            self.assertEqual(1, len(errors))
            self.assertIsInstance(errors[0], RuntimeError)

    def test_two_threads_serially_acquire_lock_without_losing_run_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, _ = self._archive_fixture(root)
            paths = store.start_batch(runtime={"parameters": {}})
            errors = []

            def writer(image_id):
                for _ in range(100):
                    try:
                        with BatchLock(paths):
                            store.update_run(paths, {"image_id": image_id, "success": True})
                        return
                    except RuntimeError as exc:
                        if "already locked" not in str(exc):
                            errors.append(exc)
                            return
                        time.sleep(0.001)
                errors.append(RuntimeError("lock retry exhausted"))

            first = threading.Thread(target=writer, args=("sha256:first",))
            second = threading.Thread(target=writer, args=("sha256:second",))
            first.start()
            second.start()
            first.join()
            second.join()
            self.assertEqual([], errors)
            images = json.loads(paths.run_in_progress.read_text(encoding="utf-8"))["images"]
            self.assertEqual({"sha256:first", "sha256:second"}, {item["image_id"] for item in images})

    @staticmethod
    def _capture_error(errors, function, *args):
        try:
            function(*args)
        except Exception as exc:  # test helper
            errors.append(exc)

    def test_finalize_recovers_after_inprogress_cleanup_failure_without_overwriting_final(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, _ = self._archive_fixture(root)
            paths = store.start_batch(runtime={"parameters": {}})
            with patch("photocut.data.dataset_store._remove_file_durably", side_effect=OSError("unlink failed")):
                with BatchLock(paths):
                    with self.assertRaisesRegex(OSError, "unlink failed"):
                        store.finalize_run(paths)
            self.assertTrue(paths.production_run.exists())
            self.assertTrue(paths.run_in_progress.exists())
            with BatchLock(paths):
                store.finalize_run(paths)
            self.assertFalse(paths.run_in_progress.exists())

    def test_finalize_rejects_invalid_existing_final_and_accepts_valid_idempotent_final(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store, _ = self._archive_fixture(root)
            paths = store.start_batch(runtime={"parameters": {}})
            paths.run_in_progress.unlink()
            paths.production_run.write_text("not-json", encoding="utf-8")
            with BatchLock(paths):
                with self.assertRaisesRegex(ValueError, "invalid finalized run"):
                    store.finalize_run(paths)
            paths.production_run.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": f"run_{paths.batch_id}",
                        "batch_id": paths.batch_id,
                        "status": "complete",
                        "completed_at": "2026-07-24T12:00:00+08:00",
                        "started_at": "now",
                        "runtime": {},
                        "images": [],
                    }
                ),
                encoding="utf-8",
            )
            with BatchLock(paths):
                store.finalize_run(paths)

    def test_final_only_requires_parseable_completed_at(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store = DatasetStore(root / "data")
            paths = store.start_batch(runtime={"parameters": {}})
            paths.run_in_progress.unlink()
            base = {
                "schema_version": 1,
                "run_id": f"run_{paths.batch_id}",
                "batch_id": paths.batch_id,
                "status": "complete",
                "runtime": {},
                "images": [],
            }
            for value in (None, "", 3, "not-a-date"):
                payload = dict(base)
                if value is not None:
                    payload["completed_at"] = value
                paths.production_run.write_text(json.dumps(payload), encoding="utf-8")
                with BatchLock(paths):
                    with self.assertRaisesRegex(ValueError, "invalid finalized run"):
                        store.finalize_run(paths)

    def test_public_finalized_loader_rejects_path_replacement_during_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DatasetStore(Path(temp_dir).resolve() / "data")
            paths = store.start_batch(runtime={"parameters": {}})
            with BatchLock(paths):
                store.finalize_run(paths)

            original = paths.production_run.read_bytes()
            displaced = Path(temp_dir) / "displaced-production-run.json"
            real_load = json.load

            def replace_after_read(stream):
                value = real_load(stream)
                paths.production_run.rename(displaced)
                paths.production_run.write_bytes(original)
                return value

            with BatchLock(paths), patch(
                "photocut.data.dataset_store.json.load", side_effect=replace_after_read
            ):
                with self.assertRaisesRegex(ValueError, "invalid finalized run"):
                    store.load_finalized_run(paths)

            self.assertEqual(original, displaced.read_bytes())

    def test_dataset_root_or_existing_ancestor_symlink_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir).resolve()
            external = base / "external"
            external.mkdir()
            root_link = base / "root-link"
            root_link.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                DatasetStore(root_link / "datasets")
            self.assertEqual([], list(external.iterdir()))

    def test_preflight_and_archive_recheck_newly_created_ancestor_for_symlinks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir).resolve()
            source = base / "scan.jpg"
            source.write_bytes(ObjectArchiveTests.JPEG_BYTES)
            external = base / "external"
            external.mkdir()
            root = base / "missing" / "datasets"
            store = DatasetStore(root)
            (base / "missing").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                store.preflight_sources([source])
            self.assertEqual([], list(external.iterdir()))

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir).resolve()
            source = base / "scan.jpg"
            source.write_bytes(ObjectArchiveTests.JPEG_BYTES)
            external = base / "external"
            external.mkdir()
            root = base / "missing" / "datasets"
            store = DatasetStore(root)
            (base / "missing").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                store.archive_image(source, image_size=(20, 20))
            self.assertEqual([], list(external.iterdir()))

            ancestor = base / "ancestor"
            ancestor.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                DatasetStore(ancestor / "datasets")
            self.assertEqual([], list(external.iterdir()))


class BatchLockTests(unittest.TestCase):
    def test_second_writer_cannot_acquire_batch_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir).resolve() / "batch.lock"
            with BatchLock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with BatchLock(lock_path):
                        self.fail("second lock acquisition unexpectedly succeeded")
            with BatchLock(lock_path):
                self.assertTrue(lock_path.exists())
            self.assertFalse(lock_path.exists())

    def test_preexisting_or_exception_lock_is_never_silently_removed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir).resolve() / "batch.lock"
            lock_path.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already locked"):
                with BatchLock(lock_path):
                    pass
            self.assertEqual("existing", lock_path.read_text(encoding="utf-8"))

    def test_lock_created_by_this_context_is_removed_only_after_clean_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir).resolve() / "nested" / "batch.lock"
            with BatchLock(lock_path):
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                self.assertEqual(os.getpid(), payload["pid"])
                self.assertIn("created_at", payload)
            self.assertFalse(lock_path.exists())

    def test_replacing_lock_path_does_not_delete_replacement_on_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            lock_path = root / "batch.lock"
            replacement = root / "replacement.lock"
            with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                with BatchLock(lock_path):
                    replacement.write_text("belongs-to-someone-else", encoding="utf-8")
                    os.replace(replacement, lock_path)
            self.assertEqual("belongs-to-someone-else", lock_path.read_text(encoding="utf-8"))

    def test_replaced_lock_path_blocks_second_writer_and_old_owner_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            store = DatasetStore(root / "data")
            paths = store.start_batch(runtime={"parameters": {}})
            replacement = root / "replacement.lock"
            with self.assertRaisesRegex(RuntimeError, "lock ownership"):
                with BatchLock(paths):
                    replacement.write_text("replacement", encoding="utf-8")
                    os.replace(replacement, paths.lock)
                    with self.assertRaisesRegex(RuntimeError, "already locked"):
                        with BatchLock(paths):
                            pass
                    store.update_run(paths, {"image_id": "sha256:one", "success": True})
            self.assertEqual("replacement", paths.lock.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
