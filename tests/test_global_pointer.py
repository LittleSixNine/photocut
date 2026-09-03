import math
import unittest
from types import SimpleNamespace

from photocut.confirmation.pointer import (
    MacOSGlobalPointerReader,
    PointerReadError,
    PointerSample,
    create_global_pointer_reader,
)


class FakeNative:
    def __init__(self, x=120.0, y=80.0, down=True, *, location_error=None, button_error=None):
        self.x = x
        self.y = y
        self.down = down
        self.location_error = location_error
        self.button_error = button_error
        self.releases = 0

    def create_event(self):
        return object()

    def get_location(self, _event):
        if self.location_error:
            raise self.location_error
        return SimpleNamespace(x=self.x, y=self.y)

    def button_down(self):
        if self.button_error:
            raise self.button_error
        return self.down

    def release(self, _event):
        self.releases += 1


class GlobalPointerTests(unittest.TestCase):
    def test_reader_returns_position_and_left_button_state(self):
        native = FakeNative()
        reader = MacOSGlobalPointerReader(native=native)
        self.assertEqual(PointerSample(120.0, 80.0, True), reader.sample())
        self.assertEqual(1, native.releases)

    def test_reader_rejects_nonfinite_position_and_releases_event(self):
        native = FakeNative(x=math.nan)
        reader = MacOSGlobalPointerReader(native=native)
        with self.assertRaises(PointerReadError):
            reader.sample()
        self.assertEqual(1, native.releases)

    def test_reader_releases_event_when_location_or_button_fails(self):
        for native in (
            FakeNative(location_error=RuntimeError("location")),
            FakeNative(button_error=RuntimeError("button")),
        ):
            with self.subTest(native=native):
                reader = MacOSGlobalPointerReader(native=native)
                with self.assertRaises(PointerReadError):
                    reader.sample()
                self.assertEqual(1, native.releases)

    def test_reader_release_failure_is_not_reported_as_success(self):
        class ReleaseBroken(FakeNative):
            def release(self, _event):
                self.releases += 1
                raise PointerReadError("release")

        native = ReleaseBroken()
        reader = MacOSGlobalPointerReader(native=native)
        with self.assertRaises(PointerReadError):
            reader.sample()
        self.assertEqual(1, native.releases)

    def test_factory_returns_none_when_platform_is_not_macos(self):
        self.assertIsNone(create_global_pointer_reader(platform="linux"))


if __name__ == "__main__":
    unittest.main()
