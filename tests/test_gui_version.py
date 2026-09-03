import unittest
from unittest.mock import Mock

from photocut.confirmation.version import (
    GUI_VERSION,
    OPENCV_GUI_VERSION,
    WEB_GUI_VERSION,
    format_gui_title,
    gui_version_for_frontend,
    gui_identity_lines,
    set_native_gui_title,
    validate_version,
)


class GuiVersionTests(unittest.TestCase):
    def test_frontend_versions_are_independent_and_default_alias_is_web(self):
        self.assertEqual("1.0", OPENCV_GUI_VERSION)
        self.assertEqual("2.0", WEB_GUI_VERSION)
        self.assertEqual(WEB_GUI_VERSION, GUI_VERSION)
        self.assertEqual("1.0", gui_version_for_frontend("opencv"))
        self.assertEqual("2.0", gui_version_for_frontend("web"))

    def test_display_helpers_accept_frontend_specific_version(self):
        self.assertEqual(
            "PhotoCut GUI 2.0 · Algorithm 7.0",
            format_gui_title("7.0", gui_version=WEB_GUI_VERSION),
        )
        self.assertEqual(
            ("GUI: 2.0", "Algorithm: 7.0"),
            gui_identity_lines(
                {"algorithm_version": "7.0"}, gui_version=WEB_GUI_VERSION
            ),
        )
        self.assertEqual("PhotoCut GUI 2.0 · Algorithm 7.0", format_gui_title("7.0"))

    def test_current_gui_version_and_algorithm_titles_are_independent(self):
        self.assertEqual("2.0", GUI_VERSION)
        self.assertEqual(
            "PhotoCut GUI 2.0 · Algorithm 7.0", format_gui_title("7.0")
        )
        self.assertEqual(
            "PhotoCut GUI 2.0 · Algorithm 5.2", format_gui_title("5.2")
        )
        self.assertEqual(
            "PhotoCut GUI 2.0 · Algorithm unknown", format_gui_title(None)
        )

    def test_version_validation_is_strict_and_bounded(self):
        for value in (
            None, True, 1.0, "", " 1.0", "1.0 ", "v1.0", "1", "1.2.3.4",
            "1." + "2" * 31,
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "gui_version"):
                    validate_version(value, "gui_version")
        self.assertEqual("1.0.1", validate_version("1.0.1", "gui_version"))

    def test_auto_is_request_metadata_not_algorithm_version(self):
        lines = gui_identity_lines({
            "algorithm_version": "7.0",
            "detector_requested": "auto",
            "detector_used": "v7",
        })
        self.assertEqual((
            "GUI: 2.0",
            "Algorithm: 7.0",
            "Requested: auto",
            "Used: v7",
        ), lines)

    def test_schema_v1_revision_labels_previous_gui_unknown(self):
        lines = gui_identity_lines(
            {"algorithm_version": "7.0"},
            previous_annotation={"schema_version": 1},
        )
        self.assertIn("Previous GUI: legacy / unknown", lines)

    def test_native_title_failure_returns_fallback_text_without_raising(self):
        cv2_module = Mock()
        cv2_module.setWindowTitle.side_effect = RuntimeError("unsupported")
        title = set_native_gui_title(cv2_module, "PhotoCut", "7.0")
        self.assertEqual("PhotoCut GUI 1.0 · Algorithm 7.0", title)
        cv2_module.setWindowTitle.assert_called_once_with("PhotoCut", title)


if __name__ == "__main__":
    unittest.main()
