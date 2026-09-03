import json
import unittest

from playwright.sync_api import sync_playwright

from photocut.confirmation.controller import ConfirmationSessionController
from photocut.confirmation.web.server import LocalConfirmationServer
from tests.test_confirmation_controller import MemoryBackend, _loaded_item


class ConfirmationWebE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        items = [
            _loaded_item("scan-1.jpg"),
            _loaded_item("scan-2.jpg"),
            _loaded_item("scan-3.jpg"),
        ]
        for index, item in enumerate(items):
            item.image_token = f"current-{index}"
        items[0].entry["risks"] = [
            "outside_background_unverifiable",
            "insufficient_support",
        ]
        self.backend = MemoryBackend(items)
        self.session = ConfirmationSessionController(self.backend, "2.0")
        self.server = LocalConfirmationServer(self.session)
        self.server.start()
        self.context = self.browser.new_context(
            viewport={"width": 1280, "height": 800}
        )

    def tearDown(self):
        self.context.close()
        self.server.close()

    def open_page(self, context=None):
        page = (context or self.context).new_page()
        page.goto(self.server.url)
        page.locator("#workbench[aria-busy='false']").wait_for(state="visible")
        page.locator("#stage-loading").wait_for(state="hidden")
        return page

    def wait_revision(self, page, revision):
        page.locator(f"#workbench[data-revision='{revision}']").wait_for()

    def test_workbench_layout_and_second_tab_readonly(self):
        writer = self.open_page()
        for selector in (
            "#photo-stage", "#inspector", "#actionbar",
            "#candidate-card", "#magnifier",
        ):
            self.assertTrue(writer.locator(selector).is_visible(), selector)
        self.assertTrue(writer.locator("#confirm-button").is_enabled())
        self.assertEqual(4, writer.locator(".corner-crosshair").count())
        self.assertEqual(4, writer.locator(".corner-center").count())
        self.assertEqual("角 1 · 2×", writer.locator("#zoom-label").inner_text())
        self.assertEqual(
            "true",
            writer.locator("#corner-coordinates li").first.get_attribute("aria-current"),
        )
        self.assertIsNone(
            writer.locator("#corner-coordinates li").nth(1).get_attribute("aria-current")
        )
        selected_stroke = writer.locator(
            ".corner-point.is-selected .corner-crosshair"
        ).evaluate("element => getComputedStyle(element).stroke")
        selected_fill = writer.locator(
            ".corner-point.is-selected .corner-crosshair"
        ).evaluate("element => getComputedStyle(element).fill")
        unselected_stroke = writer.locator(
            ".corner-point:not(.is-selected) .corner-crosshair"
        ).first.evaluate("element => getComputedStyle(element).stroke")
        unselected_fill = writer.locator(
            ".corner-point:not(.is-selected) .corner-crosshair"
        ).first.evaluate("element => getComputedStyle(element).fill")
        self.assertNotEqual("none", selected_stroke)
        self.assertEqual("none", unselected_stroke)
        self.assertNotEqual(selected_fill, unselected_fill)
        magnifier_overlay_pixel = writer.locator("#magnifier").evaluate(
            """canvas => Array.from(
                canvas.getContext('2d').getImageData(110, 128, 1, 1).data
            )"""
        )
        self.assertGreater(sum(magnifier_overlay_pixel[:3]), 0)
        center_pixel = writer.locator("#magnifier").evaluate(
            "canvas => Array.from(canvas.getContext('2d').getImageData(128, 128, 1, 1).data)"
        )
        guide_pixel = writer.locator("#magnifier").evaluate(
            "canvas => Array.from(canvas.getContext('2d').getImageData(110, 128, 1, 1).data)"
        )
        self.assertGreater(center_pixel[0], center_pixel[1])
        self.assertGreater(guide_pixel[1], guide_pixel[0])
        self.assertEqual("V8 主候选", writer.locator("#candidate-id").inner_text())
        self.assertIn("建议人工检查", writer.locator("#candidate-status").inner_text())
        self.assertIn("外侧背景不足，请核对边界", writer.locator("#risk-list").inner_text())
        self.assertFalse(writer.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        ))
        for button in writer.locator("button").all():
            box = button.bounding_box()
            self.assertIsNotNone(box)
            self.assertGreaterEqual(box["height"], 40)

        reader = self.open_page()
        self.assertIn("只读", reader.locator("#lease-state").inner_text())
        self.assertTrue(reader.locator("#confirm-button").is_disabled())

        with writer.expect_popup() as popup:
            writer.evaluate("window.open(location.href, '_blank')")
        duplicate = popup.value
        duplicate.locator("#workbench[aria-busy='false']").wait_for()
        self.assertIn("只读", duplicate.locator("#lease-state").inner_text())
        self.assertTrue(duplicate.locator("#confirm-button").is_disabled())
        writer.reload()
        writer.locator("#lease-state[data-writer='true']").wait_for()

    def test_technical_details_are_visible_and_large_audit_is_bounded(self):
        with self.session.lock:
            self.session.current_item.entry["candidate_audit"] *= 30
        page = self.open_page()
        page.locator("#technical-details summary").click()
        page.locator("#details-list dd").first.wait_for(state="visible")

        self.assertIn("/fixtures/scan-1.jpg", page.locator("#details-list").inner_text())
        audit = page.locator("#candidate-audit")
        self.assertTrue(audit.is_visible())
        self.assertLessEqual(audit.bounding_box()["height"], 256.5)
        details_box = page.locator("#details-list").bounding_box()
        inspector_box = page.locator("#inspector").bounding_box()
        self.assertGreaterEqual(details_box["y"], inspector_box["y"])
        self.assertLess(details_box["y"], inspector_box["y"] + inspector_box["height"])

    def test_keys_buttons_drag_zoom_navigation_confirm_and_pause(self):
        page = self.open_page()
        page.locator("#workbench").focus()

        page.keyboard.press("Digit1")
        self.wait_revision(page, 1)
        page.keyboard.press("KeyW")
        self.wait_revision(page, 2)
        self.assertIn("10, 0", page.locator("#corner-coordinates li").first.inner_text())
        page.keyboard.press("ArrowRight")
        self.wait_revision(page, 3)
        self.assertIn("15, 0", page.locator("#corner-coordinates li").first.inner_text())

        page.locator("#reset-button").click()
        self.wait_revision(page, 4)
        page.keyboard.press("KeyC")
        self.wait_revision(page, 5)
        self.assertEqual("备选候选", page.locator("#candidate-id").inner_text())
        page.keyboard.press("KeyB")
        self.wait_revision(page, 6)
        self.assertEqual("v5.2 备选", page.locator("#candidate-id").inner_text())

        page.locator("#zoom-8-button").click()
        self.wait_revision(page, 7)
        self.assertEqual("角 1 · 8×", page.locator("#zoom-label").inner_text())

        page.locator("#reset-button").click()
        self.wait_revision(page, 8)
        point = page.locator(".corner-point").first.bounding_box()
        self.assertIsNotNone(point)
        start_x = point["x"] + point["width"] / 2
        start_y = point["y"] + point["height"] / 2
        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.mouse.move(start_x + 24, start_y + 12, steps=4)
        page.mouse.up()
        self.wait_revision(page, 9)
        self.assertNotIn("10, 10", page.locator("#corner-coordinates li").first.inner_text())
        self.assertTrue(page.locator("#candidate-button").is_disabled())

        page.locator("#reset-button").click()
        self.wait_revision(page, 10)
        page.locator("#next-button").click()
        self.wait_revision(page, 11)
        self.assertEqual("2 / 3", page.locator("#progress").inner_text())
        page.locator("#previous-button").click()
        self.wait_revision(page, 12)
        page.locator("#confirm-button").click()
        self.wait_revision(page, 13)
        self.assertEqual(1, self.backend.commit_count)
        self.assertEqual("2 / 3", page.locator("#progress").inner_text())
        page.locator("#pause-button").click()
        self.wait_revision(page, 14)
        self.assertTrue(page.locator("#confirm-button").is_disabled())

    def test_zoom_and_quit_shortcuts(self):
        page = self.open_page()
        page.locator("#workbench").focus()

        page.keyboard.press("Equal")
        self.wait_revision(page, 1)
        self.assertEqual("角 1 · 4×", page.locator("#zoom-label").inner_text())
        page.keyboard.press("Minus")
        self.wait_revision(page, 2)
        self.assertEqual("角 1 · 2×", page.locator("#zoom-label").inner_text())
        page.keyboard.press("KeyQ")
        self.wait_revision(page, 3)
        self.assertEqual(["scan-1.jpg"], self.backend.checkpoints)
        self.assertTrue(page.locator("#confirm-button").is_disabled())

    def test_magnifier_drag_moves_corner_opposite_to_canvas(self):
        page = self.open_page()
        api_requests = []
        page.on("request", lambda request: api_requests.append(
            (request.url, request.post_data)
        ) if "/api/" in request.url else None)
        box = page.locator("#magnifier").bounding_box()
        self.assertIsNotNone(box)
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        page.mouse.move(center_x, center_y)
        page.mouse.down()
        page.mouse.move(center_x + 32, center_y + 16, steps=4)
        page.wait_for_timeout(50)
        self.assertEqual("0", page.locator("#workbench").get_attribute("data-revision"))
        self.assertEqual([], api_requests)
        page.mouse.up()

        self.wait_revision(page, 1)
        self.assertIn("0, 2", page.locator("#corner-coordinates li").first.inner_text())
        action_requests = [request for request in api_requests if "/api/action" in request[0]]
        self.assertEqual(1, len(action_requests))
        self.assertEqual("set_corner", json.loads(action_requests[0][1])["kind"])
        self.assertEqual(1, len([
            request for request in api_requests if "/api/magnifier?" in request[0]
        ]))

    def test_main_drag_redraws_magnifier_live_without_pointermove_requests(self):
        page = self.open_page()
        api_requests = []
        page.on("request", lambda request: api_requests.append(
            (request.url, request.post_data)
        ) if "/api/" in request.url else None)
        before = page.locator("#magnifier").evaluate("canvas => canvas.toDataURL()")
        point = page.locator(".corner-point").first.bounding_box()
        self.assertIsNotNone(point)
        start_x = point["x"] + point["width"] / 2
        start_y = point["y"] + point["height"] / 2

        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.mouse.move(start_x + 32, start_y + 16, steps=4)
        page.wait_for_timeout(50)

        during = page.locator("#magnifier").evaluate("canvas => canvas.toDataURL()")
        self.assertNotEqual(before, during)
        self.assertEqual("0", page.locator("#workbench").get_attribute("data-revision"))
        self.assertEqual([], api_requests)

        page.mouse.up()
        self.wait_revision(page, 1)
        action_requests = [request for request in api_requests if "/api/action" in request[0]]
        self.assertEqual(1, len(action_requests))
        self.assertEqual("set_corner", json.loads(action_requests[0][1])["kind"])
        self.assertEqual(1, len([
            request for request in api_requests if "/api/magnifier?" in request[0]
        ]))

    def test_long_magnifier_drag_refills_buffer_without_committing_until_release(self):
        page = self.open_page()
        page.keyboard.press("Digit2")
        self.wait_revision(page, 1)
        self.assertEqual("角 2 · 2×", page.locator("#zoom-label").inner_text())
        self.assertEqual(
            "true",
            page.locator("#corner-coordinates li").nth(1).get_attribute("aria-current"),
        )
        api_requests = []
        page.on("request", lambda request: api_requests.append(
            (request.url, request.post_data)
        ) if "/api/" in request.url else None)
        box = page.locator("#magnifier").bounding_box()
        self.assertIsNotNone(box)
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        page.mouse.move(center_x, center_y)
        page.mouse.down()
        page.mouse.move(center_x + 220, center_y, steps=8)
        page.wait_for_timeout(150)

        transient = [request for request in api_requests if "/api/magnifier?" in request[0]]
        self.assertGreaterEqual(len(transient), 1)
        self.assertLessEqual(len(transient), 2)
        self.assertTrue(all("size=768" in request[0] for request in transient))
        self.assertEqual([], [request for request in api_requests if "/api/action" in request[0]])
        self.assertEqual("1", page.locator("#workbench").get_attribute("data-revision"))

        page.mouse.up()
        self.wait_revision(page, 2)
        actions = [request for request in api_requests if "/api/action" in request[0]]
        self.assertEqual(1, len(actions))
        self.assertEqual("set_corner", json.loads(actions[0][1])["kind"])

    def test_supported_viewports_and_high_dpi_have_no_page_overflow(self):
        page = self.open_page()
        for width, height in (
            (320, 720), (375, 760), (414, 800), (768, 900),
            (1280, 800), (1440, 900), (1920, 1080),
        ):
            with self.subTest(width=width, height=height):
                page.set_viewport_size({"width": width, "height": height})
                self.assertFalse(page.evaluate(
                    "document.documentElement.scrollWidth > document.documentElement.clientWidth"
                ))
                self.assertTrue(page.locator("#photo-stage").is_visible())
                self.assertTrue(page.locator("#inspector").is_visible())

        high_dpi = self.browser.new_context(
            viewport={"width": 1920, "height": 1080}, device_scale_factor=2
        )
        try:
            high_dpi_page = self.open_page(high_dpi)
            self.assertEqual(2, high_dpi_page.evaluate("window.devicePixelRatio"))
            self.assertFalse(high_dpi_page.evaluate(
                "document.documentElement.scrollWidth > document.documentElement.clientWidth"
            ))
        finally:
            high_dpi.close()


if __name__ == "__main__":
    unittest.main()
