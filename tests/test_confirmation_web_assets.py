import urllib.error
import urllib.request
import unittest
from pathlib import Path

import numpy as np

from photocut.confirmation.web.server import LocalConfirmationServer, SECURITY_HEADERS


ROOT = Path(__file__).parents[1]
STATIC = ROOT / "photocut" / "confirmation" / "web" / "static"


class FakeSession:
    def __init__(self):
        self.image = np.zeros((80, 100, 3), dtype=np.uint8)

    def snapshot(self):
        return {
            "revision": 0,
            "session": {"status": "active", "readonly": False},
            "progress": {"index": 0, "number": 1, "total": 1},
            "image": {
                "token": "image-token",
                "filename": "scan.jpg",
                "width": 100,
                "height": 80,
            },
            "editor": {
                "corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
                "selected_corner": 0,
                "zoom": 8,
                "candidate": {
                    "id": "candidate-1",
                    "algorithm_version": "8.2",
                    "status": "manual_review",
                    "risks": [],
                },
                "capabilities": {},
                "dirty": False,
            },
            "identity": {"gui": "2.0", "algorithm": "8.2"},
            "storage": {"dirty": False, "formal": False},
            "error": None,
        }

    def technical_details(self):
        return {"path": "/fixtures/scan.jpg", "candidate_audit": []}

    def current_image(self):
        return self.image, "image-token"

    def next_preview_source(self):
        raise RuntimeError("no next image")

    def wait_until_stopped(self):
        return None


def read_static(name):
    return (STATIC / name).read_text(encoding="utf-8")


class ConfirmationWebAssetsTests(unittest.TestCase):
    def test_workbench_has_required_regions_and_no_remote_assets(self):
        html = read_static("index.html")
        for element_id in (
            "topbar",
            "photo-stage",
            "corner-overlay",
            "inspector",
            "magnifier",
            "details-status",
            "actionbar",
            "error-banner",
        ):
            self.assertIn(f'id="{element_id}"', html)
        combined = "\n".join(
            (
                html,
                read_static("app.css"),
                read_static("app.js"),
                read_static("tokens.css"),
            )
        )
        self.assertNotRegex(
            combined,
            r"(?:src|href)=[\"']https?://|data:image|base64",
        )
        stage = html.split('<section id="photo-stage"', 1)[1].split("</section>", 1)[0]
        candidate = html.split('<section id="candidate-card"', 1)[1].split("</section>", 1)[0]
        magnifier = html.split('<section class="inspector-section magnifier-section"', 1)[1].split("</section>", 1)[0]
        self.assertNotIn('id="image-size"', stage)
        self.assertIn('id="image-size"', candidate)
        self.assertIn('id="zoom-2-button"', magnifier)
        self.assertNotIn('id="zoom-16-button"', magnifier)

    def test_css_is_desktop_complete_and_narrow_screen_safe(self):
        css = read_static("app.css")
        self.assertIn("min-width: 80rem", css)
        self.assertIn("grid-template-areas", css)
        self.assertIn("overflow-x: clip", css)
        self.assertIn("stroke-opacity: 0.55", css)
        self.assertIn(".corner-center", css)
        self.assertIn("fill: var(--color-crosshair)", css)
        self.assertIn("fill: var(--color-corner-center)", css)
        self.assertIn("fill-opacity: 1", css)
        self.assertIn("stroke: var(--color-focus)", css)
        self.assertIn("fill: var(--color-crosshair-selected)", css)
        self.assertIn("stroke-width: 5", css)
        self.assertIn("paint-order: stroke fill", css)
        self.assertIn(".coordinates li.is-selected", css)
        self.assertIn("max-height: 16rem", css)
        tokens = read_static("tokens.css")
        self.assertIn("--color-crosshair: #39ff14", tokens)
        self.assertIn("--color-crosshair-selected: #fff500", tokens)
        self.assertIn("--color-corner-center: #ff1a1a", tokens)
        self.assertNotIn("width: 100vw", css)

    def test_readonly_client_bootstrap_stays_small(self):
        script = read_static("app.js")
        for contract in (
            "window.name",
            'api("lease"',
            'api("state"',
            'api("heartbeat"',
            'api("details"',
            "AbortController",
            "URL.createObjectURL",
            "URL.revokeObjectURL",
        ):
            self.assertIn(contract, script)
        self.assertIn("5000", script)
        for contract in (
            "正在加载技术详情…",
            "技术详情加载失败，请重试。",
            'scrollIntoView({ block: "nearest" })',
        ):
            self.assertIn(contract, script)

    def test_keyboard_pointer_and_button_contracts_share_actions(self):
        script = read_static("app.js")
        for binding in (
            "KeyC", "KeyB", "KeyR", "KeyX",
            "KeyQ", "Equal", "Minus",
            "KeyW", "KeyA", "KeyS", "KeyD",
            "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
            "Digit1", "Digit2", "Digit3", "Digit4",
            "Enter", "Space",
        ):
            self.assertIn(binding, script)
        for contract in (
            "sendAction", "preventDefault()", "pointermove",
            "requestAnimationFrame", "setPointerCapture", "set_corner",
            "CLIENT_NAME_PREFIX", "ready", "corner-crosshair",
            "corner-center", "magnifierDrag", "drawMagnifier", "roundTiesToEven",
            "drawMagnifierForMainDrag", "zoom-2-button",
            "MAGNIFIER_BUFFER_SIZE = 768", "MAGNIFIER_REFRESH_MS = 100",
            "MAGNIFIER_CROSSHAIR_GAP = 12", "MAGNIFIER_CROSSHAIR_EXTENT = 30",
            "MAGNIFIER_EDGE_LINE_WIDTH = 3", "MAGNIFIER_CENTER_RADIUS = 1.5",
            "MAIN_CROSSHAIR_GAP_RATIO = 0.7", "MAIN_CENTER_DOT_RATIO = 0.12",
            "CROSSHAIR_APEX_ANGLE_DEGREES = 20", "crosshairHalfBase",
            'getPropertyValue("--color-crosshair")',
            'getPropertyValue("--color-corner-center")',
            'setAttribute("aria-current", "true")',
            "scheduleMagnifierRefresh", "center_x", "center_y",
            "V8 主候选", "自动检测", "外侧背景不足，请核对边界",
        ):
            self.assertIn(contract, script)

    def test_assets_are_only_served_below_the_session_token(self):
        with LocalConfirmationServer(FakeSession()) as server:
            routes = {
                "": "text/html; charset=utf-8",
                "app.css": "text/css; charset=utf-8",
                "app.js": "text/javascript; charset=utf-8",
                "tokens.css": "text/css; charset=utf-8",
            }
            for suffix, content_type in routes.items():
                with self.subTest(suffix=suffix):
                    with urllib.request.urlopen(server.url + suffix, timeout=2) as response:
                        self.assertEqual(200, response.status)
                        self.assertEqual(content_type, response.headers["Content-Type"])
                        self.assertGreater(int(response.headers["Content-Length"]), 0)
                        for name, value in SECURITY_HEADERS:
                            self.assertEqual(value, response.headers[name])
            for url in (
                server.origin + "/session/wrong/",
                server.url + "../tokens.css",
                server.url + "missing.js",
            ):
                with self.subTest(url=url):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(url, timeout=2)
                    self.assertEqual(404, caught.exception.code)


if __name__ == "__main__":
    unittest.main()
