import unittest

from photocut.confirmation.model import ConfirmationState
from photocut.algorithms.v7.gui_model import V7ConfirmationViewModel
from photocut.algorithms.v7.parameters import V7Parameters
from photocut.algorithms.v7.types import DetectionIdentity, DetectionResult, DetectionStatus


class GuiModelTests(unittest.TestCase):
    def result(self, status=DetectionStatus.V7_RECOMMENDED):
        params = V7Parameters()
        identity = DetectionIdentity("d1", "image", "identity", "7.0", params.sha256(), "safe")
        return DetectionResult(
            identity=identity, status=status,
            corners=((2., 2.), (18., 2.), (18., 18.), (2., 18.)),
            alternate_corners=((3., 3.), (17., 3.), (17., 17.), (3., 17.)),
            overall_confidence=.91, edge_confidences=(.9, .8, .7, .6),
            corner_confidences=(.9, .8, .7, .6), risks=("weak_edge",),
        )

    def model(self):
        return V7ConfirmationViewModel(self.result(), image_size=(20, 20), v52_corners=((1, 1), (19, 1), (19, 19), (1, 19)))

    def test_risk_labels_and_per_edge_colors(self):
        model = self.model()
        self.assertTrue(model.risk_labels)
        self.assertEqual(4, len(model.edge_colors))
        self.assertEqual("recommended", model.status_label)

    def test_a_and_b_switch_candidates_before_drag(self):
        model = self.model()
        self.assertEqual("top1", model.selection)
        model.toggle_alternate()
        self.assertEqual("alternate", model.selection)
        self.assertEqual([3, 3], model.algorithm_corners[0])
        model.toggle_v52()
        self.assertEqual("v52", model.selection)
        model.toggle_v52()
        self.assertEqual("top1", model.selection)

    def test_drag_classifies_and_blocks_candidate_toggle_until_reset(self):
        model = self.model()
        model.select_corner(0)
        model.move_selected(1, 0)
        self.assertEqual("dragged", model.operation)
        with self.assertRaises(ValueError):
            model.toggle_alternate()
        model.reset()
        self.assertEqual("top1", model.selection)
        self.assertEqual("direct_top1", model.operation)

    def test_error_and_no_primary_cannot_confirm_but_can_skip(self):
        for status in (DetectionStatus.ERROR, DetectionStatus.NO_PRIMARY_PHOTO, DetectionStatus.CANCELLED):
            model = V7ConfirmationViewModel(self.result(status), image_size=(20, 20))
            self.assertFalse(model.can_confirm)
            with self.assertRaises(ValueError):
                model.confirm()
            self.assertEqual("skipped", model.skip("no_primary"))

    def test_confirm_returns_operation_and_integer_corners(self):
        model = self.model()
        operation, corners = model.confirm()
        self.assertEqual("direct_top1", operation)
        self.assertEqual([[2, 2], [18, 2], [18, 18], [2, 18]], corners)


if __name__ == "__main__":
    unittest.main()
