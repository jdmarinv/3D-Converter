import unittest

import cv2
import numpy as np

from src.encoder_3d import compose_sbs
from src.stereo_artifact_repair import one_sided_signature, parse_ranges


class TestStereoArtifactRepair(unittest.TestCase):
    def setUp(self):
        self.source = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.rectangle(self.source, (40, 35), (130, 150), (240, 90, 20), -1)
        cv2.circle(self.source, (230, 90), 45, (20, 220, 180), -1)
        self.shifted = np.roll(self.source, 12, axis=1)

    def test_detects_left_pristine_right_only_conversion(self):
        packed = compose_sbs(self.source, self.shifted, mode="half_sbs")
        detected, eye, metrics = one_sided_signature(self.source, packed)
        self.assertTrue(detected, metrics)
        self.assertEqual(eye, "left")

    def test_does_not_flag_symmetric_both_eye_conversion(self):
        left = np.roll(self.source, -6, axis=1)
        right = np.roll(self.source, 6, axis=1)
        packed = compose_sbs(left, right, mode="half_sbs")
        detected, eye, _ = one_sided_signature(self.source, packed)
        self.assertFalse(detected)
        self.assertEqual(eye, "none")

    def test_parses_multiple_second_ranges(self):
        self.assertEqual(parse_ranges("1.5-2.0,10-12", 24.0), [(36, 48), (240, 288)])


if __name__ == "__main__":
    unittest.main()
