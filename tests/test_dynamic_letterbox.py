import unittest

import numpy as np

from src.dibr_stereo import StereoSynthesizer
from src.preprocessor import (
    detect_active_image_bounds,
    neutralize_bar_depth,
    prepare_depth_input,
)


class DynamicLetterboxTests(unittest.TestCase):
    def make_letterboxed_frame(self):
        rng = np.random.default_rng(7)
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[20:80] = rng.integers(40, 240, size=(60, 160, 3), dtype=np.uint8)
        return frame

    def test_detects_letterbox_without_changing_canvas(self):
        frame = self.make_letterboxed_frame()
        prepared, bounds = prepare_depth_input(frame)
        self.assertEqual(bounds, (20, 80, 0, 160))
        self.assertEqual(prepared.shape, frame.shape)
        self.assertTrue(np.array_equal(prepared[0], prepared[20]))
        self.assertTrue(np.array_equal(prepared[-1], prepared[79]))

    def test_full_black_fade_is_not_treated_as_letterbox(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        self.assertEqual(detect_active_image_bounds(frame), (0, 100, 0, 160))

    def test_depth_bars_are_put_on_convergence_plane(self):
        depth = np.ones((100, 160), dtype=np.float32)
        masked = neutralize_bar_depth(depth, (20, 80, 0, 160), 0.5)
        self.assertTrue(np.all(masked[:20] == 0.5))
        self.assertTrue(np.all(masked[80:] == 0.5))
        self.assertTrue(np.all(masked[20:80] == 1.0))

    def test_cinematic_style_cannot_shift_protected_bars(self):
        frame = self.make_letterboxed_frame()
        depth = np.ones((100, 160), dtype=np.float32)
        mask = np.zeros((100, 160), dtype=bool)
        mask[:20] = True
        mask[80:] = True
        synth = StereoSynthesizer(
            divergence=0.05,
            convergence=0.5,
            render_mode="both",
            style_3d="cinematic",
        )
        left, right = synth.render_stereo(frame, depth, zero_disparity_mask=mask)
        self.assertTrue(np.array_equal(left[:20], frame[:20]))
        self.assertTrue(np.array_equal(right[:20], frame[:20]))
        self.assertTrue(np.array_equal(left[80:], frame[80:]))
        self.assertTrue(np.array_equal(right[80:], frame[80:]))


if __name__ == "__main__":
    unittest.main()
