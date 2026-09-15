import unittest

import cv2
import numpy as np

from src.depth_engine import DepthEngine, TemporalDepthFilter


class DepthStabilityTests(unittest.TestCase):
    def test_regularization_flattens_object_interior_and_preserves_boundary(self):
        rng = np.random.default_rng(7)
        rgb = np.zeros((120, 180, 3), dtype=np.uint8)
        rgb[:, :90] = (25, 45, 80)
        rgb[:, 90:] = (230, 180, 40)
        depth = np.empty((120, 180), dtype=np.float32)
        depth[:, :90] = 0.25 + rng.normal(0, 0.08, (120, 90))
        depth[:, 90:] = 0.78 + rng.normal(0, 0.08, (120, 90))
        depth = np.clip(depth, 0, 1)

        stable = DepthEngine.regularize_depth(rgb, depth)

        self.assertLess(stable[:, 20:70].std(), depth[:, 20:70].std() * 0.55)
        self.assertLess(stable[:, 110:160].std(), depth[:, 110:160].std() * 0.55)
        left = stable[:, 60:80].mean()
        right = stable[:, 100:120].mean()
        self.assertGreater(right - left, 0.35)

    def test_scene_cut_resets_motion_aligned_history(self):
        filt = TemporalDepthFilter(alpha=0.8)
        first_rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        second_rgb = np.full((64, 96, 3), 255, dtype=np.uint8)
        filt.filter(np.full((64, 96), 0.1, np.float32), rgb=first_rgb)
        current = np.full((64, 96), 0.9, np.float32)
        result = filt.filter(current, rgb=second_rgb)
        np.testing.assert_allclose(result, current)


if __name__ == '__main__':
    unittest.main()
