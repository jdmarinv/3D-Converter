import unittest

import cv2
import numpy as np

from src.encoder_3d import compose_sbs
from src.stereo_artifact_detector import frame_artifact_metrics


class TestStereoArtifactDetector(unittest.TestCase):
    def test_broken_edges_score_above_clean_shift(self):
        source = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(source, "LEGENDARY", (25, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    1.35, (235, 220, 180), 3, cv2.LINE_AA)
        clean = np.roll(source, 5, axis=1)
        damaged = clean.copy()
        damaged[:, 125:165] = cv2.GaussianBlur(damaged[:, 125:165], (21, 21), 0)
        clean_sbs = compose_sbs(source, clean, mode="half_sbs")
        damaged_sbs = compose_sbs(source, damaged, mode="half_sbs")
        clean_score = frame_artifact_metrics(source, clean_sbs)["score"]
        damaged_score = frame_artifact_metrics(source, damaged_sbs)["score"]
        self.assertGreater(damaged_score, clean_score + 1.0)


if __name__ == "__main__":
    unittest.main()
