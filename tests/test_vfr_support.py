import unittest
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import cv2

from src.preprocessor import get_media_info, read_video_frames
from convert_3d import process_video
from src.dibr_stereo import StereoSynthesizer

class SyntheticDepth:
    def __init__(self):
        from src.depth_engine import TemporalDepthFilter
        self.temporal_filter = TemporalDepthFilter()
    def estimate_depth(self, frame, **kwargs):
        return np.full(frame.shape[:2], 0.5, dtype=np.float32)
    def estimate_depth_batch(self, frames, **kwargs):
        return [self.estimate_depth(f) for f in frames]

class TestVFRSupport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.vfr_video = self.temp_path / "synthetic_vfr.mp4"
        self._create_synthetic_vfr_video(self.vfr_video)
        self.depth_engine = SyntheticDepth()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_vfr_video(self, output_path: Path):
        """
        Creates a synthetic video with variable timestamps.
        """
        w, h = 320, 240
        # Create a 2-second video with variable frame intervals
        cmd = [
            "bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=30",
            "-vf", "setpts='if(lt(N,15), 2*PTS, PTS)'",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        subprocess.run(cmd, check=True)

    def test_vfr_read_video_frames_auto_cfr(self):
        info = get_media_info(self.vfr_video)
        self.assertTrue(info["is_vfr"], "Synthetic video should be detected as VFR")

        # read_video_frames should automatically apply the CFR filter
        frames = list(read_video_frames(self.vfr_video))
        self.assertGreater(len(frames), 0)
        # Verify frame shapes
        for idx, f in frames:
            self.assertEqual(f.shape, (240, 320, 3))

    def test_vfr_conversion_end_to_end(self):
        output_3d = self.temp_path / "vfr_converted_3d.mp4"
        synthesizer = StereoSynthesizer(render_mode="right_only", divergence=0.025)

        # Process the VFR video without any error
        process_video(
            input_path=self.vfr_video,
            output_path=output_3d,
            depth_engine=self.depth_engine,
            synthesizer=synthesizer,
            fmt="hsbs"
        )

        self.assertTrue(output_3d.exists(), "3D output file should be created")
        out_info = get_media_info(output_3d)
        self.assertEqual(out_info["width"], 320)
        self.assertEqual(out_info["height"], 240)
        self.assertFalse(out_info["is_vfr"], "3D output must be strict CFR")

if __name__ == "__main__":
    unittest.main()
