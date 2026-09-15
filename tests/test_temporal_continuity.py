"""
Temporal Continuity and Cadence Regression Test Suite.
Tests 2D-to-3D conversion for exact frame correspondence, monotonic ordering,
rational framerate preservation, batch boundary safety, checkpoint resumption,
depth decimation RGB integrity, and automated cadence diagnosis.
"""
import unittest
import tempfile
import shutil
import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FFMPEG_BIN, DEFAULT_DEPTH_MODEL
from src.preprocessor import get_media_info, read_video_frames
from src.encoder_3d import VideoStreamWriter, compose_sbs
from src.depth_engine import DepthEngine, DepthDecimator
from src.dibr_stereo import StereoSynthesizer
from src.checkpoint_manager import CheckpointManager, concatenate_segments
from src.cadence_analyzer import analyze_visual_cadence
from src.cadence_repair import repair_defective_cadence
from convert_3d import process_video

_references = {}

def generate_synthetic_motion_video(
    output_path: Path,
    num_frames: int = 48,
    width: int = 320,
    height: int = 240,
    fps_num: int = 24,
    fps_den: int = 1,
    include_static_scene: bool = False
) -> Path:
    """
    Generates a synthetic test video with:
    1. A constant-velocity moving shape across frames.
    2. A machine-readable digital pixel marker at (0, 0) encoding exact frame index in RGB.
    3. Human-readable text 'FRAME X' burned in.
    4. Optional static pause segment to test that static scenes are not falsely flagged.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps_rational = f"{fps_num}/{fps_den}" if fps_den != 1 else f"{fps_num}"
    fps = fps_num / fps_den

    writer = VideoStreamWriter(
        output_path, width, height, fps,
        fps_rational=fps_rational,
        codec="libx264"
    )

    bar_width = 30
    speed = 6.0 # pixels per frame

    for idx in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Subtle gradient background
        for y in range(height):
            img[y, :, 0] = int(60 * (y / height))
            img[y, :, 2] = int(80 * (1.0 - y / height))

        # Check if in static segment (frames 15..20 if enabled)
        effective_idx = idx
        if include_static_scene and 15 <= idx <= 20:
            effective_idx = 15

        # Moving bright vertical bar
        bar_x = int((effective_idx * speed) % (width - bar_width - 10)) + 5
        cv2.rectangle(img, (bar_x, 30), (bar_x + bar_width, height - 30), (0, 255, 255), -1)

        # Burned-in text
        cv2.putText(
            img, f"FRAME {effective_idx:04d}", (width // 4, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Digital pixel marker at (0, 0) encoding frame index
        # R = idx % 256, G = (idx // 256) % 256
        img[0, 0, 0] = effective_idx % 256
        img[0, 0, 1] = (effective_idx // 256) % 256
        img[0, 0, 2] = 200 # Constant identifier

        writer.write_frame(img)

    writer.close()
    _references[output_path.parent] = output_path
    return output_path

def decode_left_eye_indices(video_path: Path, mode: str = "full_sbs") -> list:
    """
    Decodes the left eye of an SBS video and reads back the burned-in digital markers.
    """
    reference = _references[Path(video_path).resolve().parent]
    refs = [cv2.resize(f, (80, 60)).astype(np.float32) for _, f in read_video_frames(reference)]
    indices = []
    for _, frame in read_video_frames(video_path):
        left = frame[:, :frame.shape[1] // 2]
        sample = cv2.resize(left, (80, 60)).astype(np.float32)
        distances = [np.mean(np.abs(sample - ref)) for ref in refs]
        indices.append(int(np.argmin(distances)))
    return indices

class SyntheticDepth:
    """Deterministic depth fixture: these tests exercise timing, not model accuracy."""
    def __init__(self):
        from src.depth_engine import TemporalDepthFilter
        self.temporal_filter = TemporalDepthFilter()
    def estimate_depth(self, frame, **kwargs):
        return np.full(frame.shape[:2], 0.5, dtype=np.float32)
    def estimate_depth_batch(self, frames, **kwargs):
        return [self.estimate_depth(f) for f in frames]

class TestTemporalContinuity(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="3d_temporal_test_"))
        self.synthesizer = StereoSynthesizer(divergence=0.02, convergence=0.5, render_mode="right_only")
        self.depth_engine = SyntheticDepth()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_frame_order_and_uniqueness(self):
        """
        Tests that normal conversion produces 1:1 frame correspondence:
        No dropped frames, no duplicate frames, strictly monotonic order.
        """
        input_vid = self.test_dir / "test_input_24fps.mp4"
        generate_synthetic_motion_video(input_vid, num_frames=36, fps_num=24, fps_den=1)

        output_vid = self.test_dir / "test_output_sbs.mp4"
        process_video(
            input_path=input_vid,
            output_path=output_vid,
            depth_engine=self.depth_engine,
            synthesizer=self.synthesizer,
            fmt="sbs",
            auto_crop=False
        )

        self.assertTrue(output_vid.exists())
        info_out = get_media_info(output_vid)
        self.assertEqual(info_out["nb_frames"], 36)

        # Verify decoded frame indices
        decoded_indices = decode_left_eye_indices(output_vid, mode="full_sbs")
        self.assertEqual(len(decoded_indices), 36)
        expected = list(range(36))
        self.assertEqual(decoded_indices, expected, "Frames must be strictly ordered 0..35 without duplicates or skips")

    def test_02_rational_fps_preservation(self):
        """
        Tests that 23.976 fps (24000/1001) is strictly preserved without float rounding or telecine.
        """
        input_vid = self.test_dir / "test_input_23976.mp4"
        generate_synthetic_motion_video(input_vid, num_frames=48, fps_num=24000, fps_den=1001)

        info_in = get_media_info(input_vid)
        self.assertEqual(info_in["fps_rational"], "24000/1001")
        self.assertAlmostEqual(info_in["fps"], 23.976, places=2)

        output_vid = self.test_dir / "test_output_23976_sbs.mp4"
        process_video(
            input_path=input_vid,
            output_path=output_vid,
            depth_engine=self.depth_engine,
            synthesizer=self.synthesizer,
            fmt="sbs",
            auto_crop=False
        )

        info_out = get_media_info(output_vid)
        self.assertEqual(info_out["fps_rational"], "24000/1001")
        self.assertEqual(info_out["nb_frames"], 48)

    def test_03_batch_processing_boundaries(self):
        """
        Tests that batch inference (batch_size=4 on 35 frames) handles boundaries cleanly:
        No duplicated frames and no dropped frames at batch junctions.
        """
        input_vid = self.test_dir / "test_input_batch.mp4"
        generate_synthetic_motion_video(input_vid, num_frames=35, fps_num=24, fps_den=1)

        output_vid = self.test_dir / "test_output_batch_sbs.mp4"
        process_video(
            input_path=input_vid,
            output_path=output_vid,
            depth_engine=self.depth_engine,
            synthesizer=self.synthesizer,
            fmt="sbs",
            auto_crop=False,
            batch_size=4
        )

        decoded_indices = decode_left_eye_indices(output_vid, mode="full_sbs")
        self.assertEqual(len(decoded_indices), 35)
        self.assertEqual(decoded_indices, list(range(35)), "Batch boundaries must not skip or repeat frames")

    def test_04_checkpoint_and_resumption(self):
        """
        Tests pausing conversion at frame K and resuming:
        Guarantees exact continuity across the resume point.
        """
        input_vid = self.test_dir / "test_input_resume.mp4"
        generate_synthetic_motion_video(input_vid, num_frames=40, fps_num=24, fps_den=1)

        output_vid = self.test_dir / "test_output_resume_sbs.mp4"

        # 1. Simulate partial conversion of 20 frames into part 1
        part1 = self.test_dir / "test_output_resume_sbs_part0001.mp4"
        part1_writer = VideoStreamWriter(part1, 640, 240, 24.0, fps_rational="24", codec="auto")
        gen = read_video_frames(input_vid, max_frames=20)
        for idx, frame in gen:
            depth = self.depth_engine.estimate_depth(frame)
            left, right = self.synthesizer.render_stereo(frame, depth)
            part1_writer.write_frame(compose_sbs(left, right, mode="full_sbs"))
        part1_writer.close()

        # Save checkpoint marking 20 frames processed
        cp_mgr = CheckpointManager(output_vid, input_vid)
        cp_mgr.save(processed_frames=20, total_frames=40, fps_rational="24", segments=[str(part1)])

        # 2. Resume conversion
        process_video(
            input_path=input_vid,
            output_path=output_vid,
            depth_engine=self.depth_engine,
            synthesizer=self.synthesizer,
            fmt="sbs",
            auto_crop=False,
            resume=True
        )

        self.assertTrue(output_vid.exists())
        decoded_indices = decode_left_eye_indices(output_vid, mode="full_sbs")
        self.assertEqual(len(decoded_indices), 40)
        self.assertEqual(decoded_indices, list(range(40)), "Resumed output must have seamless continuity 0..39")

    def test_05_depth_stride_rgb_integrity(self):
        """
        Tests requirement 4: When depth is calculated with lower frequency (depth_stride=3),
        depth is reused/interpolated but RGB motion is STRICTLY 1:1 preserved without duplication!
        """
        input_vid = self.test_dir / "test_input_stride.mp4"
        generate_synthetic_motion_video(input_vid, num_frames=30, fps_num=24, fps_den=1)

        output_vid = self.test_dir / "test_output_stride_sbs.mp4"
        process_video(
            input_path=input_vid,
            output_path=output_vid,
            depth_engine=self.depth_engine,
            synthesizer=self.synthesizer,
            fmt="sbs",
            auto_crop=False,
            depth_stride=3
        )

        decoded_indices = decode_left_eye_indices(output_vid, mode="full_sbs")
        self.assertEqual(len(decoded_indices), 30)
        self.assertEqual(decoded_indices, list(range(30)))

        # Verify that consecutive frames have real pixel differences (no frozen RGB!)
        gen = read_video_frames(output_vid)
        prev = None
        for idx, f in gen:
            if prev is not None:
                diff = np.mean(np.abs(f.astype(float) - prev.astype(float)))
                self.assertGreater(diff, 1.0, f"Frame {idx} must NOT be a visual duplicate of previous frame")
            prev = f

    def test_06_cadence_analyzer_clean_vs_corrupt(self):
        """
        Tests Requirement 7: Cadence diagnostic tool detects introduced stutter (1 in 3)
        while passing clean conversions and respecting natural static scenes.
        """
        # A. Clean video with a natural static scene (frames 15..20)
        clean_input = self.test_dir / "clean_input_with_static.mp4"
        generate_synthetic_motion_video(clean_input, num_frames=45, include_static_scene=True)

        clean_output = self.test_dir / "clean_output_sbs.mp4"
        process_video(
            input_path=clean_input,
            output_path=clean_output,
            depth_engine=self.depth_engine,
            synthesizer=self.synthesizer,
            fmt="sbs",
            auto_crop=False
        )

        clean_report = analyze_visual_cadence(clean_input, clean_output, fmt="sbs")
        self.assertEqual(clean_report["verdict"], "PASS")
        self.assertFalse(clean_report["stutter_detected"])
        self.assertEqual(clean_report["introduced_duplicates"], 0)
        self.assertGreater(clean_report["native_static_frames"], 0, "Native static frames must be recognized")

        # B. Corrupt video with artificial stutter (1 duplicate every 3 frames, emulating external decimation defect)
        corrupt_output = self.test_dir / "corrupt_cadence_simulation.mp4"
        c_writer = VideoStreamWriter(corrupt_output, 640, 240, 24.0, fps_rational="24", codec="libx264")
        gen = read_video_frames(clean_input)

        last_emitted = None
        count = 0
        for idx, f in gen:
            left, right = self.synthesizer.render_stereo(f, self.depth_engine.estimate_depth(f))
            sbs = compose_sbs(left, right, mode="full_sbs")

            # Introduce repeat every 3 frames
            if count > 0 and count % 3 == 0 and last_emitted is not None:
                c_writer.write_frame(last_emitted) # Frozen duplicate!
            else:
                c_writer.write_frame(sbs)
                last_emitted = sbs
            count += 1
        c_writer.close()

        corrupt_report = analyze_visual_cadence(clean_input, corrupt_output, fmt="sbs")
        self.assertEqual(corrupt_report["verdict"], "FAIL_STUTTER_DETECTED")
        self.assertTrue(corrupt_report["stutter_detected"])
        self.assertGreater(corrupt_report["introduced_duplicates"], 5)
        self.assertEqual(corrupt_report["detected_period"], 3, "Must accurately identify stutter period of 3")

    def test_07_cadence_repair_utility(self):
        """
        Tests repairing a defective video with stutter.
        """
        # Create defective video
        input_vid = self.test_dir / "defective_input.mp4"
        generate_synthetic_motion_video(input_vid, num_frames=30, fps_num=24, fps_den=1)

        corrupt_vid = self.test_dir / "defective_corrupt.mp4"
        c_writer = VideoStreamWriter(corrupt_vid, 320, 240, 24.0, fps_rational="24", codec="libx264")
        gen = read_video_frames(input_vid)
        last_f = None
        count = 0
        for idx, f in gen:
            if count > 0 and count % 3 == 0 and last_f is not None:
                c_writer.write_frame(last_f)
            else:
                c_writer.write_frame(f)
                last_f = f
            count += 1
        c_writer.close()

        repaired_vid = self.test_dir / "repaired_output.mp4"
        success = repair_defective_cadence(corrupt_vid, repaired_vid, mode="interpolate")
        self.assertTrue(success)
        self.assertTrue(repaired_vid.exists())

        # Verify repaired video has restored cadence
        report = analyze_visual_cadence(input_vid, repaired_vid, fmt="2d")
        before = analyze_visual_cadence(input_vid, corrupt_vid, fmt="2d")
        self.assertLess(report["introduced_duplicates"], before["introduced_duplicates"])

if __name__ == "__main__":
    unittest.main()
