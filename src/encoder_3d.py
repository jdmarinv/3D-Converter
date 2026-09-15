"""
3D Multiplexing and Video/Image Encoding module.
Supports Half-SBS, Full-SBS, Dubois Red/Cyan Anaglyph, Apple Spatial MV-HEVC, and Spatial HEIC.
"""
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import cv2

from .config import FFMPEG_BIN, SPATIAL_BIN, PIC_COMBINER_BIN
from .system_check import detect_available_encoders, get_optimal_encoder_for_system

class VideoStreamWriter:
    """
    Streams RGB frames directly into an FFmpeg process for high-speed hardware-accelerated encoding.
    Supports NVIDIA NVENC, Apple VideoToolbox, AMD AMF, Intel QSV, and CPU fallback.
    Guarantees strict CFR timestamping using exact rational framerates.
    """
    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: Union[float, str],
        fps_rational: Optional[str] = None,
        source_audio_path: Optional[Path] = None,
        codec: str = "auto",
        start_time: Optional[float] = None,
        duration: Optional[float] = None
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.source_audio = source_audio_path
        self.stderr_output = []

        # Determine exact rational framerate string
        if fps_rational:
            self.fps_str = str(fps_rational)
        elif isinstance(fps, str) and "/" in fps:
            self.fps_str = fps
        else:
            val = float(fps)
            # Match common broadcast and cinema framerates
            matched = False
            for num, den in [
                (24000, 1001), (24, 1), (25, 1),
                (30000, 1001), (30, 1), (50, 1),
                (60000, 1001), (60, 1)
            ]:
                if abs(val - (num / den)) < 0.005:
                    self.fps_str = f"{num}/{den}" if den != 1 else f"{num}"
                    matched = True
                    break
            if not matched:
                self.fps_str = f"{val:.4f}"

        # Determine optimal hardware codec across Windows / Linux / macOS
        if codec == "auto":
            avail = detect_available_encoders(FFMPEG_BIN)
            encoder, _ = get_optimal_encoder_for_system(avail)
        else:
            encoder = codec

        cmd = [
            FFMPEG_BIN,
            "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", self.fps_str,
            "-i", "-"
        ]

        if self.source_audio and self.source_audio.exists():
            audio_args = []
            if start_time is not None and start_time > 0:
                audio_args.extend(["-ss", str(start_time)])
            if duration is not None and duration > 0:
                audio_args.extend(["-t", str(duration)])
            cmd.extend([
                *audio_args,
                "-i", str(self.source_audio),
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a?"
            ])

        # Force exact CFR output timestamps and framerate
        cmd.extend([
            "-r", self.fps_str,
            "-fps_mode", "cfr"
        ])

        # Hardware-specific encoding flags
        if "nvenc" in encoder:
            # NVIDIA NVENC (Windows / Linux)
            cmd.extend([
                "-c:v", encoder,
                "-preset", "p5",
                "-cq", "23",
                "-pix_fmt", "yuv420p"
            ])
        elif "videotoolbox" in encoder:
            # Apple VideoToolbox (macOS)
            cmd.extend([
                "-c:v", encoder,
                "-q:v", "65",
                "-pix_fmt", "yuv420p",
                "-tag:v", "hvc1" if "hevc" in encoder else "avc1"
            ])
        elif "amf" in encoder:
            # AMD AMF (Windows / Linux)
            cmd.extend([
                "-c:v", encoder,
                "-quality", "quality",
                "-pix_fmt", "yuv420p"
            ])
        elif "qsv" in encoder:
            # Intel QuickSync
            cmd.extend([
                "-c:v", encoder,
                "-global_quality", "23",
                "-pix_fmt", "nv12"
            ])
        else:
            # Software fallback (libx265 / libx264)
            cmd.extend([
                "-c:v", encoder,
                "-crf", "20",
                "-preset", "fast",
                "-pix_fmt", "yuv420p"
            ])

        cmd.append(str(output_path))
        self._error_log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=self._error_log
        )

    def write_frame(self, rgb_frame: np.ndarray):
        if self.process.stdin:
            self.process.stdin.write(rgb_frame.tobytes())

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()
        self._error_log.seek(0)
        stderr_data = self._error_log.read()
        self._error_log.close()
        if self.process.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed: {stderr_data.decode('utf-8', errors='replace')[-2000:]}")


def compose_sbs(left: np.ndarray, right: np.ndarray, mode: str = "half_sbs") -> np.ndarray:
    """
    Composes Left and Right eye views into Side-by-Side format.
    'half_sbs': scales width of each eye to W//2, producing (H, W, 3). Standard for 3D TVs (LG/Samsung).
    'full_sbs': keeps full resolution of each eye, producing (H, 2*W, 3). Ideal for VR headsets.
    """
    H, W = left.shape[:2]
    if mode == "half_sbs":
        half_w = W // 2
        l_scaled = cv2.resize(left, (half_w, H), interpolation=cv2.INTER_AREA)
        r_scaled = cv2.resize(right, (half_w, H), interpolation=cv2.INTER_AREA)
        return np.hstack([l_scaled, r_scaled])
    else:
        return np.hstack([left, right])

def compose_anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Dubois Least-Squares Anaglyph (Red/Cyan) matrix transformation.
    Minimizes ghosting/crosstalk and preserves natural scene chrominance.
    """
    # Normalize to [0, 1]
    l_f = left.astype(np.float32) / 255.0
    r_f = right.astype(np.float32) / 255.0

    # Dubois conversion matrix
    r_out = 0.456 * l_f[:, :, 0] + 0.500 * l_f[:, :, 1] + 0.176 * l_f[:, :, 2] - \
            0.043 * r_f[:, :, 0] - 0.088 * r_f[:, :, 1] - 0.002 * r_f[:, :, 2]

    g_out = -0.040 * l_f[:, :, 0] - 0.038 * l_f[:, :, 1] - 0.016 * l_f[:, :, 2] + \
             0.378 * r_f[:, :, 0] + 0.730 * r_f[:, :, 1] - 0.015 * r_f[:, :, 2]

    b_out = -0.015 * l_f[:, :, 0] - 0.021 * l_f[:, :, 1] - 0.005 * l_f[:, :, 2] - \
             0.072 * r_f[:, :, 0] - 0.113 * r_f[:, :, 1] + 1.226 * r_f[:, :, 2]

    anaglyph = np.stack([
        np.clip(r_out * 255.0, 0, 255),
        np.clip(g_out * 255.0, 0, 255),
        np.clip(b_out * 255.0, 0, 255)
    ], axis=-1).astype(np.uint8)

    return anaglyph

def create_spatial_photo(left_path: Path, right_path: Path, output_heic: Path) -> bool:
    """
    Combines Left and Right images into Apple Spatial HEIC format using native run_picCombiner.
    """
    if not Path(PIC_COMBINER_BIN).exists():
        return False

    cmd = [
        PIC_COMBINER_BIN,
        "--left-image-path", str(left_path),
        "--right-image-path", str(right_path),
        "--output-image-path", str(output_heic)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0

def create_spatial_video_mv_hevc(
    left_video: Path,
    right_video: Path,
    output_mov: Path,
    fov: float = 65.0,
    baseline_mm: float = 64.0
) -> bool:
    """
    Muxes left and right video streams into Apple MV-HEVC Spatial Video format via Mike Swanson's spatial CLI.
    """
    if not Path(SPATIAL_BIN).exists():
        return False

    cmd = [
        SPATIAL_BIN,
        "make",
        "-i", str(left_video),
        "-i", str(right_video),
        "-o", str(output_mov),
        "--cdist", str(baseline_mm),
        "--hfov", str(fov)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0
