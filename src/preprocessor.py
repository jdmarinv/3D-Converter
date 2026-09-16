"""
Video and Image Preprocessing module.
Handles black bar detection (cropdetect), metadata extraction, and streaming frame I/O.
"""
import subprocess
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Generator, Tuple, Optional
import numpy as np

from .config import FFMPEG_BIN, FFPROBE_BIN


def detect_active_image_bounds(
    rgb: np.ndarray,
    black_threshold: int = 16,
    black_ratio: float = 0.985,
    min_bar_size: int = 4,
    max_bar_fraction: float = 0.32,
) -> Tuple[int, int, int, int]:
    """Find black border bars without changing the frame dimensions.

    Returns ``(top, bottom, left, right)`` using exclusive bottom/right
    coordinates. Detection is deliberately limited to the outer 32% so a
    fade-to-black or a naturally dark shot is not mistaken for letterboxing.
    """
    if rgb.ndim != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise ValueError("Expected an RGB image with shape HxWx3")

    # Integer luma approximation avoids another OpenCV conversion per frame.
    gray = (
        rgb[..., 0].astype(np.uint16) * 54
        + rgb[..., 1].astype(np.uint16) * 183
        + rgb[..., 2].astype(np.uint16) * 19
    ) >> 8
    row_black = np.mean(gray <= black_threshold, axis=1) >= black_ratio
    col_black = np.mean(gray <= black_threshold, axis=0) >= black_ratio

    def edge_count(lines: np.ndarray, reverse: bool = False) -> int:
        limit = max(min_bar_size, int(len(lines) * max_bar_fraction))
        count = 0
        iterable = lines[::-1] if reverse else lines
        for is_black in iterable[:limit]:
            if not is_black:
                break
            count += 1
        # If darkness continues past the maximum plausible bar size, this is
        # probably a fade or dark frame rather than a matte.
        if count == limit and limit < len(lines) and iterable[limit]:
            return 0
        return count if count >= min_bar_size else 0

    top = edge_count(row_black)
    bottom_bar = edge_count(row_black, reverse=True)
    left = edge_count(col_black)
    right_bar = edge_count(col_black, reverse=True)
    bottom = rgb.shape[0] - bottom_bar
    right = rgb.shape[1] - right_bar
    if bottom <= top or right <= left:
        return 0, rgb.shape[0], 0, rgb.shape[1]
    return top, bottom, left, right


def prepare_depth_input(rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Neutralize black bars for depth inference while preserving the canvas.

    Border pixels are extended from the active image. This prevents solid
    mattes from skewing model normalization without zooming or cropping the
    source frame.
    """
    top, bottom, left, right = detect_active_image_bounds(rgb)
    prepared = rgb.copy()
    if top:
        prepared[:top, left:right] = prepared[top:top + 1, left:right]
    if bottom < rgb.shape[0]:
        prepared[bottom:, left:right] = prepared[bottom - 1:bottom, left:right]
    if left:
        prepared[:, :left] = prepared[:, left:left + 1]
    if right < rgb.shape[1]:
        prepared[:, right:] = prepared[:, right - 1:right]
    return prepared, (top, bottom, left, right)


def neutralize_bar_depth(
    depth: np.ndarray,
    bounds: Tuple[int, int, int, int],
    convergence: float,
) -> np.ndarray:
    """Put matte bars on the zero-disparity plane in a full-size depth map."""
    top, bottom, left, right = bounds
    if top == 0 and bottom == depth.shape[0] and left == 0 and right == depth.shape[1]:
        return depth
    result = depth.copy()
    result[:top, :] = convergence
    result[bottom:, :] = convergence
    result[:, :left] = convergence
    result[:, right:] = convergence
    return result

def parse_rational_fraction(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parses a string fraction (e.g. '24000/1001', '24/1') or float into (numerator, denominator).
    """
    if not s or s in ("0/0", "0/1", "0"):
        return None
    if "/" in s:
        parts = s.split("/")
        try:
            num = int(parts[0])
            den = int(parts[1])
            if den != 0:
                return num, den
        except ValueError:
            return None
    else:
        try:
            val = float(s)
            if val <= 0:
                return None
            for num, den in [
                (24000, 1001), (24, 1), (25, 1),
                (30000, 1001), (30, 1), (50, 1),
                (60000, 1001), (60, 1)
            ]:
                if abs(val - (num / den)) < 0.005:
                    return num, den
            return int(round(val * 1000)), 1000
        except ValueError:
            return None
    return None

def get_media_info(file_path: Path) -> Dict[str, Any]:
    """
    Extracts video metadata (dimensions, fps, duration, stream count, rational framerate, VFR flag) via ffprobe.
    """
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames:format=duration",
        "-of", "json",
        str(file_path)
    ]
    result = None
    for attempt in range(4):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            break
        time.sleep(0.3)
    else:
        err = result.stderr.strip() if result else "unknown error"
        raise RuntimeError(f"ffprobe failed on {file_path}: {err or 'Exit code ' + str(result.returncode)}")
    data = json.loads(result.stdout)
    
    stream = data.get("streams", [{}])[0]
    format_data = data.get("format", {})
    
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    
    r_fps_str = stream.get("r_frame_rate", "30/1")
    avg_fps_str = stream.get("avg_frame_rate", "30/1")
    
    r_frac = parse_rational_fraction(r_fps_str)
    avg_frac = parse_rational_fraction(avg_fps_str)
    
    # Standard rates check
    standard_rates = [
        (24000, 1001), (24, 1), (25, 1),
        (30000, 1001), (30, 1), (50, 1),
        (60000, 1001), (60, 1)
    ]

    # Resolve optimal rational framerate
    chosen_frac = None
    is_vfr = False

    # Detect container timebase anomalies (e.g. MKV 1000/1 or 90000/1)
    if r_frac and (r_frac[0] / r_frac[1]) > 120 and avg_frac and (avg_frac[0] / avg_frac[1]) <= 120:
        chosen_frac = avg_frac
    elif r_frac and (r_frac[0] / r_frac[1]) <= 120:
        # Check if r_frac matches standard rate
        r_val = r_frac[0] / r_frac[1]
        matched_std = None
        for std_num, std_den in standard_rates:
            if abs(r_val - (std_num / std_den)) < 0.005:
                matched_std = (std_num, std_den)
                break
        chosen_frac = matched_std if matched_std else r_frac
    elif avg_frac and (avg_frac[0] / avg_frac[1]) <= 120:
        chosen_frac = avg_frac
    else:
        chosen_frac = (30, 1)

    # Detect VFR if r_frac and avg_frac differ significantly
    if r_frac and avg_frac:
        r_v = r_frac[0] / r_frac[1]
        a_v = avg_frac[0] / avg_frac[1]
        if r_v <= 120 and a_v <= 120 and abs(r_v - a_v) > 0.05:
            is_vfr = True

    fps_num, fps_den = chosen_frac
    fps = fps_num / fps_den
    fps_rational = f"{fps_num}/{fps_den}" if fps_den != 1 else f"{fps_num}"
        
    duration = float(stream.get("duration") or format_data.get("duration") or 0.0)
    nb_frames = int(round(duration * fps)) if (is_vfr and duration > 0) else int(stream.get("nb_frames") or round(duration * fps))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "fps_rational": fps_rational,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "duration": duration,
        "nb_frames": nb_frames,
        "is_vfr": is_vfr
    }

def has_audio_stream(file_path: Path) -> bool:
    """
    Checks if media file contains at least one audio stream.
    """
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "json",
        str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        return len(data.get("streams", [])) > 0
    return False

def detect_black_bars(file_path: Path, samples: int = 15) -> Optional[str]:
    """
    Runs FFmpeg cropdetect on spaced samples to detect letterboxing/pillarboxing.
    Returns FFmpeg crop filter string (e.g. 'crop=1920:800:0:140') or None if full frame.
    """
    cmd = [
        FFMPEG_BIN,
        "-i", str(file_path),
        "-vf", "cropdetect=24:16:0",
        "-vframes", str(samples),
        "-f", "null",
        "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", result.stderr)
    if not crops:
        return None
    
    # Find most common crop parameter
    from collections import Counter
    most_common, count = Counter(crops).most_common(1)[0]
    
    # Compare with native resolution
    info = get_media_info(file_path)
    native = f"{info['width']}:{info['height']}:0:0"
    if most_common == native:
        return None
        
    return f"crop={most_common}"

def read_video_frames(
    file_path: Path,
    crop_filter: Optional[str] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    start_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
    force_cfr: bool = False
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Streams raw RGB24 frames from video via FFmpeg standard output pipe.
    Guarantees frame-accurate index tracking and timestamp continuity.
    Yields: (frame_index, numpy_array[H, W, 3] in RGB)
    """
    info = get_media_info(file_path)
    w, h = info["width"], info["height"]
    fps_rational = info["fps_rational"]

    vf_filters = []
    if crop_filter:
        vf_filters.append(crop_filter)
        parts = crop_filter.replace("crop=", "").split(":")
        w, h = int(parts[0]), int(parts[1])

    # If force_cfr is requested or input is VFR, add fps filter to ensure strict CFR timing
    use_cfr = force_cfr or bool(info.get("is_vfr", False))
    if use_cfr:
        vf_filters.append(f"fps=fps={fps_rational}:round=near")

    # If start_frame is specified, calculate seek and trim if beneficial
    discard_initial_frames = 0
    cmd = [FFMPEG_BIN]

    if start_frame is not None and start_frame > 0:
        fps_val = info["fps"]
        target_sec = start_frame / fps_val
        if target_sec > 10.0:
            # Fast seek to 3 seconds before target frame to decode GOP safely
            safe_seek_sec = max(0.0, target_sec - 3.0)
            cmd.extend(["-ss", f"{safe_seek_sec:.3f}"])
            # We will stream and discard frames until exact start_frame is reached
            # Or use accurate seek after -i
        cmd.extend(["-i", str(file_path)])
        if target_sec > 10.0:
            # Accurate seek offset from keyframe
            cmd.extend(["-ss", f"{target_sec - safe_seek_sec:.3f}"])
        else:
            cmd.extend(["-ss", f"{target_sec:.3f}"])
    else:
        if start_time is not None and start_time > 0:
            cmd.extend(["-ss", str(start_time)])
        cmd.extend(["-i", str(file_path)])

    if duration is not None and duration > 0:
        cmd.extend(["-t", str(duration)])

    vf_arg = ["-vf", ",".join(vf_filters)] if vf_filters else []
    cmd.extend([
        *vf_arg,
        "-fps_mode", "cfr" if use_cfr else "passthrough",
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-"
    ])

    frame_bytes = w * h * 3
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_bytes * 4
    )

    current_idx = start_frame if start_frame is not None else 0
    frames_yielded = 0

    try:
        while True:
            if max_frames is not None and frames_yielded >= max_frames:
                break
            raw_frame = process.stdout.read(frame_bytes)
            if not raw_frame or len(raw_frame) < frame_bytes:
                break
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))
            yield current_idx, frame
            current_idx += 1
            frames_yielded += 1
    finally:
        if process.stdout:
            process.stdout.close()
        process.wait()
