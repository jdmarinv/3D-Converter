"""
Visual Cadence and Temporal Continuity Diagnostic Tool.
Audits 2D input vs 3D output frame-by-frame to detect pipeline-introduced stutter,
dropped frames, and periodic duplicate frames without falsely flagging natural static scenes.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

from .preprocessor import get_media_info, read_video_frames

def extract_reference_view(
    out_frame: np.ndarray,
    fmt: str = "auto",
    orig_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Extracts the pristine reference eye (Left eye) from stereoscopic composite frames.
    Handles Full-SBS, Half-SBS, and generic images.
    """
    H, W = out_frame.shape[:2]
    if orig_shape:
        orig_h, orig_w = orig_shape
    else:
        orig_h, orig_w = H, W // 2

    if fmt == "2d":
        return out_frame

    if fmt == "sbs" or (fmt == "auto" and W == orig_w * 2):
        # Full SBS: Left eye is the left half at full resolution
        return out_frame[:, :W // 2]
    elif fmt == "hsbs":
        # Half SBS: Left eye is the left half horizontally compressed
        half_w = W // 2
        left_half = out_frame[:, :half_w]
        return cv2.resize(left_half, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        if W >= orig_w * 2:
            return out_frame[:, :W // 2]
        # For auto with same width: if input and output dimensions match and not specified as 3D, treat as full frame
        if W == orig_w:
            return out_frame
        return out_frame[:, :W // 2]

def detect_periodicity(duplicate_mask: np.ndarray, min_duplicates: int = 5) -> Tuple[Optional[int], float]:
    """
    Analyzes duplicate frame indices using lag autocorrelation and interval histograms
    to detect periodic patterns (e.g. 1 duplicate every 3 frames, 1 every 4 frames).
    Returns (detected_period, confidence_0_to_1).
    """
    dup_indices = np.where(duplicate_mask > 0)[0]
    if len(dup_indices) < min_duplicates:
        return None, 0.0

    # Calculate intervals between consecutive duplicates
    intervals = np.diff(dup_indices)
    if len(intervals) == 0:
        return None, 0.0

    from collections import Counter
    counts = Counter(intervals)
    most_common_period, count = counts.most_common(1)[0]
    confidence = count / len(intervals)

    if confidence >= 0.40 and most_common_period in (2, 3, 4, 5, 6, 8, 12):
        return int(most_common_period), float(confidence)

    # Autocorrelation check
    n = len(duplicate_mask)
    if n >= 20:
        signal = duplicate_mask.astype(np.float32) - np.mean(duplicate_mask)
        norm = np.sum(signal ** 2)
        if norm > 1e-6:
            for lag in [3, 4, 2, 5, 6]:
                if lag < n:
                    r = np.sum(signal[:-lag] * signal[lag:]) / norm
                    if r > 0.35:
                        return lag, float(r)

    return None, 0.0

def analyze_visual_cadence(
    input_path: Path,
    output_path: Path,
    fmt: str = "auto",
    max_frames: Optional[int] = None,
    diff_threshold: float = 1.5,
    sample_stride: int = 1,
    start_time: float = 0.0,
    crop_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compares consecutive frame differences between input and output video streams.
    Identifies:
      - Native static scenes: D_in < threshold and D_out < threshold (VALID).
      - Normal continuous motion: D_in >= threshold and D_out >= threshold (VALID).
      - Pipeline-introduced duplicate: D_in >= threshold and D_out < threshold (STUTTER DEFECT).
      - Periodicity of introduced duplicates.
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    info_in = get_media_info(input_path)
    info_out = get_media_info(output_path)

    w_in, h_in = info_in["width"], info_in["height"]
    total_frames = min(info_in["nb_frames"], info_out["nb_frames"])
    if max_frames:
        total_frames = min(total_frames, max_frames)

    gen_in = read_video_frames(input_path, max_frames=total_frames, start_time=start_time, crop_filter=crop_filter)
    gen_out = read_video_frames(output_path, max_frames=total_frames)

    prev_in = None
    prev_out = None

    native_static_count = 0
    normal_motion_count = 0
    introduced_duplicate_indices = []
    diff_in_history = []
    diff_out_history = []

    frame_count = 0

    while True:
        try:
            idx_in, f_in = next(gen_in)
            idx_out, f_out_raw = next(gen_out)
        except StopIteration:
            break

        f_out = extract_reference_view(f_out_raw, fmt=fmt, orig_shape=(h_in, w_in))

        # Downscale slightly for robust luminance metric
        f_in_small = cv2.resize(f_in, (320, int(320 * h_in / w_in)), interpolation=cv2.INTER_AREA).astype(np.float32)
        f_out_small = cv2.resize(f_out, (320, int(320 * h_in / w_in)), interpolation=cv2.INTER_AREA).astype(np.float32)

        if prev_in is not None and prev_out is not None:
            # Calculate mean absolute pixel difference
            d_in = float(np.mean(np.abs(f_in_small - prev_in)))
            d_out = float(np.mean(np.abs(f_out_small - prev_out)))

            diff_in_history.append(d_in)
            diff_out_history.append(d_out)

            is_in_moving = d_in >= diff_threshold
            is_out_moving = d_out >= diff_threshold

            if not is_in_moving:
                # Legitimate natural static scene or freeze frame in source
                native_static_count += 1
            elif is_in_moving and not is_out_moving:
                # Pipeline-induced duplicate frame: original moves, but output is frozen!
                introduced_duplicate_indices.append(frame_count)
            else:
                normal_motion_count += 1

        prev_in = f_in_small
        prev_out = f_out_small
        frame_count += 1

    analyzed_transitions = frame_count - 1
    dup_count = len(introduced_duplicate_indices)
    dup_pct = (dup_count / max(1, analyzed_transitions)) * 100.0

    duplicate_mask = np.zeros(max(1, frame_count), dtype=np.uint8)
    if introduced_duplicate_indices:
        duplicate_mask[introduced_duplicate_indices] = 1

    detected_period, period_conf = detect_periodicity(duplicate_mask)

    # Verdict determination: Stutter if introduced duplicates exceed 1.5%
    stutter_detected = dup_pct >= 1.5
    verdict = "FAIL_STUTTER_DETECTED" if stutter_detected else "PASS"

    pattern_desc = "None"
    if detected_period:
        pattern_desc = f"1 duplicate roughly every {detected_period} frames (confidence: {period_conf*100:.0f}%)"

    return {
        "verdict": verdict,
        "stutter_detected": stutter_detected,
        "total_frames_analyzed": frame_count,
        "analyzed_transitions": analyzed_transitions,
        "native_static_frames": native_static_count,
        "normal_motion_frames": normal_motion_count,
        "introduced_duplicates": dup_count,
        "introduced_duplicate_pct": round(dup_pct, 2),
        "detected_period": detected_period,
        "period_confidence": round(period_conf, 2),
        "pattern_description": pattern_desc,
        "input_fps_rational": info_in["fps_rational"],
        "output_fps_rational": info_out["fps_rational"],
        "input_is_vfr": info_in["is_vfr"],
        "output_is_vfr": info_out["is_vfr"]
    }

def print_report(report: Dict[str, Any]):
    print("=" * 72)
    print("      VISUAL CADENCE & TEMPORAL CONTINUITY DIAGNOSTIC REPORT")
    print("=" * 72)
    print(f"Status / Verdict:        {report['verdict']}")
    print(f"Frames Analyzed:         {report['total_frames_analyzed']}")
    print(f"Input Framerate:         {report['input_fps_rational']} (VFR: {report['input_is_vfr']})")
    print(f"Output Framerate:        {report['output_fps_rational']} (VFR: {report['output_is_vfr']})")
    print("-" * 72)
    print(f"Normal Moving Frames:    {report['normal_motion_frames']}")
    print(f"Legitimate Static:       {report['native_static_frames']} (Preserved without false alarm)")
    print(f"Introduced Duplicates:   {report['introduced_duplicates']} ({report['introduced_duplicate_pct']}%)")
    print(f"Detected Stutter Pattern:{report['pattern_description']}")
    print("=" * 72)
    if report["stutter_detected"]:
        print("⚠️  WARNING: Pipeline-induced temporal stutter detected!")
        print("   The converted video contains frozen/repeated frames while the original has active motion.")
    else:
        print("✓ No introduced stutter detected in the analyzed frames.")
    print("=" * 72)

def main():
    parser = argparse.ArgumentParser(description="Visual Cadence and Temporal Continuity Diagnostic")
    parser.add_argument("-i", "--input", required=True, help="Path to original 2D reference video")
    parser.add_argument("-o", "--output", required=True, help="Path to converted 3D video")
    parser.add_argument("-f", "--format", choices=["auto", "hsbs", "sbs"], default="auto", help="3D output format")
    parser.add_argument("-n", "--max-frames", type=int, default=None, help="Max frames to analyze (default: all)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    report = analyze_visual_cadence(
        Path(args.input),
        Path(args.output),
        fmt=args.format,
        max_frames=args.max_frames
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    sys.exit(1 if report["stutter_detected"] else 0)

if __name__ == "__main__":
    main()
