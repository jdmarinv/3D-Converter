"""Repair one-sided 2D-to-3D conversions with symmetric dual-eye rendering."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .depth_engine import DepthEngine
from .dibr_stereo import StereoSynthesizer
from .encoder_3d import VideoStreamWriter, compose_sbs
from .preprocessor import (
    get_media_info,
    prepare_depth_input,
    neutralize_bar_depth,
    read_video_frames,
)
from .stereo_artifact_detector import detect_artifact_shots


def _eye_pair(frame: np.ndarray, layout: str) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    if layout == "hsbs":
        if width % 2:
            raise ValueError("HSBS width must be even")
        return frame[:, : width // 2], frame[:, width // 2 :]
    if layout == "sbs":
        if width % 2:
            raise ValueError("Full-SBS width must be even")
        return frame[:, : width // 2], frame[:, width // 2 :]
    raise ValueError(f"Unsupported repair layout: {layout}")


def one_sided_signature(
    source_rgb: np.ndarray,
    converted_rgb: np.ndarray,
    layout: str = "hsbs",
    pristine_threshold: float = 3.0,
    shifted_threshold: float = 2.0,
    separation_ratio: float = 1.8,
    separation_margin: float = 1.0,
) -> tuple[bool, str, dict]:
    """Detect whether exactly one packed eye is effectively the pristine source.

    Metrics are measured on a small grayscale proxy. Compression noise is allowed,
    while a genuine shifted eye must differ materially from the source.
    """
    left, right = _eye_pair(converted_rgb, layout)
    size = (160, 90)
    source = cv2.resize(source_rgb, size, interpolation=cv2.INTER_AREA)
    left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY).astype(np.float32)
    left_mae = float(np.mean(np.abs(left - source)))
    right_mae = float(np.mean(np.abs(right - source)))
    left_pristine = (
        left_mae <= pristine_threshold
        and right_mae >= shifted_threshold
        and right_mae >= left_mae * separation_ratio
        and right_mae - left_mae >= separation_margin
    )
    right_pristine = (
        right_mae <= pristine_threshold
        and left_mae >= shifted_threshold
        and left_mae >= right_mae * separation_ratio
        and left_mae - right_mae >= separation_margin
    )
    eye = "left" if left_pristine else ("right" if right_pristine else "none")
    return left_pristine or right_pristine, eye, {
        "left_source_mae": round(left_mae, 4),
        "right_source_mae": round(right_mae, 4),
    }


def parse_ranges(spec: str, fps: float) -> list[tuple[int, int]]:
    if not spec or spec.lower() == "auto":
        return []
    ranges: list[tuple[int, int]] = []
    for item in spec.split(","):
        fields = item.strip().split("-")
        if len(fields) != 2:
            raise ValueError(f"Invalid repair range: {item!r}; use START-END in seconds")
        start, end = (float(value) for value in fields)
        if start < 0 or end <= start:
            raise ValueError(f"Invalid repair range: {item!r}")
        ranges.append((round(start * fps), round(end * fps)))
    return ranges


def _in_ranges(frame_index: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start <= frame_index < end for start, end in ranges)


def repair_stereo_conversion(
    source_2d: Path,
    converted_3d: Path,
    output_path: Path,
    *,
    layout: str = "hsbs",
    ranges: str = "auto",
    divergence: float = 0.025,
    convergence: float = 0.5,
    pop_out: float = 0.0,
    temporal_smooth: float = 0.65,
    style_3d: str = "natural",
    depth_profile: str = "balanced",
    auto_crop: bool = True,
    depth_engine: Optional[DepthEngine] = None,
) -> dict:
    """Repair selected or automatically detected one-sided stereo frames.

    The existing conversion supplies untouched frames and all audio streams. The
    original 2D master supplies RGB and newly inferred depth only for repaired
    frames. Repaired frames are always rendered in symmetric ``both`` mode.
    """
    source_2d = Path(source_2d).resolve()
    converted_3d = Path(converted_3d).resolve()
    output_path = Path(output_path).resolve()
    for path in (source_2d, converted_3d):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path in (source_2d, converted_3d):
        raise ValueError("Repair output must be a new file")

    source_info = get_media_info(source_2d)
    converted_info = get_media_info(converted_3d)
    if abs(source_info["fps"] - converted_info["fps"]) > 0.01:
        raise ValueError("The 2D master and converted file have different frame rates")
    if abs(source_info["duration"] - converted_info["duration"]) > 1.0:
        raise ValueError("The 2D master and converted file differ by more than one second")
    if converted_info["width"] % 2:
        raise ValueError("The converted SBS video width must be even")

    fps = converted_info["fps"]
    automatic = ranges.lower() == "auto"
    report_path = output_path.with_suffix(output_path.suffix + ".artifacts.json")
    if automatic:
        report = detect_artifact_shots(
            source_2d, converted_3d, layout=layout, report_path=report_path
        )
        explicit_ranges = [
            (round(item["start"] * fps), round(item["end"] * fps))
            for item in report["shots"]
        ]
        if not explicit_ranges:
            raise RuntimeError(f"No damaged shots were detected. Report: {report_path}")
        print(f"Detected {len(explicit_ranges)} damaged shots. Report: {report_path}")
        for item in report["shots"]:
            print(f"  Shot {item['shot']}: {item['start']:.3f}s-{item['end']:.3f}s "
                  f"(peak {item['peak_score']:.2f})")
    else:
        explicit_ranges = parse_ranges(ranges, fps)
    engine = depth_engine or DepthEngine(depth_profile=depth_profile)
    engine.temporal_filter.alpha = temporal_smooth
    synthesizer = StereoSynthesizer(
        divergence=divergence,
        convergence=convergence,
        pop_out=pop_out,
        render_mode="both",
        edge_refine=True,
        style_3d=style_3d,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoStreamWriter(
        output_path,
        converted_info["width"],
        converted_info["height"],
        fps,
        fps_rational=converted_info["fps_rational"],
        source_audio_path=converted_3d,
    )
    source_frames = read_video_frames(source_2d, force_cfr=source_info.get("is_vfr", False))
    converted_frames = read_video_frames(converted_3d, force_cfr=converted_info.get("is_vfr", False))
    repaired = 0
    copied = 0
    total = min(source_info["nb_frames"], converted_info["nb_frames"])

    try:
        with tqdm(total=total, desc="Repairing stereo artifacts", unit="frame") as progress:
            for (source_index, source_rgb), (converted_index, converted_rgb) in zip(source_frames, converted_frames):
                if source_index != converted_index:
                    raise RuntimeError("Frame alignment was lost during repair")
                should_repair = _in_ranges(source_index, explicit_ranges)

                if should_repair:
                    depth_input, active_bounds = prepare_depth_input(source_rgb) if auto_crop else (
                        source_rgb, (0, source_rgb.shape[0], 0, source_rgb.shape[1])
                    )
                    depth = engine.estimate_depth(depth_input, apply_temporal_smoothing=True)
                    zero_mask = None
                    if auto_crop:
                        depth = neutralize_bar_depth(depth, active_bounds, synthesizer.convergence)
                        top, bottom, left, right = active_bounds
                        zero_mask = np.ones(source_rgb.shape[:2], dtype=bool)
                        zero_mask[top:bottom, left:right] = False
                    left_eye, right_eye = synthesizer.render_stereo(
                        source_rgb, depth, zero_disparity_mask=zero_mask
                    )
                    mode = "half_sbs" if layout == "hsbs" else "full_sbs"
                    result = compose_sbs(left_eye, right_eye, mode=mode)
                    if result.shape[:2] != converted_rgb.shape[:2]:
                        result = cv2.resize(
                            result,
                            (converted_rgb.shape[1], converted_rgb.shape[0]),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                    writer.write_frame(result)
                    repaired += 1
                else:
                    writer.write_frame(converted_rgb)
                    copied += 1
                progress.update(1)
    finally:
        writer.close()

    if repaired == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            "No one-sided frames were detected. Use --repair-ranges START-END to repair a known interval."
        )
    return {
        "frames_repaired": repaired,
        "frames_copied": copied,
        "artifact_report": str(report_path) if automatic else "manual ranges",
        "render_mode": "both",
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair one-sided 2D-to-3D SBS conversions")
    parser.add_argument("--source-2d", required=True, type=Path, help="Original synchronized 2D master")
    parser.add_argument("--converted-3d", required=True, type=Path, help="Existing HSBS or Full-SBS conversion")
    parser.add_argument("-o", "--output", required=True, type=Path, help="New repaired output file")
    parser.add_argument("--layout", choices=("hsbs", "sbs"), default="hsbs")
    parser.add_argument("--repair-ranges", default="auto", help="auto or comma-separated START-END seconds")
    parser.add_argument("--depth-intensity", type=float, default=0.025)
    parser.add_argument("--convergence", type=float, default=0.5)
    parser.add_argument("--pop-out", type=float, default=0.0)
    parser.add_argument("--temporal-smooth", type=float, default=0.65)
    parser.add_argument("--style-3d", choices=("natural", "cinematic"), default="natural")
    parser.add_argument("--depth-profile", choices=("fast", "balanced", "high_fidelity"), default="balanced")
    parser.add_argument("--no-crop", action="store_true")
    args = parser.parse_args()
    report = repair_stereo_conversion(
        args.source_2d,
        args.converted_3d,
        args.output,
        layout=args.layout,
        ranges=args.repair_ranges,
        divergence=args.depth_intensity,
        convergence=args.convergence,
        pop_out=args.pop_out,
        temporal_smooth=args.temporal_smooth,
        style_3d=args.style_3d,
        depth_profile=args.depth_profile,
        auto_crop=not args.no_crop,
    )
    print("Stereo artifact repair completed:")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
