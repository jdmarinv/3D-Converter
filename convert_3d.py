#!/usr/bin/env python3
"""
2D to 3D Offline Converter CLI.
Converts 2D images and videos to SBS (Half/Full), Anaglyph, and Apple Spatial formats.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from tqdm import tqdm

from src.config import (
    DEVICE,
    DEFAULT_DIVERGENCE,
    DEFAULT_CONVERGENCE,
    DEFAULT_POP_OUT,
    DEFAULT_TEMPORAL_SMOOTH,
    DEFAULT_OUTPUT_DIR
)
from src.depth_engine import DepthEngine, DepthDecimator
from src.dibr_stereo import StereoSynthesizer
from src.preprocessor import get_media_info, detect_black_bars, read_video_frames, has_audio_stream
from src.checkpoint_manager import CheckpointManager, concatenate_segments
from src.cadence_analyzer import analyze_visual_cadence, print_report
from src.cadence_repair import repair_defective_cadence
from src.encoder_3d import (
    compose_sbs,
    compose_anaglyph,
    VideoStreamWriter,
    create_spatial_photo,
    create_spatial_video_mv_hevc
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

def process_image(
    input_path: Path,
    output_path: Path,
    depth_engine: Optional[DepthEngine],
    synthesizer: StereoSynthesizer,
    fmt: str,
    custom_depth_path: Optional[Path] = None,
    save_depth: bool = False
):
    print(f"[Image Mode] Loading: {input_path.name}")
    bgr = cv2.imread(str(input_path))
    if bgr is None:
        raise ValueError(f"Could not open image: {input_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if custom_depth_path and custom_depth_path.exists():
        print(f"[Image Mode] Loading custom depth map from: {custom_depth_path.name}")
        d_img = cv2.imread(str(custom_depth_path), cv2.IMREAD_GRAYSCALE)
        d_img = cv2.resize(d_img, (rgb.shape[1], rgb.shape[0]))
        depth = d_img.astype(np.float32) / 255.0
    else:
        print("[Image Mode] Estimating monocular depth map via AI...")
        t0 = time.time()
        depth = depth_engine.estimate_depth(rgb, apply_temporal_smoothing=False)
        print(f"[Image Mode] Depth generated in {time.time() - t0:.2f}s.")

    if save_depth:
        depth_out_path = output_path.with_name(f"{output_path.stem}_depth.png")
        cv2.imwrite(str(depth_out_path), (depth * 255.0).astype(np.uint8))
        print(f"[Image Mode] ✓ Depth map saved to: {depth_out_path.name}")

    print(f"[Image Mode] Synthesizing stereo (Render Mode: {synthesizer.render_mode})...")
    left, right = synthesizer.render_stereo(rgb, depth)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "anaglyph":
        out_img = compose_anaglyph(left, right)
        cv2.imwrite(str(output_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    elif fmt == "sbs":
        out_img = compose_sbs(left, right, mode="full_sbs")
        cv2.imwrite(str(output_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    elif fmt == "spatial":
        tmp_left = output_path.with_name(f"{output_path.stem}_left.png")
        tmp_right = output_path.with_name(f"{output_path.stem}_right.png")
        cv2.imwrite(str(tmp_left), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tmp_right), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
        
        target_heic = output_path.with_suffix(".heic")
        print(f"[Image Mode] Packaging Apple Spatial Photo: {target_heic.name}")
        success = create_spatial_photo(tmp_left, tmp_right, target_heic)
        tmp_left.unlink(missing_ok=True)
        tmp_right.unlink(missing_ok=True)
        if not success:
            print("[Warning] picCombiner failed. Falling back to Half-SBS.")
            out_img = compose_sbs(left, right, mode="half_sbs")
            cv2.imwrite(str(output_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        else:
            output_path = target_heic
    else: # hsbs default
        out_img = compose_sbs(left, right, mode="half_sbs")
        cv2.imwrite(str(output_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    print(f"✓ Successfully saved 3D image to: {output_path}")

def process_video(
    input_path: Path,
    output_path: Path,
    depth_engine: Optional[DepthEngine],
    synthesizer: StereoSynthesizer,
    fmt: str,
    auto_crop: bool = True,
    start_time: float = 0.0,
    duration: float = 0.0,
    custom_depth_path: Optional[Path] = None,
    save_depth: bool = False,
    batch_size: int = 1,
    depth_stride: int = 1,
    resume: bool = False,
    check_cadence: bool = False
):
    print(f"[Video Mode] Inspecting media: {input_path.name}")
    info = get_media_info(input_path)
    total_frames = info["nb_frames"]
    fps = info["fps"]
    fps_rational = info["fps_rational"]
    has_audio = has_audio_stream(input_path)

    if duration > 0:
        total_frames = int(min(total_frames, round(duration * fps)))

    vfr_status = "Yes (Auto-CFR sync active)" if info.get("is_vfr") else "No"
    print(f"  • Resolution: {info['width']}x{info['height']}")
    print(f"  • Frame Rate: {fps_rational} ({fps:.2f} fps, VFR: {vfr_status})")
    print(f"  • Total Frames to process: ~{total_frames}")
    if start_time > 0 or duration > 0:
        print(f"  • Time range: Start={start_time}s, Duration={'Full' if duration <= 0 else f'{duration}s'}")
    print(f"  • Audio: {'Present' if has_audio else 'None'}")
    print(f"  • Render Mode: {synthesizer.render_mode.upper()} (Left eye untouched)")
    if depth_stride > 1:
        print(f"  • Depth Stride: Every {depth_stride} frames (RGB frames strictly 1:1 preserved)")
    if batch_size > 1:
        print(f"  • Batch Size: {batch_size} frames (Strict FIFO order)")

    crop_filter = None
    if auto_crop:
        print("[Video Mode] Checking for letterboxing (black bars)...")
        crop_filter = detect_black_bars(input_path)
        if crop_filter:
            print(f"  • Detected letterboxing: auto-applying '{crop_filter}'")

    # Determine dimensions for output
    w, h = info["width"], info["height"]
    if crop_filter:
        parts = crop_filter.replace("crop=", "").split(":")
        w, h = int(parts[0]), int(parts[1])

    out_w = w * 2 if fmt == "sbs" else w
    out_h = h

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint setup
    checkpoint_mgr = CheckpointManager(output_path, input_path)
    start_frame = 0
    segment_paths = []

    if resume:
        cp_data = checkpoint_mgr.load()
        if cp_data:
            start_frame = cp_data.get("processed_frames", 0)
            segment_paths = [Path(s) for s in cp_data.get("segments", [])]
            print(f"[Checkpoint] Resuming from frame {start_frame}/{total_frames} ({len(segment_paths)} prior segments found)")

    # Current segment target
    target_video_path = output_path
    if resume or checkpoint_mgr.exists() or start_frame > 0:
        target_video_path = checkpoint_mgr.get_segment_path(len(segment_paths) + 1)
        segment_paths.append(target_video_path)

    # If save_depth is requested, open a depth video stream
    depth_writer = None
    if save_depth:
        depth_vid_path = output_path.with_name(f"{output_path.stem}_depth.mp4")
        print(f"[Video Mode] Will save separate depth map video to: {depth_vid_path.name}")
        depth_writer = VideoStreamWriter(
            depth_vid_path, w, h, fps,
            fps_rational=fps_rational,
            source_audio_path=None
        )

    # Custom depth stream reader if provided
    custom_depth_gen = None
    if custom_depth_path and custom_depth_path.exists():
        print(f"[Video Mode] Re-export mode: using custom depth map from {custom_depth_path.name}")
        custom_depth_gen = read_video_frames(
            custom_depth_path, crop_filter=crop_filter,
            start_time=start_time if start_time > 0 else None,
            duration=duration if duration > 0 else None,
            start_frame=start_frame if start_frame > 0 else None,
            force_cfr=info.get("is_vfr", False)
        )

    # Main video writer
    if fmt == "spatial":
        tmp_left_vid = output_path.with_name(f"{output_path.stem}_left_temp.mp4")
        tmp_right_vid = output_path.with_name(f"{output_path.stem}_right_temp.mp4")
        left_writer = VideoStreamWriter(
            tmp_left_vid, w, h, fps,
            fps_rational=fps_rational,
            source_audio_path=input_path if has_audio else None,
            start_time=start_time, duration=duration
        )
        right_writer = VideoStreamWriter(
            tmp_right_vid, w, h, fps,
            fps_rational=fps_rational,
            source_audio_path=None
        )
    else:
        writer = VideoStreamWriter(
            target_video_path,
            out_w,
            out_h,
            fps,
            fps_rational=fps_rational,
            source_audio_path=input_path if has_audio else None,
            start_time=start_time,
            duration=duration
        )

    if depth_engine:
        depth_engine.temporal_filter.reset()

    decimator = DepthDecimator(stride=depth_stride) if depth_stride > 1 else None

    frame_generator = read_video_frames(
        input_path, crop_filter=crop_filter,
        start_time=start_time if start_time > 0 else None,
        duration=duration if duration > 0 else None,
        start_frame=start_frame if start_frame > 0 else None,
        force_cfr=info.get("is_vfr", False)
    )

    frames_written = start_frame
    frames_remaining = total_frames - start_frame

    def write_single_frame(idx: int, rgb_frame: np.ndarray, depth_map: np.ndarray):
        nonlocal frames_written
        # Optional depth map saving
        if depth_writer:
            depth_vis = (depth_map * 255.0).astype(np.uint8)
            depth_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
            depth_writer.write_frame(depth_rgb)

        # Stereo Synthesis: STRICTLY use current rgb_frame
        left, right = synthesizer.render_stereo(rgb_frame, depth_map)

        # Format Composition
        if fmt == "spatial":
            left_writer.write_frame(left)
            right_writer.write_frame(right)
        elif fmt == "anaglyph":
            frame_out = compose_anaglyph(left, right)
            writer.write_frame(frame_out)
        elif fmt == "sbs":
            frame_out = compose_sbs(left, right, mode="full_sbs")
            writer.write_frame(frame_out)
        else: # hsbs default
            frame_out = compose_sbs(left, right, mode="half_sbs")
            writer.write_frame(frame_out)

        frames_written += 1

    with tqdm(total=total_frames, initial=start_frame, desc="Converting 2D->3D", unit="frame") as pbar:
        # Buffer for batch processing
        batch_buffer = []

        for idx, rgb_frame in frame_generator:
            if custom_depth_gen:
                _, d_rgb = next(custom_depth_gen)
                depth = cv2.cvtColor(d_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                write_single_frame(idx, rgb_frame, depth)
                pbar.update(1)
            elif decimator:
                # Decimated depth mode: depth computed every N frames, RGB strictly untouched!
                if decimator.should_compute(idx):
                    depth = depth_engine.estimate_depth(rgb_frame, apply_temporal_smoothing=True)
                    decimator.register_computed_depth(depth)
                else:
                    depth = decimator.get_depth_for_frame(idx)
                write_single_frame(idx, rgb_frame, depth)
                pbar.update(1)
            elif batch_size > 1:
                # Batch processing mode
                batch_buffer.append((idx, rgb_frame))
                if len(batch_buffer) >= batch_size:
                    imgs = [f for _, f in batch_buffer]
                    depth_maps = depth_engine.estimate_depth_batch(imgs, apply_temporal_smoothing=True)
                    for (b_idx, b_rgb), b_depth in zip(batch_buffer, depth_maps):
                        write_single_frame(b_idx, b_rgb, b_depth)
                        pbar.update(1)
                    batch_buffer.clear()
            else:
                # Standard single-frame mode
                depth = depth_engine.estimate_depth(rgb_frame, apply_temporal_smoothing=True)
                write_single_frame(idx, rgb_frame, depth)
                pbar.update(1)

            # Checkpoint saving periodically
            if (frames_written % 250 == 0) and (resume or checkpoint_mgr.exists()):
                checkpoint_mgr.save(
                    frames_written, total_frames, fps_rational,
                    [str(s) for s in segment_paths]
                )

        # Flush any remaining frames in batch buffer
        if batch_buffer:
            imgs = [f for _, f in batch_buffer]
            depth_maps = depth_engine.estimate_depth_batch(imgs, apply_temporal_smoothing=True)
            for (b_idx, b_rgb), b_depth in zip(batch_buffer, depth_maps):
                write_single_frame(b_idx, b_rgb, b_depth)
                pbar.update(1)
            batch_buffer.clear()

    if depth_writer:
        depth_writer.close()

    if fmt == "spatial":
        left_writer.close()
        right_writer.close()
        target_mov = output_path.with_suffix(".mov")
        print(f"[Video Mode] Multiplexing Apple MV-HEVC Spatial Video to: {target_mov.name}")
        success = create_spatial_video_mv_hevc(tmp_left_vid, tmp_right_vid, target_mov)
        tmp_left_vid.unlink(missing_ok=True)
        tmp_right_vid.unlink(missing_ok=True)
        if not success:
            print("[Warning] spatial CLI failed. Please check logs.")
        else:
            print(f"✓ Apple Spatial Video MV-HEVC saved to: {target_mov}")
    else:
        writer.close()

        # If multi-segment, join losslessly
        if len(segment_paths) > 1:
            print(f"[Checkpoint] Joining {len(segment_paths)} encoded video segments...")
            success = concatenate_segments(segment_paths, output_path, cleanup_segments=True)
            if success:
                print(f"✓ Seamless segment concatenation complete: {output_path}")
                checkpoint_mgr.remove()
            else:
                print(f"[Warning] Failed to concatenate segments automatically.")
        elif len(segment_paths) == 1 and segment_paths[0] != output_path:
            segment_paths[0].replace(output_path)
            checkpoint_mgr.remove()

        print(f"✓ 3D Video successfully rendered to: {output_path}")

    # Cadence diagnosis if requested
    if check_cadence and fmt in ("sbs", "hsbs"):
        print("\n[Diagnostics] Running Visual Cadence & Temporal Continuity Audit...")
        try:
            report = analyze_visual_cadence(input_path, output_path, fmt=fmt,
                                            start_time=start_time, crop_filter=crop_filter)
            import json
            report_path = output_path.with_suffix(output_path.suffix + ".cadence.json")
            report_path.write_text(json.dumps(report, indent=2))
            print("PORTAL_CADENCE " + json.dumps(report), flush=True)
            print_report(report)
        except Exception as e:
            print(f"[Warning] Visual cadence diagnostic could not complete: {e}", flush=True)

PROFILES = {
    "balanced": {"divergence": 0.025, "convergence": 0.50, "pop_out": 0.0, "temporal_smooth": 0.65},
    "cinematic_depth": {"divergence": 0.030, "convergence": 0.45, "pop_out": 0.1, "temporal_smooth": 0.75},
    "subtle_parallax": {"divergence": 0.020, "convergence": 0.50, "pop_out": 0.0, "temporal_smooth": 0.50},
    "high_contrast": {"divergence": 0.022, "convergence": 0.50, "pop_out": 0.0, "temporal_smooth": 0.70}
}
# Backward-compatibility aliases
PROFILES["standard"] = PROFILES["balanced"]
PROFILES["cinematic"] = PROFILES["cinematic_depth"]
PROFILES["sharp_edges"] = PROFILES["subtle_parallax"]
PROFILES["movies_anime"] = PROFILES["high_contrast"]

def main():
    parser = argparse.ArgumentParser(
        description="Offline 2D to 3D Image & Video Converter (SBS, Anaglyph, Apple Spatial)"
    )
    parser.add_argument("-i", "--input", type=str, default=None, help="Path to input 2D image or video")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to output file")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Interface and diagnostics language code (default: 'en', secondary: 'es')"
    )
    parser.add_argument(
        "--check-system", "--diagnostics",
        action="store_true",
        help="Run full cross-platform hardware, GPU, drivers and encoder diagnostics"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["hsbs", "sbs", "anaglyph", "spatial"],
        default="hsbs",
        help="3D format: hsbs (Half Side-by-Side), sbs (Full SBS), anaglyph (Dubois Red/Cyan), spatial (MV-HEVC / Spatial Photo)"
    )
    parser.add_argument(
        "--depth-intensity",
        type=float,
        default=DEFAULT_DIVERGENCE,
        help=f"Disparity / stereo separation strength (default: {DEFAULT_DIVERGENCE})"
    )
    parser.add_argument(
        "--convergence",
        type=float,
        default=DEFAULT_CONVERGENCE,
        help=f"Zero-parallax convergence plane 0.0-1.0 (default: {DEFAULT_CONVERGENCE})"
    )
    parser.add_argument(
        "--pop-out",
        type=float,
        default=DEFAULT_POP_OUT,
        help=f"Foreground pop-out bias (default: {DEFAULT_POP_OUT})"
    )
    parser.add_argument(
        "--temporal-smooth",
        type=float,
        default=DEFAULT_TEMPORAL_SMOOTH,
        help=f"Temporal stability smoothing factor for video (default: {DEFAULT_TEMPORAL_SMOOTH})"
    )
    parser.add_argument(
        "--profile",
        choices=["balanced", "cinematic_depth", "subtle_parallax", "high_contrast", "movies_anime", "standard", "cinematic", "sharp_edges"],
        default=None,
        help="Preset profile (standard stereoscopic profiles: balanced, cinematic_depth, subtle_parallax, high_contrast)"
    )
    parser.add_argument(
        "--render-mode",
        choices=["right_only", "both"],
        default="right_only",
        help="Stereo render mode: right_only (Default: Left eye untouched, eliminates ghosting) or both"
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="Save the estimated depth map as a separate video (_depth.mp4) or image (_depth.png) for manual editing"
    )
    parser.add_argument(
        "--custom-depth",
        type=str,
        default=None,
        help="Path to pre-computed / manually edited depth map file. Skips AI inference for instant Fast Re-export!"
    )
    parser.add_argument(
        "-s", "--start-time",
        type=float,
        default=0.0,
        help="Start time in seconds for video conversion (default: 0.0)"
    )
    parser.add_argument(
        "-t", "--duration",
        type=float,
        default=0.0,
        help="Duration in seconds to convert (default: 0.0 = full video)"
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable automatic black bar letterbox detection"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel depth inference (default: 1, strict FIFO ordering)"
    )
    parser.add_argument(
        "--depth-stride",
        type=int,
        default=1,
        help="Compute depth every N frames while preserving 100%% genuine RGB frame motion (default: 1)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted conversion from the last frame-accurate checkpoint"
    )
    parser.add_argument(
        "--check-cadence",
        action="store_true",
        help="Run post-conversion visual cadence and temporal continuity diagnostic comparing 2D input vs 3D output"
    )
    parser.add_argument(
        "--repair-stutter",
        type=str,
        default=None,
        help="Optional repair mode: path to output file for repairing an already defective video with stutter/repeated frames"
    )

    args = parser.parse_args()

    from src.config import BIN_DIR
    from src.system_check import run_full_system_diagnostic
    from src.i18n import t, set_active_language

    set_active_language(args.lang)

    # If --check-system or no arguments passed, run diagnostic report
    if args.check_system or not args.input:
        diag = run_full_system_diagnostic(BIN_DIR, lang=args.lang)
        print("=" * 72)
        print(f"        {t('cli.title', lang=args.lang)}")
        print("=" * 72)
        print(f"{t('cli.os', lang=args.lang)}:   {diag['os_details']}")
        print(f"{t('cli.gpu', lang=args.lang)}: {diag['gpu_name']} [{diag['device']}]")
        print(f"{t('cli.encoder', lang=args.lang)}:   {diag['encoder_desc']} ({diag['encoder']})")
        print(f"{t('cli.ffmpeg', lang=args.lang)}:    {'✓ Yes' if diag['ffmpeg_installed'] else '✗ NOT FOUND'}" if args.lang != 'es' else f"{t('cli.ffmpeg', lang=args.lang)}:    {'✓ Sí' if diag['ffmpeg_installed'] else '✗ NO ENCONTRADO'}")
        print(f"{t('cli.spatial', lang=args.lang)}:         {'✓ Yes' if diag['spatial_installed'] else '○ No (Optional MV-HEVC)'}" if args.lang != 'es' else f"{t('cli.spatial', lang=args.lang)}:         {'✓ Sí' if diag['spatial_installed'] else '○ No (Opcional MV-HEVC)'}")
        print(f"{t('cli.status', lang=args.lang)}:      {diag['status'].upper()}")
        print("-" * 72)

        if not diag["missing_items"]:
            print(t("cli.all_ok", lang=args.lang))
        else:
            print(f"\n{t('cli.missing_header', lang=args.lang)}\n")
            for item in diag["missing_items"]:
                urg_label = {
                    "CRITICAL": t("diag.urgency_critical", lang=args.lang),
                    "RECOMMENDED": t("diag.urgency_recommended", lang=args.lang),
                    "OPTIONAL": t("diag.urgency_optional", lang=args.lang)
                }.get(item["urgency"], item["urgency"])

                print(f"[{urg_label}] {item['item']}")
                print(f"  • {t('cli.category', lang=args.lang)}:    {item.get('category', 'General')}")
                print(f"  • {t('cli.reason', lang=args.lang)}:  {item['reason']}")
                if item.get("command"):
                    print(f"  • {t('cli.command', lang=args.lang)}:      {item['command']}")
                if item.get("url"):
                    print(f"  • {t('cli.download', lang=args.lang)}: {item['url']}")
                if item.get("instructions"):
                    print(f"  • {t('cli.instructions', lang=args.lang)}:\n    {item['instructions'].replace(chr(10), chr(10) + '    ')}")
                print("-" * 72)

        if not args.input:
            print(f"\n{t('cli.usage_hint', lang=args.lang)}")
        sys.exit(0)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)

    if args.repair_stutter:
        print(f"[Repair Mode] Attempting cadence repair on: {input_path.name}")
        repair_defective_cadence(input_path, Path(args.repair_stutter).resolve(),
                                layout=args.format if args.format in ("sbs", "hsbs") else "2d")
        sys.exit(0)

    # Apply profile defaults if selected
    if args.profile and args.profile in PROFILES:
        prof = PROFILES[args.profile]
        args.depth_intensity = prof["divergence"]
        args.convergence = prof["convergence"]
        args.pop_out = prof["pop_out"]
        args.temporal_smooth = prof["temporal_smooth"]

    ext = input_path.suffix.lower()
    is_video = ext in VIDEO_EXTS
    is_image = ext in IMAGE_EXTS

    if not is_video and not is_image:
        print(f"Error: Unsupported file extension: {ext}")
        sys.exit(1)

    custom_depth_path = Path(args.custom_depth).resolve() if args.custom_depth else None

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        suffix = "_3d_spatial.mov" if (args.format == "spatial" and is_video) else f"_3d_{args.format}{ext}"
        output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}{suffix}"

    diag = run_full_system_diagnostic(BIN_DIR, lang=args.lang)

    print("=" * 70)
    print("        2D TO 3D STUDIO - OFFLINE STEREOSCOPIC CONVERTER")
    print("=" * 70)
    print(f"{t('cli.os', lang=args.lang)}: {diag['os_details']}")
    print(f"{t('cli.gpu', lang=args.lang)}:    {diag['gpu_name']} ({diag['device']})")
    print(f"{t('cli.encoder', lang=args.lang)}: {diag['encoder_desc']} [{diag['encoder']}]")
    print(f"Input:             {input_path}")
    print(f"Output:            {output_path}")
    print(f"Format:            {args.format.upper()}")
    print(f"Render Mode:       {args.render_mode.upper()} (Left eye untouched)")
    print(f"Parallax / Depth:  {args.depth_intensity} (Convergence: {args.convergence})")
    print(f"Save Depth Map:    {'Enabled' if args.save_depth else 'Disabled'}")
    if custom_depth_path:
        print(f"Fast Re-export:    {custom_depth_path}")
    if is_video and args.duration > 0:
        print(f"Quick Preview:     Start={args.start_time}s, Duration={args.duration}s")
    if is_video:
        if args.batch_size > 1:
            print(f"Batch Processing:  {args.batch_size} frames/batch (FIFO)")
        if args.depth_stride > 1:
            print(f"Depth Stride:      Every {args.depth_stride} frames (1:1 genuine RGB motion)")
        if args.resume:
            print(f"Resume Mode:       Enabled (Frame-accurate checkpoint)")
        if args.check_cadence:
            print(f"Cadence Diagnostic:Enabled")
    print("=" * 70)

    # If drivers or tools are missing, alert the user with links and commands
    if diag["missing_items"]:
        print("\n" + "!" * 70)
        print("  ⚠️  AVISO DE REQUISITOS / DRIVERS DEL SISTEMA")
        print("!" * 70)
        for item in diag["missing_items"]:
            urg = "CRÍTICO (Requerido)" if item["urgency"] == "CRITICAL" else "RECOMENDADO (Aceleración)"
            print(f"\n[!] {item['item']} - {urg}")
            print(f"    Razón:     {item['reason']}")
            if item.get("command"):
                print(f"    Comando:   {item['command']}")
            if item.get("url"):
                print(f"    Descarga:  {item['url']}")
            if item.get("instructions"):
                print(f"    Instrucción: {item['instructions']}")
        print("\n" + "!" * 70 + "\n")

        # If a critical component (like FFmpeg) is completely missing, cannot proceed
        if diag["status"] == "error":
            print("ERROR: Faltan dependencias críticas para continuar. Instálalas según las instrucciones de arriba.")
            sys.exit(1)

    # Initialize depth engine only if not using custom depth map
    depth_engine = None
    if not custom_depth_path:
        depth_engine = DepthEngine()
        depth_engine.temporal_filter.alpha = args.temporal_smooth

    synthesizer = StereoSynthesizer(
        divergence=args.depth_intensity,
        convergence=args.convergence,
        pop_out=args.pop_out,
        render_mode=args.render_mode,
        edge_refine=True
    )

    if is_image:
        process_image(
            input_path, output_path, depth_engine, synthesizer, args.format,
            custom_depth_path=custom_depth_path, save_depth=args.save_depth
        )
    else:
        process_video(
            input_path, output_path, depth_engine, synthesizer, args.format,
            auto_crop=not args.no_crop,
            start_time=args.start_time,
            duration=args.duration,
            custom_depth_path=custom_depth_path,
            save_depth=args.save_depth,
            batch_size=args.batch_size,
            depth_stride=args.depth_stride,
            resume=args.resume,
            check_cadence=args.check_cadence
        )

if __name__ == "__main__":
    main()
