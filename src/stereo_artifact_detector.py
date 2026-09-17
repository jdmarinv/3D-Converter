"""Shot-aware detector for DIBR deformation in packed SBS conversions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from tqdm import tqdm

from .preprocessor import get_media_info, read_video_frames


@dataclass
class ArtifactShot:
    shot: int
    start: float
    end: float
    peak_score: float
    mean_score: float
    samples: int
    pristine_eye: str
    reason: str


def _split(frame: np.ndarray, layout: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    if layout not in ("hsbs", "sbs") or w % 2:
        raise ValueError("Artifact detection requires an even-width HSBS or Full-SBS video")
    return frame[:, : w // 2], frame[:, w // 2 :]


def _gray_proxy(frame: np.ndarray, size=(320, 180)) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


def _align_translation(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    shift, response = cv2.phaseCorrelate(reference.astype(np.float32), candidate.astype(np.float32))
    if response < 0.05 or abs(shift[0]) > 32 or abs(shift[1]) > 8:
        return candidate
    matrix = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
    return cv2.warpAffine(candidate, matrix, (candidate.shape[1], candidate.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def frame_artifact_metrics(source_rgb: np.ndarray, converted_rgb: np.ndarray,
                           layout: str = "hsbs") -> dict:
    """Measure local edge damage after compensating ordinary global disparity."""
    source = _gray_proxy(source_rgb)
    eyes = [_gray_proxy(eye) for eye in _split(converted_rgb, layout)]
    maes = [float(np.mean(cv2.absdiff(source, eye))) for eye in eyes]
    pristine_index = int(np.argmin(maes))
    synthetic_index = 1 - pristine_index
    synthetic = _align_translation(source, eyes[synthetic_index])

    source_edges = cv2.Canny(source, 55, 140)
    synthetic_edges = cv2.Canny(synthetic, 55, 140)
    kernel = np.ones((3, 3), np.uint8)
    source_near = cv2.dilate(source_edges, kernel)
    synthetic_near = cv2.dilate(synthetic_edges, kernel)
    source_count = max(1, int(np.count_nonzero(source_edges)))
    missing_edges = np.count_nonzero((source_edges > 0) & (synthetic_near == 0)) / source_count
    invented_edges = np.count_nonzero((synthetic_edges > 0) & (source_near == 0)) / max(
        1, int(np.count_nonzero(synthetic_edges))
    )
    source_sharpness = float(cv2.Laplacian(source, cv2.CV_32F).var())
    synthetic_sharpness = float(cv2.Laplacian(synthetic, cv2.CV_32F).var())
    blur_loss = max(0.0, 1.0 - synthetic_sharpness / max(source_sharpness, 1.0))
    residual_p95 = float(np.percentile(cv2.absdiff(source, synthetic), 95)) / 255.0
    score = 100.0 * (0.38 * missing_edges + 0.27 * invented_edges +
                     0.20 * blur_loss + 0.15 * residual_p95)
    return {
        "score": round(score, 3),
        "pristine_eye": "left" if pristine_index == 0 else "right",
        "missing_edges": round(float(missing_edges), 4),
        "invented_edges": round(float(invented_edges), 4),
        "blur_loss": round(float(blur_loss), 4),
        "residual_p95": round(float(residual_p95), 4),
    }


def _histogram_distance(previous: np.ndarray | None, current: np.ndarray) -> float:
    hist = cv2.calcHist([current], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    if previous is None:
        return 0.0
    return float(cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA))


def detect_artifact_shots(source_2d: Path, converted_3d: Path, *, layout="hsbs",
                          sample_fps=2.0, sensitivity=3.0,
                          report_path: Path | None = None) -> dict:
    source_info = get_media_info(source_2d)
    converted_info = get_media_info(converted_3d)
    if abs(source_info["fps"] - converted_info["fps"]) > 0.01:
        raise ValueError("The source and conversion frame rates do not match")
    if abs(source_info["duration"] - converted_info["duration"]) > 1.0:
        raise ValueError("The source and conversion durations differ by more than one second")
    stride = max(1, round(source_info["fps"] / sample_fps))
    source_gen = read_video_frames(source_2d, force_cfr=source_info.get("is_vfr", False))
    converted_gen = read_video_frames(converted_3d, force_cfr=converted_info.get("is_vfr", False))
    samples = []
    shot_id = 0
    previous_hist = None
    previous_gray = None
    shot_start_time = 0.0
    total_samples = max(1, converted_info["nb_frames"] // stride)
    with tqdm(total=total_samples, desc="Scanning artifact shots", unit="sample") as progress:
        for (index, source), (converted_index, converted) in zip(source_gen, converted_gen):
            if index != converted_index:
                raise RuntimeError("Frame alignment was lost during artifact scan")
            if index % stride:
                continue
            gray = _gray_proxy(source)
            distance = _histogram_distance(previous_hist, gray)
            frame_delta = (float(np.mean(cv2.absdiff(previous_gray, gray))) / 255.0
                           if previous_gray is not None else 0.0)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)
            current_time = index / source_info["fps"]
            # A true camera cut normally produces a strong pixel or histogram
            # discontinuity. The six-second ceiling also prevents slow fades and
            # static logos from becoming a single movie-length pseudo-shot.
            if previous_hist is not None and (
                distance >= 0.20 or frame_delta >= 0.105 or current_time - shot_start_time >= 6.0
            ):
                shot_id += 1
                shot_start_time = current_time
            previous_hist = hist
            previous_gray = gray
            metrics = frame_artifact_metrics(source, converted, layout)
            samples.append({"frame": index, "time": current_time,
                            "shot": shot_id, **metrics})
            progress.update(1)

    scores = [item["score"] for item in samples]
    center = median(scores) if scores else 0.0
    mad = median([abs(value - center) for value in scores]) if scores else 0.0
    threshold = max(13.0, center + sensitivity * max(mad, 0.75))
    by_shot: dict[int, list[dict]] = {}
    for item in samples:
        by_shot.setdefault(item["shot"], []).append(item)
    duration = converted_info["duration"]
    shots = []
    ordered = sorted(by_shot.items())
    for position, (number, items) in enumerate(ordered):
        peak = max(item["score"] for item in items)
        flagged = [item for item in items if item["score"] >= threshold]
        if not flagged:
            continue
        start = items[0]["time"]
        next_start = ordered[position + 1][1][0]["time"] if position + 1 < len(ordered) else duration
        eyes = [item["pristine_eye"] for item in flagged]
        shots.append(ArtifactShot(
            shot=number,
            start=round(max(0.0, start - 0.15), 3),
            end=round(min(duration, next_start + 0.15), 3),
            peak_score=round(peak, 3),
            mean_score=round(sum(item["score"] for item in items) / len(items), 3),
            samples=len(flagged),
            pristine_eye=max(set(eyes), key=eyes.count),
            reason="edge loss, invented contours, blur, or high residual after disparity alignment",
        ))
    report = {
        "source_2d": str(Path(source_2d).resolve()),
        "converted_3d": str(Path(converted_3d).resolve()),
        "layout": layout,
        "sample_fps": sample_fps,
        "threshold": round(threshold, 3),
        "median_score": round(center, 3),
        "mad": round(mad, 3),
        "shots": [asdict(shot) for shot in shots],
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
