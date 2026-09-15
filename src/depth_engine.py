"""
Depth Estimation Engine with Temporal Filtering and Apple Silicon MPS acceleration.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Union, Tuple, List

from .config import DEVICE, DEFAULT_DEPTH_MODEL, DEFAULT_PATCH_SIZE
from .depth_model import DepthAnything, load_depth_model

class TemporalDepthFilter:
    """
    Temporal filter to prevent flicker (depth jitter) across video frames.
    Uses an adaptive exponential moving average with motion-aware edge preserving
    and automatic scene cut detection.
    """
    def __init__(self, alpha: float = 0.65, cut_threshold: float = 0.35):
        self.alpha = float(np.clip(alpha, 0.0, 0.95))
        self.cut_threshold = cut_threshold
        self.prev_depth: Optional[np.ndarray] = None

    def reset(self):
        self.prev_depth = None

    def filter(self, current_depth: np.ndarray, is_scene_cut: bool = False) -> np.ndarray:
        if self.prev_depth is None or self.alpha <= 0.0 or is_scene_cut:
            self.prev_depth = current_depth.copy()
            return current_depth

        # Calculate absolute difference to preserve fast motion edges
        diff = np.abs(current_depth - self.prev_depth)

        # Automatic scene cut detection: if average depth change is drastic, reset
        if np.mean(diff) > self.cut_threshold:
            self.prev_depth = current_depth.copy()
            return current_depth

        # Adapt alpha: lower smoothing where there is rapid movement to prevent ghosting
        adaptive_alpha = self.alpha * np.exp(-diff * 3.0)
        smoothed = adaptive_alpha * self.prev_depth + (1.0 - adaptive_alpha) * current_depth
        self.prev_depth = smoothed.copy()
        return smoothed

class DepthDecimator:
    """
    Manages reduced-frequency depth estimation (e.g. depth computed every N frames).
    Allows reusing or interpolating depth maps across frames while guaranteeing
    that stereo synthesis ALWAYS receives the genuine, current RGB frame.
    NEVER replaces or freezes the RGB frame!
    """
    def __init__(self, stride: int = 1, interpolate: bool = True):
        self.stride = max(1, stride)
        self.interpolate = interpolate
        self.last_depth: Optional[np.ndarray] = None

    def should_compute(self, frame_idx: int) -> bool:
        if self.stride <= 1:
            return True
        return (frame_idx % self.stride) == 0

    def register_computed_depth(self, depth: np.ndarray):
        self.last_depth = depth

    def get_depth_for_frame(self, frame_idx: int, newly_computed: Optional[np.ndarray] = None) -> np.ndarray:
        if newly_computed is not None:
            self.last_depth = newly_computed
            return newly_computed
        if self.last_depth is not None:
            return self.last_depth
        raise RuntimeError(f"No prior depth map available for frame index {frame_idx}")

class DepthEngine:
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        patch_size: int = DEFAULT_PATCH_SIZE
    ):
        self.device = device or DEVICE
        self.model_path = Path(model_path or DEFAULT_DEPTH_MODEL)
        self.patch_size = patch_size
        self.temporal_filter = TemporalDepthFilter()

        # ImageNet normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        print(f"[DepthEngine] Loading model weights from {self.model_path} on {self.device}...")
        self.model = load_depth_model(self.model_path, self.device)
        print("[DepthEngine] Model ready for high-fidelity offline inference.")

    def _prepare_tensor(self, rgb_image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Resize image to multiples of 14 matching target patch_size while preserving aspect ratio.
        """
        orig_h, orig_w = rgb_image.shape[:2]
        
        # Scale keeping aspect ratio such that the larger dimension matches patch_size
        scale = self.patch_size / max(orig_h, orig_w)
        new_h = int(round(orig_h * scale / 14.0)) * 14
        new_w = int(round(orig_w * scale / 14.0)) * 14
        new_h = max(14, new_h)
        new_w = max(14, new_w)

        resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Convert HWC uint8 [0, 255] to BCHW float [0, 1]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor, (orig_h, orig_w)

    @torch.no_grad()
    def estimate_depth_batch(
        self,
        rgb_images: List[np.ndarray],
        apply_temporal_smoothing: bool = False
    ) -> List[np.ndarray]:
        """
        Estimates normalized depth maps [0.0, 1.0] for a batch of RGB frames.
        Guarantees strict FIFO ordering and deterministic sequential temporal filtering.
        """
        if not rgb_images:
            return []

        tensors = []
        orig_sizes = []
        for img in rgb_images:
            t, (h, w) = self._prepare_tensor(img)
            tensors.append(t)
            orig_sizes.append((h, w))

        batch_tensor = torch.cat(tensors, dim=0)
        raw_depths = self.model(batch_tensor)

        depth_maps = []
        for i in range(len(rgb_images)):
            orig_h, orig_w = orig_sizes[i]
            d_up = F.interpolate(
                raw_depths[i:i+1], size=(orig_h, orig_w), mode="bilinear", align_corners=True
            )
            depth_np = d_up.squeeze().cpu().numpy()

            d_min = np.percentile(depth_np, 1)
            d_max = np.percentile(depth_np, 99)
            if d_max - d_min > 1e-6:
                depth_norm = np.clip((depth_np - d_min) / (d_max - d_min), 0.0, 1.0)
            else:
                depth_norm = np.zeros_like(depth_np, dtype=np.float32)

            if apply_temporal_smoothing:
                depth_norm = self.temporal_filter.filter(depth_norm)

            depth_maps.append(depth_norm.astype(np.float32))

        return depth_maps

    @torch.no_grad()
    def estimate_depth(
        self,
        rgb_image: np.ndarray,
        apply_temporal_smoothing: bool = False
    ) -> np.ndarray:
        """
        Estimates a normalized depth map [0.0, 1.0] for a single RGB frame.
        """
        return self.estimate_depth_batch([rgb_image], apply_temporal_smoothing=apply_temporal_smoothing)[0]
