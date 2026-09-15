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
        self.prev_gray: Optional[np.ndarray] = None

    def reset(self):
        self.prev_depth = None
        self.prev_gray = None

    def filter(
        self,
        current_depth: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        is_scene_cut: bool = False
    ) -> np.ndarray:
        gray = None
        if rgb is not None:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.prev_depth is None or self.alpha <= 0.0 or is_scene_cut:
            self.prev_depth = current_depth.copy()
            self.prev_gray = None if gray is None else gray.copy()
            return current_depth

        aligned_previous = self.prev_depth
        confidence = 1.0
        if gray is not None and self.prev_gray is not None:
            # Track scene content at quarter resolution.  Warping the previous
            # depth into the current frame prevents a moving object from
            # inheriting the background depth that occupied the same pixels.
            h, w = gray.shape
            flow_w, flow_h = max(32, w // 4), max(24, h // 4)
            current_small = cv2.resize(gray, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
            previous_small = cv2.resize(self.prev_gray, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
            flow = cv2.calcOpticalFlowFarneback(
                current_small, previous_small, None, 0.5, 3, 15, 3, 5, 1.1, 0
            )
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= w / flow_w
            flow[..., 1] *= h / flow_h
            grid_x, grid_y = np.meshgrid(
                np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
            )
            map_x = grid_x + flow[..., 0]
            map_y = grid_y + flow[..., 1]
            aligned_previous = cv2.remap(
                self.prev_depth, map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            previous_gray_aligned = cv2.remap(
                self.prev_gray, map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            photometric_error = np.abs(gray.astype(np.float32) - previous_gray_aligned.astype(np.float32))
            confidence = np.exp(-photometric_error / 24.0)

        # Preserve newly revealed regions and fast-changing object boundaries.
        diff = np.abs(current_depth - aligned_previous)

        # Automatic scene cut detection: if average depth change is drastic, reset
        image_cut = gray is not None and self.prev_gray is not None and np.mean(
            np.abs(gray.astype(np.float32) - self.prev_gray.astype(np.float32))
        ) > 55.0
        if np.mean(diff) > self.cut_threshold or image_cut:
            self.prev_depth = current_depth.copy()
            self.prev_gray = None if gray is None else gray.copy()
            return current_depth

        adaptive_alpha = self.alpha * np.exp(-diff * 4.0) * confidence
        smoothed = adaptive_alpha * aligned_previous + (1.0 - adaptive_alpha) * current_depth
        self.prev_depth = smoothed.copy()
        self.prev_gray = None if gray is None else gray.copy()
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
        patch_size: Optional[int] = None,
        depth_profile: str = "balanced"
    ):
        self.device = device or DEVICE
        self.model_path = Path(model_path or DEFAULT_DEPTH_MODEL)
        self.depth_profile = depth_profile

        # Configure patch size based on depth_profile if patch_size not explicitly given
        if patch_size is not None:
            self.patch_size = patch_size
        elif depth_profile == "fast":
            self.patch_size = 392
        else:
            self.patch_size = DEFAULT_PATCH_SIZE

        self.temporal_filter = TemporalDepthFilter()

        # ImageNet normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        print(f"[DepthEngine] Loading model weights from {self.model_path} on {self.device} (Profile: {self.depth_profile}, Patch: {self.patch_size})...")
        self.model = load_depth_model(self.model_path, self.device)
        print("[DepthEngine] Model ready for high-fidelity offline inference.")

    @staticmethod
    def regularize_depth(rgb: np.ndarray, depth: np.ndarray, radius: int = 9) -> np.ndarray:
        """Smooth object interiors while keeping disparity discontinuities on RGB edges."""
        guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        source = depth.astype(np.float32)
        size = (radius * 2 + 1, radius * 2 + 1)
        mean_i = cv2.boxFilter(guide, -1, size, borderType=cv2.BORDER_REFLECT)
        mean_p = cv2.boxFilter(source, -1, size, borderType=cv2.BORDER_REFLECT)
        corr_i = cv2.boxFilter(guide * guide, -1, size, borderType=cv2.BORDER_REFLECT)
        corr_ip = cv2.boxFilter(guide * source, -1, size, borderType=cv2.BORDER_REFLECT)
        var_i = corr_i - mean_i * mean_i
        cov_ip = corr_ip - mean_i * mean_p
        a = cov_ip / (var_i + 2.5e-3)
        b = mean_p - a * mean_i
        mean_a = cv2.boxFilter(a, -1, size, borderType=cv2.BORDER_REFLECT)
        mean_b = cv2.boxFilter(b, -1, size, borderType=cv2.BORDER_REFLECT)
        guided = mean_a * guide + mean_b

        # Flat-color interiors (especially animation) should move as coherent
        # surfaces. Preserve the guided result around visible outlines.
        grad_x = cv2.Scharr(guide, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(guide, cv2.CV_32F, 0, 1)
        edge_strength = cv2.GaussianBlur(cv2.magnitude(grad_x, grad_y), (0, 0), 1.2)
        edge_weight = np.clip(edge_strength / 2.0, 0.0, 1.0)
        interior = cv2.bilateralFilter(guided.astype(np.float32), 9, 0.08, 7)
        result = edge_weight * guided + (1.0 - edge_weight) * (0.25 * guided + 0.75 * interior)
        return np.clip(result, 0.0, 1.0).astype(np.float32)

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

            depth_norm = self.regularize_depth(rgb_images[i], depth_norm)
            if apply_temporal_smoothing:
                depth_norm = self.temporal_filter.filter(depth_norm, rgb=rgb_images[i])

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
