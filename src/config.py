import os
import sys
import shutil
from pathlib import Path
import torch

from .system_check import detect_os, detect_hardware_gpu, run_full_system_diagnostic

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
BIN_DIR = PROJECT_ROOT / "bin"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hardware acceleration detection (CUDA / MPS / CPU)
def get_optimal_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_optimal_device()
OS_NAME = detect_os()
EXE_SUFFIX = ".exe" if OS_NAME == "windows" else ""

# Resolve binary paths strictly from project bin/, falling back to system PATH
def resolve_binary(name: str) -> str:
    base_name = f"{name}{EXE_SUFFIX}"
    # 1. Local project bin/ (standalone binaries)
    local_bin = BIN_DIR / base_name
    if local_bin.exists() and (OS_NAME == "windows" or os.access(local_bin, os.X_OK)):
        return str(local_bin)
    
    # 2. Also check without extension on Windows if alias exists
    if OS_NAME == "windows":
        local_raw = BIN_DIR / name
        if local_raw.exists():
            return str(local_raw)

    # 3. System PATH
    system_path = shutil.which(base_name) or shutil.which(name)
    if system_path:
        return system_path
    
    return str(local_bin)

FFMPEG_BIN = resolve_binary("ffmpeg")
FFPROBE_BIN = resolve_binary("ffprobe")
SPATIAL_BIN = resolve_binary("spatial")
PIC_COMBINER_BIN = resolve_binary("run_picCombiner")

# Resolve model checkpoints strictly inside project models/
DEFAULT_DEPTH_MODEL = MODELS_DIR / "depth_anything_v2_vits.safetensors"

# Default 3D Conversion Hyperparameters
DEFAULT_DIVERGENCE = 0.022          # Separation / stereo parallax intensity (0.015 - 0.05)
DEFAULT_CONVERGENCE = 0.50          # Zero-parallax plane (0.0=all pop-out, 1.0=all deep in screen, 0.5=balanced)
DEFAULT_POP_OUT = 0.0               # Extra pop-out bias
DEFAULT_TEMPORAL_SMOOTH = 0.70      # Temporal smoothing factor (0.0=no smoothing, 0.95=heavy smoothing)
DEFAULT_PATCH_SIZE = 518            # ViT patch resolution
