"""Depth model catalogue and lazy inference backends.

Large third-party models are deliberately loaded only after the user selects one.
This keeps the default installation small and lets the portal report a useful
dependency error instead of failing at startup.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class DepthModelSpec:
    key: str
    name: str
    backend: str
    license: str
    repo_id: Optional[str] = None
    filename: Optional[str] = None
    metric: bool = False
    input_size: Optional[tuple[int, int]] = None  # width, height
    notes: str = ""

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["input_size"] = list(self.input_size) if self.input_size else None
        return value


DEPTH_MODELS: tuple[DepthModelSpec, ...] = (
    DepthModelSpec("builtin-da-v2-small", "Depth Anything v2 Small (Built-in)", "builtin", "Apache-2.0"),
    DepthModelSpec("da-v2-small-onnx", "Depth Anything v2 Small (ONNX)", "onnx", "Apache-2.0", "onnx-community/depth-anything-v2-small-ONNX", "onnx/model.onnx"),
    DepthModelSpec("distill-any-depth-large-hf", "Distill-Any-Depth Large", "hf", "MIT", "xingyang1/Distill-Any-Depth-Large-hf"),
    DepthModelSpec("distill-any-depth-small-onnx", "Distill-Any-Depth Small (ONNX)", "onnx", "MIT", "FuryTMP/Distill-Any-Depth-Small-onnx", "Distill Any Depth Small/model.onnx", input_size=(518, 518)),
    DepthModelSpec("distill-any-depth-base-onnx", "Distill-Any-Depth Base (ONNX)", "onnx", "MIT", "FuryTMP/Distill-Any-Depth-Base-onnx", "Distill Any Depth Base/model.onnx", input_size=(518, 518)),
    DepthModelSpec("distill-any-depth-large-onnx", "Distill-Any-Depth Large (ONNX)", "onnx", "MIT", "FuryTMP/Distill-Any-Depth-Large-onnx", "Distill Any Depth Large/model.onnx", input_size=(518, 518)),
    DepthModelSpec("video-depth-anything-onnx", "Video Depth Anything (ONNX)", "onnx", "Apache-2.0", "FuryTMP/Video-Depth-Anything-L-ONNX-512x288", "VideoDepthAnything/model.onnx", input_size=(512, 288)),
    DepthModelSpec("da3-small", "DA3-SMALL", "da3", "Apache-2.0", "depth-anything/DA3-SMALL"),
    DepthModelSpec("da3-base", "DA3-BASE", "da3", "Apache-2.0", "depth-anything/DA3-BASE"),
    DepthModelSpec("da3mono-large", "DA3MONO-LARGE", "da3", "Apache-2.0", "depth-anything/DA3MONO-LARGE"),
    DepthModelSpec("da3metric-large", "DA3METRIC-LARGE", "da3", "Apache-2.0", "depth-anything/DA3METRIC-LARGE", metric=True),
    DepthModelSpec("da3-large-1.1", "DA3-LARGE-1.1", "da3", "Apache-2.0", "depth-anything/DA3-LARGE-1.1"),
    DepthModelSpec("da-v2-metric-outdoor-large", "Depth Anything v2 Metric Outdoor (Large)", "hf", "Apache-2.0", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", metric=True),
    DepthModelSpec("da-v1-small", "Depth Anything v1 Small", "hf", "Apache-2.0", "LiheYoung/depth-anything-small-hf"),
    DepthModelSpec("da-v1-base", "Depth Anything v1 Base", "hf", "Apache-2.0", "LiheYoung/depth-anything-base-hf"),
    DepthModelSpec("da-v1-large", "Depth Anything v1 Large", "hf", "Apache-2.0", "LiheYoung/depth-anything-large-hf"),
    DepthModelSpec("zoedepth-n", "ZoeDepth N (NYU)", "zoe", "MIT", "isl-org/ZoeDepth", "ZoeD_N", metric=True),
    DepthModelSpec("zoedepth-k", "ZoeDepth K (KITTI)", "zoe", "MIT", "isl-org/ZoeDepth", "ZoeD_K", metric=True),
    DepthModelSpec("zoedepth-nk", "ZoeDepth NK (Recommended)", "zoe", "MIT", "isl-org/ZoeDepth", "ZoeD_NK", metric=True),
    DepthModelSpec("dpt-beit-large-512", "DPT BEiT Large 512", "hf", "MIT", "Intel/dpt-beit-large-512"),
    DepthModelSpec(
        "prompt-da-vits-transparent", "Prompt Depth Anything VITS Transparent", "prompt_hf", "Apache-2.0",
        "depth-anything/prompt-depth-anything-vits-transparent-hf", metric=True,
        notes="Uses the built-in estimator as the dense depth prompt when no LiDAR prompt is supplied.",
    ),
)

DEPTH_MODEL_BY_KEY = {item.key: item for item in DEPTH_MODELS}
DEFAULT_DEPTH_MODEL_KEY = "builtin-da-v2-small"


def get_depth_model_spec(key: Optional[str]) -> DepthModelSpec:
    requested = key or DEFAULT_DEPTH_MODEL_KEY
    try:
        return DEPTH_MODEL_BY_KEY[requested]
    except KeyError as exc:
        choices = ", ".join(DEPTH_MODEL_BY_KEY)
        raise ValueError(f"Unknown depth model '{requested}'. Available models: {choices}") from exc


def _missing(package: str, extra: str) -> RuntimeError:
    return RuntimeError(
        f"The selected depth model needs {package}. Install its backend with: "
        f"python -m pip install -r {extra}"
    )


class ExternalDepthBackend:
    """Uniform adapter returning unnormalised HxW depth arrays in source order."""

    def __init__(self, spec: DepthModelSpec, device: torch.device, profile_size: int = 518):
        self.spec = spec
        self.device = device
        self.profile_size = profile_size
        self.processor = None
        self.model = None
        self.session = None
        self.model_path = None
        self._load()

    def _create_onnx_session(self, model_path: str, force_cpu: bool = False, disable_opt: bool = False) -> Any:
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        import onnxruntime as ort
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()
        opts = ort.SessionOptions()
        if disable_opt:
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        available = set(ort.get_available_providers())
        if force_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            preferred = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in preferred if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]

        return ort.InferenceSession(model_path, sess_options=opts, providers=providers)

    def _load(self) -> None:
        if self.spec.backend == "onnx":
            try:
                os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
                import onnxruntime as ort
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise _missing("ONNX Runtime and huggingface_hub", "requirements-pro.txt") from exc
            patterns = [self.spec.filename, f"{self.spec.filename}_data"]
            snapshot = snapshot_download(self.spec.repo_id, allow_patterns=patterns)
            self.model_path = str(Path(snapshot) / self.spec.filename)
            
            # VideoDepthAnything is exported with a 5D temporal transformer graph
            # that is incompatible with CoreML partitioners and standard graph optimization.
            if self.spec.key == "video-depth-anything-onnx":
                strategies = [(True, True)]
            else:
                strategies = [
                    (False, False),  # Preferred providers, default optimization
                    (False, True),   # Preferred providers, ORT_DISABLE_ALL
                    (True, False),   # CPU only, default optimization
                    (True, True),    # CPU only, ORT_DISABLE_ALL
                ]
            last_exc = None
            for force_cpu, disable_opt in strategies:
                try:
                    self.session = self._create_onnx_session(self.model_path, force_cpu=force_cpu, disable_opt=disable_opt)
                    return
                except Exception as exc:
                    last_exc = exc
                    continue
            if last_exc:
                raise last_exc
            return

        if self.spec.backend in {"hf", "prompt_hf"}:
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            except ImportError as exc:
                raise _missing("Transformers", "requirements-pro.txt") from exc
            if self.spec.backend == "prompt_hf":
                try:
                    from transformers import PromptDepthAnythingForDepthEstimation, PromptDepthAnythingImageProcessor
                except ImportError as exc:
                    raise RuntimeError("Prompt Depth Anything requires a recent Transformers release from requirements-pro.txt") from exc
                self.processor = PromptDepthAnythingImageProcessor.from_pretrained(self.spec.repo_id)
                self.model = PromptDepthAnythingForDepthEstimation.from_pretrained(self.spec.repo_id)
            else:
                self.processor = AutoImageProcessor.from_pretrained(self.spec.repo_id)
                self.model = AutoModelForDepthEstimation.from_pretrained(self.spec.repo_id)
            self.model.to(self.device).eval()
            return

        if self.spec.backend == "da3":
            # DA3 imports Open3D, which bundles a second OpenMP runtime on macOS.
            # Allow both runtimes before importing its API.
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            try:
                from depth_anything_3.api import DepthAnything3
            except ImportError as exc:
                raise _missing("Depth Anything 3", "requirements-da3.txt") from exc
            self.model = DepthAnything3.from_pretrained(self.spec.repo_id)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            return

        if self.spec.backend == "zoe":
            # ZoeDepth checkpoints contain obsolete relative-position index
            # buffers that recent timm versions no longer register. Build the
            # official architecture, then discard only those derived buffers.
            self.model = torch.hub.load(
                "isl-org/ZoeDepth", self.spec.filename, pretrained=False, trust_repo=True
            )
            checkpoint_name = {
                "ZoeD_N": "ZoeD_M12_N.pt",
                "ZoeD_K": "ZoeD_M12_K.pt",
                "ZoeD_NK": "ZoeD_M12_NK.pt",
            }[self.spec.filename]
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
            if not checkpoint.is_file():
                raise RuntimeError(f"Missing ZoeDepth checkpoint: {checkpoint}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = state.get("model", state)
            state = {key.removeprefix("module."): value for key, value in state.items()
                     if not key.endswith("relative_position_index")}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if unexpected or any(not key.endswith("relative_position_index") for key in missing):
                raise RuntimeError(f"ZoeDepth checkpoint mismatch; missing={missing}, unexpected={unexpected}")
            # MiDaS' BEiT compatibility shim targets old timm's ``drop_path``;
            # current timm splits it into ``drop_path1`` and ``drop_path2``.
            for module in self.model.modules():
                if not hasattr(module, "drop_path") and hasattr(module, "drop_path1"):
                    module.drop_path = module.drop_path1
            self.model.to(self.device).eval()
            return

        raise ValueError(f"Unsupported depth backend: {self.spec.backend}")

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        value = np.asarray(value)
        while value.ndim > 2:
            value = value[0]
        return value.astype(np.float32)

    def _onnx_input(self, rgb: np.ndarray) -> np.ndarray:
        inp = self.session.get_inputs()[0]
        shape = inp.shape
        nchw = len(shape) == 4 and (shape[1] == 3 or not isinstance(shape[-1], int) or shape[-1] != 3)
        if self.spec.input_size:
            width, height = self.spec.input_size
        elif nchw and isinstance(shape[2], int) and isinstance(shape[3], int):
            height, width = shape[2], shape[3]
        elif not nchw and isinstance(shape[1], int) and isinstance(shape[2], int):
            height, width = shape[1], shape[2]
        else:
            scale = self.profile_size / max(rgb.shape[:2])
            height = max(14, round(rgb.shape[0] * scale / 14) * 14)
            width = max(14, round(rgb.shape[1] * scale / 14) * 14)
        image = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        if nchw:
            image = image.transpose(2, 0, 1)
        return image[None].astype(np.float32)

    def _infer_onnx(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        input_meta = self.session.get_inputs()[0]
        input_name = input_meta.name
        shape = input_meta.shape
        is_5d = len(shape) == 5

        if is_5d:
            width, height = self.spec.input_size or (512, 288)
            processed = []
            for img in images:
                resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
                norm = (resized - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
                processed.append(norm.transpose(2, 0, 1))

            n_frames = len(processed)
            chunk_size = 8
            results = []
            for i in range(0, n_frames, chunk_size):
                chunk = processed[i:i + chunk_size]
                actual_len = len(chunk)
                if actual_len < chunk_size:
                    chunk = chunk + [chunk[-1]] * (chunk_size - actual_len)
                tensor = np.stack(chunk, axis=0)[None].astype(np.float32)
                try:
                    out = self.session.run(None, {input_name: tensor})[0]
                except Exception:
                    self.session = self._create_onnx_session(self.model_path, force_cpu=True, disable_opt=True)
                    out = self.session.run(None, {input_name: tensor})[0]
                for f in range(actual_len):
                    results.append(self._array(out[0, f]))
            return results

        results = []
        for image in images:
            feed = {input_name: self._onnx_input(image)}
            try:
                output = self.session.run(None, feed)[0]
            except Exception:
                self.session = self._create_onnx_session(self.model_path, force_cpu=True, disable_opt=True)
                output = self.session.run(None, feed)[0]
            results.append(self._array(output))
        return results

    @torch.inference_mode()
    def infer(self, images: Sequence[np.ndarray], prompt_depths: Optional[Sequence[np.ndarray]] = None) -> list[np.ndarray]:
        if self.spec.backend == "onnx":
            return self._infer_onnx(images)

        if self.spec.backend == "da3":
            prediction = self.model.inference(list(images))
            values = prediction.depth
            return [self._array(values[i]) for i in range(len(images))]

        if self.spec.backend == "zoe":
            results = []
            for image in images:
                tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div(255).to(self.device)
                value = self.model.infer(tensor)
                results.append(self._array(value))
            return results

        from PIL import Image
        pil_images = [Image.fromarray(image) for image in images]
        kwargs: dict[str, Any] = {"images": pil_images, "return_tensors": "pt"}
        if self.spec.backend == "prompt_hf":
            if prompt_depths is None:
                raise ValueError("Prompt Depth Anything requires prompt depth maps")
            kwargs["prompt_depth"] = [Image.fromarray(np.asarray(d, dtype=np.float32), mode="F") for d in prompt_depths]
        inputs = self.processor(**kwargs)
        inputs = {name: value.to(self.device) if hasattr(value, "to") else value for name, value in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = [(image.shape[0], image.shape[1]) for image in images]
        if hasattr(self.processor, "post_process_depth_estimation"):
            post = self.processor.post_process_depth_estimation(outputs, target_sizes=target_sizes)
            return [self._array(item["predicted_depth"]) for item in post]
        return [self._array(outputs.predicted_depth[i]) for i in range(len(images))]
