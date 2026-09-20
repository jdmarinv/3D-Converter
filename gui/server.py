"""
FastAPI Local Backend Server for 2D to 3D Offline Converter GUI.
Provides native macOS file pickers, media inspection, and real-time SSE progress streaming.
"""
import sys
import os
import shutil
import asyncio
import subprocess
import threading
import time
import json
import signal
from collections import deque
from typing import Literal
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE, DEFAULT_OUTPUT_DIR, BIN_DIR
from src.preprocessor import get_media_info, has_audio_stream
from src.system_check import run_full_system_diagnostic, detect_os
from src.i18n import get_available_locales, load_locale
from src.depth_models import DEPTH_MODELS, DEFAULT_DEPTH_MODEL_KEY, get_depth_model_spec
from convert_3d import PROFILES

app = FastAPI(title="2D to 3D Converter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

HTML_PATH = Path(__file__).resolve().parent / "index.html"

# Global state for running conversion task
conversion_state = {
    "status": "idle",       # "idle", "running", "completed", "error"
    "percent": 0.0,
    "current_frame": 0,
    "total_frames": 0,
    "fps": 0.0,
    "eta": "",
    "log": "Ready.",
    "output_file": "",
    "depth_file": "",
    "process": None,
    "input_file": "",
    "start_time": 0.0,
    "source_fps": 0.0,
    "preview_frame": -1,
    "preview_jpeg": None
}

selected_media = {"path": "", "fps": 0.0}
preview_lock = threading.Lock()
history_thumbnail_cache: dict[str, tuple[int, int, bytes]] = {}
history_thumbnail_lock = threading.Lock()

class ConversionRequest(BaseModel):
    operation: Literal["convert", "repair", "stereo_repair"] = "convert"
    repair_layout: Literal["2d", "sbs", "hsbs"] = "hsbs"
    repair_source_path: Optional[str] = None
    repair_ranges: str = "auto"
    check_cadence: bool = True
    input_path: str
    output_path: Optional[str] = None
    format: str = "hsbs"
    profile: Optional[str] = "balanced"
    render_mode: str = "both"
    depth_intensity: float = 0.025
    strength_3d: Optional[float] = None
    style_3d: Optional[str] = "natural"
    depth_profile: Optional[str] = "balanced"
    depth_model: str = DEFAULT_DEPTH_MODEL_KEY
    depth_stride: int = 1
    convergence: float = 0.50
    pop_out: float = 0.0
    temporal_smooth: float = 0.65
    auto_crop: bool = True
    save_depth: bool = False
    custom_depth: Optional[str] = None
    start_time: float = Field(default=0.0, ge=0)
    duration: float = Field(default=0.0, ge=0)

    @field_validator("depth_model")
    @classmethod
    def validate_depth_model(cls, value: str) -> str:
        return get_depth_model_spec(value).key

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/locales")
async def get_locales():
    """
    Returns list of installed translation files so UI can populate the language selector dynamically.
    """
    return get_available_locales()

@app.get("/api/depth-models")
async def get_depth_models():
    return {"default": DEFAULT_DEPTH_MODEL_KEY, "models": [item.public_dict() for item in DEPTH_MODELS]}

@app.get("/api/locales/{code}")
async def get_locale_data(code: str):
    """
    Returns translation dictionary JSON for requested locale code.
    """
    return load_locale(code)

@app.get("/api/system-diagnostics")
async def get_system_diagnostics(lang: str = "en"):
    """
    Returns full hardware diagnostics, GPU detection, encoder status and missing items with links in the requested language.
    """
    diag = run_full_system_diagnostic(BIN_DIR, lang=lang)
    return diag

@app.post("/api/pick-file")
async def pick_file(request: Request):
    """
    Opens native OS file selection dialog (macOS osascript, Windows PowerShell Forms, Linux Zenity).
    """
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    file_type = body.get("type", "media")
    os_name = detect_os()
    
    title = "Select Depth Map" if file_type == "depth" else "Select 2D Media (Video or Image)"

    if os_name == "macos":
        osa_cmd = f'''
        set chosenFile to choose file with prompt "{title}" of type {{"public.movie", "public.image", "public.data"}}
        return POSIX path of chosenFile
        '''
        try:
            res = subprocess.run(["osascript", "-e", osa_cmd], capture_output=True, text=True)
            path = res.stdout.strip()
            if path:
                return {"path": path}
        except Exception:
            pass

    elif os_name == "windows":
        ps_cmd = f'''
        Add-Type -AssemblyName System.Windows.Forms
        $f = New-Object System.Windows.Forms.OpenFileDialog
        $f.Title = "{title}"
        $f.Filter = "Media Files|*.mp4;*.mov;*.mkv;*.avi;*.png;*.jpg;*.jpeg;*.webp|All Files|*.*"
        if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
            Write-Output $f.FileName
        }}
        '''
        try:
            res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
            path = res.stdout.strip()
            if path:
                return {"path": path}
        except Exception:
            pass

    elif os_name == "linux":
        zenity = shutil.which("zenity")
        if zenity:
            try:
                res = subprocess.run([zenity, "--file-selection", f"--title={title}"], capture_output=True, text=True)
                path = res.stdout.strip()
                if path:
                    return {"path": path}
            except Exception:
                pass

        kdialog = shutil.which("kdialog")
        if kdialog:
            try:
                res = subprocess.run([kdialog, "--getopenfilename", ".", f"--title {title}"], capture_output=True, text=True)
                path = res.stdout.strip()
                if path:
                    return {"path": path}
            except Exception:
                pass

        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            selected = filedialog.askopenfilename(title=title)
            root.destroy()
            if selected:
                return {"path": selected}
        except Exception:
            pass

    return {"path": ""}

@app.post("/api/probe-file")
async def probe_file(request: Request):
    """
    Inspects media file dimensions, fps, duration, and audio.
    """
    data = await request.json()
    p = Path(data.get("path", "")).resolve()
    if not p.exists():
        return {"error": "File does not exist"}

    ext = p.suffix.lower()
    is_video = ext in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    is_image = ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    if is_image:
        import cv2
        img = cv2.imread(str(p))
        if img is None:
            return {"error": "Invalid image"}
        h, w = img.shape[:2]
        selected_media.update(path=str(p), fps=0.0)
        return {
            "type": "image",
            "name": p.name,
            "width": w,
            "height": h,
            "fps": 0,
            "duration": 0,
            "frames": 1,
            "audio": False,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2)
        }
    elif is_video:
        try:
            info = get_media_info(p)
            audio = has_audio_stream(p)
            selected_media.update(path=str(p), fps=info["fps"])
            return {
                "type": "video",
                "name": p.name,
                "width": info["width"],
                "height": info["height"],
                "fps": round(info["fps"], 2),
                "fps_rational": info.get("fps_rational", f"{round(info['fps'], 2)}"),
                "is_vfr": info.get("is_vfr", False),
                "duration": round(info["duration"], 2),
                "frames": info["nb_frames"],
                "audio": audio,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {"error": str(e)}

    return {"error": "Unsupported file format"}

state_lock = threading.Lock()

def resolve_output(req):
    source = Path(req.input_path).expanduser().resolve()
    is_video = source.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    def time_label(value: float) -> str:
        return f"{value:g}".replace(".", "p")
    clip_tag = ""
    if is_video and req.duration > 0:
        clip_tag = f"_{time_label(req.start_time)}s-{time_label(req.start_time + req.duration)}s"
    suffix = ("_repaired.mp4" if req.operation == "repair" else
              ("_stereo_repaired.mp4" if req.operation == "stereo_repair" else
              (f"{clip_tag}_3d_spatial.mov" if req.format == "spatial" else f"{clip_tag}_3d_{req.format}.mp4")) if is_video else f"_3d_{req.format}{source.suffix}"
             )
    output = Path(req.output_path).expanduser().resolve() if req.output_path else DEFAULT_OUTPUT_DIR / (source.stem + suffix)
    if not req.output_path and output.exists():
        base_stem = output.stem
        for index in range(2, 10000):
            candidate = output.with_name(f"{base_stem}_{index}{output.suffix}")
            if not candidate.exists():
                output = candidate
                break
    return source, output, is_video


def build_conversion_command(req):
    source, output, is_video = resolve_output(req)
    if req.operation == "repair":
        return [sys.executable, "-u", "-m", "src.cadence_repair", "-i", str(source),
                "-o", str(output), "--layout", req.repair_layout,
                "--start-time", str(req.start_time), "--duration", str(req.duration)]
    if req.operation == "stereo_repair":
        cmd = [sys.executable, "-u", "-m", "src.stereo_artifact_repair",
               "--source-2d", str(Path(req.repair_source_path).expanduser().resolve()),
               "--converted-3d", str(source), "--output", str(output),
               "--layout", req.repair_layout, "--repair-ranges", req.repair_ranges,
               "--depth-intensity", str(req.depth_intensity),
               "--convergence", str(req.convergence), "--pop-out", str(req.pop_out),
               "--temporal-smooth", str(req.temporal_smooth),
               "--style-3d", req.style_3d or "natural",
               "--depth-profile", req.depth_profile or "balanced"]
        if not req.auto_crop:
            cmd.append("--no-crop")
        return cmd
    intensity = req.depth_intensity
    if req.strength_3d is not None:
        intensity = round(req.strength_3d * 0.005, 4)

    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "convert_3d.py"),
           "-i", str(source), "-o", str(output), "-f", req.format,
           "--render-mode", req.render_mode, "--depth-intensity", str(intensity),
           "--convergence", str(req.convergence), "--pop-out", str(req.pop_out),
           "--temporal-smooth", str(req.temporal_smooth)]
    if req.strength_3d is not None: cmd += ["--3d-strength", str(req.strength_3d)]
    if req.style_3d: cmd += ["--style-3d", req.style_3d]
    if req.depth_profile: cmd += ["--depth-profile", req.depth_profile]
    cmd += ["--depth-model", get_depth_model_spec(req.depth_model).key]
    if req.depth_stride and req.depth_stride > 1: cmd += ["--depth-stride", str(req.depth_stride)]
    if req.profile: cmd += ["--profile", req.profile]
    if req.save_depth: cmd += ["--save-depth"]
    if req.custom_depth: cmd += ["--custom-depth", str(Path(req.custom_depth).resolve())]
    if not req.auto_crop: cmd += ["--no-crop"]
    if req.start_time: cmd += ["-s", str(req.start_time)]
    if req.duration: cmd += ["-t", str(req.duration)]
    if req.check_cadence and is_video and req.format in ("sbs", "hsbs"):
        cmd += ["--check-cadence"]
    return cmd


def run_conversion_worker(req):
    import re
    tail = deque(maxlen=12)
    proc = None
    try:
        cmd = build_conversion_command(req)
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                start_new_session=(os.name != "nt"))
        conversion_state["process"] = proc
        if conversion_state.get("cancel_requested"):
            terminate_conversion(proc)
        for line in proc.stdout:
            line = line.strip()
            if not line: continue
            tail.append(line)
            if line.startswith("PORTAL_PROGRESS "):
                conversion_state.update(json.loads(line.split(" ", 1)[1]))
            elif line.startswith("PORTAL_CADENCE "):
                conversion_state["cadence"] = json.loads(line.split(" ", 1)[1])
            else:
                match = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)", line)
                if match:
                    conversion_state.update(percent=float(match[1]), current_frame=int(match[2]),
                                            total_frames=int(match[3]))
                speed = re.search(r"([0-9.]+)(frame/s|s/frame)", line)
                if speed:
                    value = float(speed[1])
                    conversion_state["fps"] = value if speed[2] == "frame/s" else 1 / max(value, 0.001)
                eta = re.search(r"<([^,\]]+)", line)
                if eta: conversion_state["eta"] = eta[1]
                conversion_state["log"] = line[-2000:]
        proc.wait()
        if conversion_state.get("cancel_requested"):
            conversion_state.update(status="idle", log="Cancelled.")
        elif proc.returncode or not Path(conversion_state["output_file"]).is_file():
            conversion_state.update(status="error", log="\n".join(tail)[-3000:] or "Output was not created.")
        else:
            report = conversion_state.get("cadence") or {}
            total = conversion_state.get("total_frames", 0)
            conversion_state.update(
                status="completed",
                percent=100.0,
                current_frame=total if total > 0 else conversion_state.get("current_frame", 0),
                eta="00:00",
                log=("Completed; cadence warning: repeated frames detected. See the cadence report."
                     if report.get("stutter_detected") else "✓ Processing completed.")
            )
            # Record in history
            try:
                out_p = Path(conversion_state["output_file"])
                depth_p = Path(conversion_state.get("depth_file", "")) if conversion_state.get("depth_file") else None
                history_entry = {
                    "id": f"conv_{int(time.time())}_{out_p.stem[:8]}",
                    "timestamp": int(time.time()),
                    "date_str": time.strftime("%Y-%m-%d %H:%M"),
                    "input_path": conversion_state["input_file"],
                    "output_path": str(out_p),
                    "depth_path": str(depth_p) if depth_p and depth_p.exists() else "",
                    "has_depth": bool(depth_p and depth_p.exists()),
                    "operation": req.operation,
                    "format": req.format,
                    "profile": req.profile,
                    "render_mode": req.render_mode,
                    "strength_3d": req.strength_3d if req.strength_3d is not None else 5.0,
                    "style_3d": req.style_3d or "natural",
                    "depth_profile": req.depth_profile or "balanced",
                    "depth_model": req.depth_model or DEFAULT_DEPTH_MODEL_KEY,
                    "depth_stride": req.depth_stride or 1,
                    "convergence": req.convergence,
                    "temporal_smooth": req.temporal_smooth,
                    "auto_crop": req.auto_crop,
                    "save_depth": req.save_depth,
                    "check_cadence": req.check_cadence,
                    "start_time": req.start_time,
                    "clip_duration": req.duration,
                    "total_frames": conversion_state.get("total_frames", 0),
                    "resolution": f"{selected_media.get('width', 0)}x{selected_media.get('height', 0)}" if selected_media else "",
                    "duration": round(float(req.duration or selected_media.get("duration", 0.0) or 0.0), 1)
                }
                add_history_entry(history_entry)
            except Exception as e:
                print(f"[History Warning] Could not record history: {e}")
    except Exception as exc:
        conversion_state.update(status="error", log=str(exc))
    finally:
        if proc and proc.stdout: proc.stdout.close()
        conversion_state["process"] = None


def pause_conversion(proc) -> bool:
    if proc is None or proc.poll() is not None:
        return False
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGSTOP)
            return True
        except Exception as e:
            print(f"[GUI Server] Error pausing process: {e}")
            return False
    else:
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, proc.pid)
            if handle:
                ctypes.windll.ntdll.NtSuspendProcess(handle)
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
        except Exception as e:
            print(f"[GUI Server] Error pausing process on Windows: {e}")
        return False


def resume_conversion(proc) -> bool:
    if proc is None or proc.poll() is not None:
        return False
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGCONT)
            return True
        except Exception as e:
            print(f"[GUI Server] Error resuming process: {e}")
            return False
    else:
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, proc.pid)
            if handle:
                ctypes.windll.ntdll.NtResumeProcess(handle)
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
        except Exception as e:
            print(f"[GUI Server] Error resuming process on Windows: {e}")
        return False


def terminate_conversion(proc):
    if os.name != "nt":
        # The conversion process starts a new session, so its PID is also the
        # process-group ID inherited by FFmpeg encoders/readers. The Python
        # parent can exit before FFmpeg reacts to SIGTERM; always target the
        # group even when proc.poll() already reports that the parent exited.
        process_group = proc.pid
        try:
            os.killpg(process_group, signal.SIGCONT)
        except Exception:
            pass
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return
        # Give encoders a moment to flush, then kill the group unconditionally.
        # Checking the group with signal 0 can raise EPERM on macOS after the
        # session leader exits even while an orphaned FFmpeg is still alive.
        time.sleep(0.75)
        try:
            os.killpg(process_group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    else:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)


HISTORY_FILE = DEFAULT_OUTPUT_DIR / "history.json"


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            out_p = Path(item.get("output_path", ""))
            depth_p = Path(item.get("depth_path", "")) if item.get("depth_path") else None
            item["output_exists"] = out_p.exists() and out_p.stat().st_size > 0
            item["depth_exists"] = bool(depth_p and depth_p.exists() and depth_p.stat().st_size > 0)
        return items
    except Exception as e:
        print(f"[History] Error reading history: {e}")
        return []


def save_history(items: list[dict]):
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        tmp_file = HISTORY_FILE.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        tmp_file.replace(HISTORY_FILE)
    except Exception as e:
        print(f"[History] Error saving history: {e}")


def add_history_entry(entry: dict):
    items = load_history()
    items = [it for it in items if it.get("output_path") != entry.get("output_path")]
    items.insert(0, entry)
    save_history(items[:50])


def delete_history_entry(entry_id: str) -> bool:
    items = load_history()
    orig_len = len(items)
    items = [it for it in items if it.get("id") != entry_id]
    if len(items) < orig_len:
        save_history(items)
        return True
    return False


def clear_history():
    if HISTORY_FILE.exists():
        try:
            HISTORY_FILE.unlink()
        except Exception:
            pass


@app.post("/api/start-conversion")
async def start_conversion(req: ConversionRequest):
    with state_lock:
        if conversion_state["status"] == "running":
            return JSONResponse(status_code=409, content={"error": "A conversion is already running"})
        source, output, is_video = resolve_output(req)
        error = None
        if not source.is_file(): error = "Input file does not exist."
        elif source == output: error = "Choose a new output path; existing files are preserved."
        elif req.operation == "repair" and not is_video: error = "Motion repair requires a video."
        elif req.operation == "repair" and output.suffix.lower() != ".mp4": error = "Repair output must be MP4."
        elif req.operation == "stereo_repair" and not is_video: error = "Stereo repair requires an SBS video."
        elif req.operation == "stereo_repair" and req.repair_layout not in ("hsbs", "sbs"): error = "Stereo repair requires HSBS or Full-SBS layout."
        elif req.operation == "stereo_repair" and not req.repair_source_path: error = "Select the original synchronized 2D master."
        elif req.operation == "stereo_repair" and not Path(req.repair_source_path).expanduser().is_file(): error = "The original 2D master does not exist."
        elif req.custom_depth and not Path(req.custom_depth).is_file(): error = "Depth file does not exist."
        elif is_video:
            try:
                info = get_media_info(source)
                if req.start_time >= info["duration"]: error = "Start time is outside the video."
            except Exception as exc: error = str(exc)
        if error: return JSONResponse(status_code=400, content={"error": error})
        conversion_state.update(status="running", percent=0.0, current_frame=0, total_frames=0,
            fps=0.0, eta="", log="Starting…", output_file=str(output), depth_file="",
            cadence=None, process=None, cancel_requested=False, input_file=str(source),
            start_time=req.start_time, source_fps=(info["fps"] if is_video else 0.0),
            preview_frame=-1, preview_jpeg=None)
        if req.operation == "convert" and req.save_depth:
            conversion_state["depth_file"] = str(output.with_name(output.stem + ("_depth.mp4" if is_video else "_depth.png")))
        threading.Thread(target=run_conversion_worker, args=(req,), daemon=True).start()
        return {"status": "started", "output_file": str(output)}


@app.post("/api/stop-conversion")
async def stop_conversion():
    conversion_state["cancel_requested"] = True
    conversion_state.update(status="stopping", log="Stopping conversion…", eta="")
    proc = conversion_state.get("process")
    if proc: terminate_conversion(proc)
    for key in ("output_file", "depth_file"):
        partial = conversion_state.get(key)
        if partial:
            try:
                Path(partial).unlink(missing_ok=True)
            except OSError as exc:
                print(f"[GUI Server] Could not remove cancelled {key}: {exc}")
    conversion_state.update(status="idle", log="Cancelled.", fps=0.0, eta="")
    return {"status": "cancelled"}


@app.post("/api/pause-conversion")
async def pause_active_conversion():
    with state_lock:
        if conversion_state["status"] != "running":
            return JSONResponse(status_code=400, content={"error": "Conversion is not running"})
        proc = conversion_state.get("process")
        if not proc or proc.poll() is not None:
            return JSONResponse(status_code=400, content={"error": "No active process"})
        if pause_conversion(proc):
            conversion_state["status"] = "paused"
            conversion_state["log"] = "Conversion paused by user."
            return {"status": "paused"}
        return JSONResponse(status_code=500, content={"error": "Failed to pause process"})


@app.post("/api/resume-conversion")
async def resume_active_conversion():
    with state_lock:
        if conversion_state["status"] != "paused":
            return JSONResponse(status_code=400, content={"error": "Conversion is not paused"})
        proc = conversion_state.get("process")
        if not proc or proc.poll() is not None:
            return JSONResponse(status_code=400, content={"error": "No active process"})
        if resume_conversion(proc):
            conversion_state["status"] = "running"
            conversion_state["log"] = "Conversion resumed."
            return {"status": "running"}
        return JSONResponse(status_code=500, content={"error": "Failed to resume process"})


@app.get("/api/status")
async def get_status():
    return {k: v for k, v in conversion_state.items()
            if k not in ("process", "preview_jpeg")}


@app.get("/api/frame-preview")
async def frame_preview(frame: Optional[int] = None):
    """Return the source frame matching the current conversion frame."""
    running = conversion_state.get("status") == "running"
    source = Path(conversion_state.get("input_file", "") if running
                  else selected_media.get("path", ""))
    fps = float(conversion_state.get("source_fps", 0.0) if running
                else selected_media.get("fps", 0.0))
    current_frame = int(conversion_state.get("current_frame", 0)) if running else 0
    total_frames = int(conversion_state.get("total_frames", 0))
    frame = current_frame if frame is None else max(0, int(frame))
    if total_frames > 0:
        frame = min(frame, total_frames - 1)
    start = float(conversion_state.get("start_time", 0.0)) if running else 0.0
    if not source.is_file() or fps <= 0:
        return Response(status_code=404)

    with preview_lock:
        cached_frame = conversion_state.get("preview_frame", -1)
        cached_jpeg = conversion_state.get("preview_jpeg")
        if cached_jpeg is not None and cached_frame == frame:
            return Response(cached_jpeg, media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})
        timestamp = start + (frame / fps)
        result = subprocess.run([
            str(BIN_DIR / "ffmpeg"), "-hide_banner", "-loglevel", "error",
            "-ss", f"{timestamp:.6f}", "-i", str(source), "-frames:v", "1",
            "-vf", "scale=640:-2:flags=fast_bilinear", "-q:v", "4",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-"
        ], capture_output=True, timeout=15)
        if result.returncode or not result.stdout:
            return Response(status_code=404)
        conversion_state["preview_frame"] = frame
        conversion_state["preview_jpeg"] = result.stdout
        return Response(result.stdout, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


@app.get("/api/progress-stream")
async def progress_stream():
    """
    Server-Sent Events (SSE) endpoint to stream live status and progress to the UI.
    """
    async def event_generator():
        while True:
            data = {
                "status": conversion_state["status"],
                "percent": conversion_state["percent"],
                "current_frame": conversion_state["current_frame"],
                "total_frames": conversion_state["total_frames"],
                "fps": conversion_state["fps"],
                "eta": conversion_state["eta"],
                "log": conversion_state["log"],
                "output_file": conversion_state["output_file"],
                "depth_file": conversion_state["depth_file"],
                "cadence": conversion_state.get("cadence"),
                "device": DEVICE.type.upper()
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def open_in_system(path: Path):
    os_name = detect_os()
    try:
        if os_name == "macos":
            subprocess.run(["open", str(path)])
        elif os_name == "windows":
            if hasattr(os, "startfile"):
                os.startfile(str(path))
            else:
                subprocess.run(["cmd", "/c", "start", "", str(path)])
        elif os_name == "linux":
            subprocess.run(["xdg-open", str(path)])
    except Exception as e:
        print(f"[GUI] Error opening path {path}: {e}")


def reveal_in_system(path: Path):
    """Reveal a file in the platform file manager without launching the media."""
    os_name = detect_os()
    try:
        if os_name == "macos":
            subprocess.run(["open", "-R", str(path)])
        elif os_name == "windows":
            subprocess.run(["explorer", "/select,", str(path)])
        elif os_name == "linux":
            subprocess.run(["xdg-open", str(path.parent)])
    except Exception as e:
        print(f"[GUI] Error revealing path {path}: {e}")

@app.post("/api/open-folder")
async def open_folder():
    """Opens output directory in OS file manager."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    open_in_system(DEFAULT_OUTPUT_DIR)
    return {"status": "ok"}

@app.post("/api/open-output")
async def open_output():
    """Opens generated output file in default media player."""
    out = conversion_state.get("output_file")
    if out and Path(out).exists():
        open_in_system(Path(out))
        return {"status": "ok"}
    return {"error": "File not found"}


@app.get("/api/history")
async def get_conversion_history():
    """Returns list of past conversions with existence status and depth map flags."""
    return {"history": load_history()}


@app.get("/api/history/{entry_id}/thumbnail")
async def get_history_thumbnail(entry_id: str):
    """Generate an in-memory JPEG thumbnail for a recorded conversion."""
    item = next((it for it in load_history() if it.get("id") == entry_id), None)
    if not item:
        return Response(status_code=404)
    media = Path(item.get("output_path", ""))
    if not media.is_file():
        return Response(status_code=404)

    stat = media.stat()
    cache_key = str(media)
    with history_thumbnail_lock:
        cached = history_thumbnail_cache.get(cache_key)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return Response(cached[2], media_type="image/jpeg",
                            headers={"Cache-Control": "private, max-age=300"})

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    if media.suffix.lower() in image_exts:
        import cv2
        image = cv2.imread(str(media))
        if image is None:
            return Response(status_code=404)
        height, width = image.shape[:2]
        scale = min(320 / width, 180 / height, 1.0)
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        jpeg = encoded.tobytes() if ok else b""
    else:
        duration = float(item.get("duration", 0.0) or 0.0)
        timestamp = min(30.0, max(0.5, duration * 0.10)) if duration else 1.0
        result = subprocess.run([
            str(BIN_DIR / "ffmpeg"), "-hide_banner", "-loglevel", "error",
            "-ss", f"{timestamp:.3f}", "-i", str(media), "-frames:v", "1",
            "-vf", "thumbnail=24,scale=320:180:force_original_aspect_ratio=decrease",
            "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "-"
        ], capture_output=True, timeout=20)
        jpeg = result.stdout if result.returncode == 0 else b""
    if not jpeg:
        return Response(status_code=404)

    with history_thumbnail_lock:
        history_thumbnail_cache[cache_key] = (stat.st_mtime_ns, stat.st_size, jpeg)
        if len(history_thumbnail_cache) > 50:
            history_thumbnail_cache.pop(next(iter(history_thumbnail_cache)))
    return Response(jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=300"})


@app.delete("/api/history/{entry_id}")
async def remove_history_entry(entry_id: str):
    """Removes a conversion entry from history."""
    if delete_history_entry(entry_id):
        return {"status": "deleted"}
    return JSONResponse(status_code=404, content={"error": "Entry not found"})


@app.post("/api/history/clear")
async def clear_all_history():
    """Clears all conversion history."""
    clear_history()
    return {"status": "cleared"}


@app.post("/api/open-path")
async def open_specific_path(request: Request):
    """Opens a specified media file or folder on the local system."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    path_str = data.get("path")
    if path_str:
        p = Path(path_str)
        if p.exists():
            if data.get("reveal"):
                reveal_in_system(p)
            else:
                open_in_system(p)
            return {"status": "ok"}
    return JSONResponse(status_code=404, content={"error": "File not found"})
