"""
Cross-Platform System, Hardware & Dependency Diagnostic Engine.
Checks OS, GPU drivers (CUDA, ROCm, MPS), hardware video encoders (NVENC, VideoToolbox, AMF, QSV),
and external binaries (FFmpeg, Spatial). Generates actionable install links and commands if anything is missing.
"""
import sys
import os
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch

def detect_os() -> str:
    """Returns 'windows', 'macos', or 'linux'."""
    if sys.platform.startswith("win"):
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    return "linux"

def query_system_graphics_cards() -> List[str]:
    """
    Queries OS display adapter APIs/utilities to discover physical GPUs even if drivers or CUDA are not yet installed.
    """
    cards: List[str] = []
    os_name = detect_os()

    if os_name == "windows":
        # 1. PowerShell CIM query (Modern Windows 10/11)
        try:
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", 
                   "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                for line in out.stdout.strip().splitlines():
                    name = line.strip()
                    if name and name not in cards:
                        cards.append(name)
        except Exception:
            pass

        # 2. WMIC fallback for older Windows versions
        if not cards:
            try:
                out = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"], capture_output=True, text=True, timeout=4)
                if out.returncode == 0:
                    for line in out.stdout.splitlines()[1:]:
                        name = line.strip()
                        if name and name not in cards:
                            cards.append(name)
            except Exception:
                pass

    elif os_name == "linux":
        # 1. lspci query
        try:
            out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=4)
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    if any(k in line for k in ["VGA compatible controller", "3D controller", "Display controller"]):
                        parts = line.split(":")
                        card_name = parts[-1].strip() if len(parts) >= 3 else line.strip()
                        if card_name and card_name not in cards:
                            cards.append(card_name)
        except Exception:
            pass

        # 2. sysfs drm fallback
        if not cards and Path("/sys/class/drm").exists():
            try:
                for card_p in Path("/sys/class/drm").glob("card[0-9]"):
                    device_file = card_p / "device" / "uevent"
                    if device_file.exists():
                        content = device_file.read_text()
                        for line in content.splitlines():
                            if line.startswith("PCI_ID="):
                                cards.append(f"PCI Display Adapter ({line})")
            except Exception:
                pass

    elif os_name == "macos":
        try:
            out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    if "Chipset Model:" in line:
                        m = line.split("Chipset Model:")[-1].strip()
                        if m and m not in cards:
                            cards.append(m)
        except Exception:
            pass

    return cards

def detect_hardware_gpu(lang: str = "en") -> Dict[str, Any]:
    """
    Detects GPU vendor, driver availability, and PyTorch acceleration support (CUDA / MPS / ROCm / CPU).
    Supports English ('en') default and Spanish ('es') localization.
    """
    os_name = detect_os()
    is_es = lang == "es"

    res = {
        "device_type": "cpu",
        "gpu_name": "CPU (Multithread Mode)" if not is_es else "CPU (Modo Multihilo)",
        "has_cuda": False,
        "has_mps": False,
        "has_rocm": False,
        "driver_missing_warning": None,
        "remedy": None,
        "physical_gpus": []
    }

    # 1. Check Apple Silicon MPS
    if os_name == "macos" and torch.backends.mps.is_available():
        res["device_type"] = "mps"
        res["has_mps"] = True
        chip = platform.processor() or "M-Series"
        res["gpu_name"] = f"Apple Silicon ({chip}) [Metal MPS]"
        return res

    # 2. Check PyTorch CUDA (Active and Ready)
    if torch.cuda.is_available():
        res["device_type"] = "cuda"
        res["has_cuda"] = True
        device_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda or "N/A"
        res["gpu_name"] = f"{device_name} (CUDA {cuda_ver})"
        return res

    # 3. Check PyTorch ROCm (AMD on Linux)
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        res["device_type"] = "cuda"
        res["has_rocm"] = True
        res["gpu_name"] = f"AMD Radeon (ROCm {torch.version.hip})"
        return res

    # 4. If PyTorch reports no hardware acceleration, probe physical hardware
    physical_cards = query_system_graphics_cards()
    res["physical_gpus"] = physical_cards

    # Check nvidia-smi explicitly
    nvidia_smi = shutil.which("nvidia-smi")
    has_nvidia = any("nvidia" in c.lower() or "geforce" in c.lower() or "quadro" in c.lower() or "rtx" in c.lower() for c in physical_cards)
    nvidia_name = "NVIDIA GPU"

    if nvidia_smi:
        has_nvidia = True
        try:
            out = subprocess.run([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=4)
            if out.returncode == 0 and out.stdout.strip():
                nvidia_name = out.stdout.strip().split("\n")[0]
        except Exception:
            pass
    elif has_nvidia:
        for c in physical_cards:
            if "nvidia" in c.lower() or "geforce" in c.lower() or "rtx" in c.lower():
                nvidia_name = c
                break

    # If physical NVIDIA exists but PyTorch has no CUDA
    if has_nvidia:
        if is_es:
            warning_msg = (
                f"Se detectó hardware gráfico {nvidia_name}, pero PyTorch está funcionando en CPU "
                f"porque no tiene soporte CUDA activado o faltan controladores oficiales."
            )
            remedy_title = "Instalar Controladores NVIDIA & PyTorch con soporte CUDA"
            remedy_instructions = (
                "1. Si no lo has hecho, descarga e instala los controladores oficiales de NVIDIA desde su web.\n"
                "2. Ejecuta el comando pip en tu terminal o entorno virtual para activar aceleración CUDA 12.1."
            )
            driver_name = "Controladores Oficiales NVIDIA GeForce / RTX / Studio"
            gpu_display = f"{nvidia_name} (Sin aceleración CUDA en PyTorch)"
        else:
            warning_msg = (
                f"Graphics hardware {nvidia_name} detected, but PyTorch is running in CPU mode "
                f"because CUDA acceleration is not enabled or official drivers are missing."
            )
            remedy_title = "Install NVIDIA Drivers & PyTorch with CUDA support"
            remedy_instructions = (
                "1. If not installed, download and install official NVIDIA drivers from their website.\n"
                "2. Run the pip command in your terminal or virtual environment to enable CUDA 12.1 acceleration."
            )
            driver_name = "Official NVIDIA GeForce / RTX / Studio Drivers"
            gpu_display = f"{nvidia_name} (No CUDA acceleration in PyTorch)"

        res["driver_missing_warning"] = warning_msg
        res["remedy"] = {
            "title": remedy_title,
            "command": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
            "driver_url": "https://www.nvidia.com/Download/index.aspx",
            "driver_name": driver_name,
            "instructions": remedy_instructions
        }
        res["gpu_name"] = gpu_display
        return res

    # Check for physical AMD GPU
    has_amd = any("amd" in c.lower() or "radeon" in c.lower() for c in physical_cards)
    if has_amd:
        amd_name = next((c for c in physical_cards if "amd" in c.lower() or "radeon" in c.lower()), "AMD Radeon")
        if os_name == "linux":
            if is_es:
                res["driver_missing_warning"] = f"Se detectó hardware {amd_name}, pero PyTorch no tiene soporte ROCm activado."
                title = "Instalar PyTorch con aceleración AMD ROCm"
                instr = "Instala el stack de controladores AMD ROCm y reinstala PyTorch para ROCm."
            else:
                res["driver_missing_warning"] = f"Hardware {amd_name} detected, but PyTorch does not have ROCm enabled."
                title = "Install PyTorch with AMD ROCm acceleration"
                instr = "Install the AMD ROCm driver stack and reinstall PyTorch for ROCm."

            res["remedy"] = {
                "title": title,
                "command": "pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0",
                "driver_url": "https://www.amd.com/en/support",
                "driver_name": "AMD Software & ROCm Linux Drivers",
                "instructions": instr
            }
        else:
            if is_es:
                res["driver_missing_warning"] = f"Se detectó hardware {amd_name}. En Windows, la IA correrá en CPU multihilo y la codificación de video se acelerará vía AMD AMF."
                title = "Controladores AMD Radeon Adrenalin"
                instr = "Mantén actualizados tus controladores gráficos para máxima compatibilidad con el codificador AMF."
            else:
                res["driver_missing_warning"] = f"Hardware {amd_name} detected. On Windows, AI will run on multithreaded CPU, and video encoding will be accelerated via AMD AMF."
                title = "AMD Radeon Adrenalin Drivers"
                instr = "Keep your graphics drivers updated for maximum compatibility with the AMF encoder."

            res["remedy"] = {
                "title": title,
                "command": "",
                "driver_url": "https://www.amd.com/en/support",
                "driver_name": "AMD Software: Adrenalin Edition",
                "instructions": instr
            }
        res["gpu_name"] = f"{amd_name} (CPU / AMF Encoder)" if not is_es else f"{amd_name} (Modo CPU / Encoder AMF)"
        return res

    # Check for Intel Arc / Iris Xe
    has_intel = any("arc" in c.lower() or "iris" in c.lower() or "intel" in c.lower() for c in physical_cards)
    if has_intel:
        intel_name = next((c for c in physical_cards if "intel" in c.lower()), "Intel Graphics")
        res["gpu_name"] = f"{intel_name} (CPU / QSV Encoder)" if not is_es else f"{intel_name} (Modo CPU / Encoder QSV)"
        return res

    return res

def detect_available_encoders(ffmpeg_bin: str) -> List[str]:
    """
    Queries FFmpeg for compiled and active hardware encoders.
    """
    if not Path(ffmpeg_bin).exists() and not shutil.which(ffmpeg_bin):
        return []

    try:
        cmd = [str(ffmpeg_bin), "-encoders"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        text = out.stdout
        encoders = []
        candidates = [
            "hevc_nvenc", "h264_nvenc",               # NVIDIA
            "hevc_videotoolbox", "h264_videotoolbox", # Apple
            "hevc_amf", "h264_amf",                   # AMD
            "hevc_qsv", "h264_qsv",                   # Intel QuickSync
            "libx265", "libx264"                      # Software fallbacks
        ]
        for c in candidates:
            if c in text:
                encoders.append(c)
        return encoders
    except Exception:
        return ["libx264"]

def get_optimal_encoder_for_system(available_encoders: List[str], lang: str = "en") -> Tuple[str, str]:
    """
    Picks the best available video encoder based on hardware.
    Priority: NVENC > VideoToolbox > AMF > QSV > libx265 > libx264.
    """
    is_es = lang == "es"
    hw_suffix = " (Hardware Acceleration)" if not is_es else " (Aceleración por Hardware)"
    cpu_suffix = " (CPU Multithread)" if not is_es else " (CPU Multihilo)"

    if "hevc_nvenc" in available_encoders:
        return "hevc_nvenc", f"NVIDIA NVENC{hw_suffix}"
    elif "hevc_videotoolbox" in available_encoders:
        return "hevc_videotoolbox", f"Apple VideoToolbox{hw_suffix}"
    elif "hevc_amf" in available_encoders:
        return "hevc_amf", f"AMD AMF{hw_suffix}"
    elif "hevc_qsv" in available_encoders:
        return "hevc_qsv", f"Intel QuickSync{hw_suffix}"
    elif "libx265" in available_encoders:
        return "libx265", f"x265 HEVC{cpu_suffix}"
    return "libx264", f"x264 AVC{cpu_suffix}"

def run_full_system_diagnostic(bin_dir: Path, lang: str = "en") -> Dict[str, Any]:
    """
    Runs comprehensive environment diagnostics and returns status, hardware report, and actionable missing download links.
    Default language: English ('en'). Secondary: Spanish ('es').
    """
    os_name = detect_os()
    gpu_info = detect_hardware_gpu(lang=lang)
    is_es = lang == "es"
    
    # Binary names based on OS
    exe_suffix = ".exe" if os_name == "windows" else ""
    ffmpeg_name = f"ffmpeg{exe_suffix}"
    ffprobe_name = f"ffprobe{exe_suffix}"
    spatial_name = f"spatial{exe_suffix}"

    # Resolve binaries
    def find_bin(name: str) -> Optional[Path]:
        p = bin_dir / name
        if p.exists() and os.access(p, os.X_OK):
            return p
        sys_p = shutil.which(name)
        if sys_p:
            return Path(sys_p)
        return None

    ffmpeg_path = find_bin(ffmpeg_name)
    ffprobe_path = find_bin(ffprobe_name)
    spatial_path = find_bin(spatial_name)

    encoders = detect_available_encoders(str(ffmpeg_path)) if ffmpeg_path else []
    best_encoder, encoder_desc = get_optimal_encoder_for_system(encoders, lang=lang)

    missing_items: List[Dict[str, str]] = []

    # 1. FFmpeg & FFprobe missing (CRITICAL)
    if not ffmpeg_path or not ffprobe_path:
        if os_name == "windows":
            missing_items.append({
                "item": "FFmpeg & FFprobe" if not is_es else "FFmpeg & FFprobe (Binarios Multimedia)",
                "category": "Video Engine (Critical)" if not is_es else "Motor de Video (Crítico)",
                "urgency": "CRITICAL",
                "reason": "Required to decode source videos, mux audio streams, and encode 3D outputs." if not is_es else "Indispensable para decodificar, leer pistas de audio y codificar los videos 3D generados.",
                "command": "winget install Gyan.FFmpeg",
                "url": "https://www.gyan.dev/ffmpeg/builds/",
                "instructions": (
                    "Option A (Recommended): Open PowerShell or CMD and run 'winget install Gyan.FFmpeg'.\n"
                    "Option B: Download 'ffmpeg-release-essentials.zip' from gyan.dev, extract ffmpeg.exe and ffprobe.exe, and place them inside the 'bin/' folder of this project."
                ) if not is_es else (
                    "Opción A (Recomendada): Abre PowerShell o CMD y corre 'winget install Gyan.FFmpeg'.\n"
                    "Opción B: Descarga 'ffmpeg-release-essentials.zip' de gyan.dev, extrae ffmpeg.exe y ffprobe.exe y colócalos dentro de la carpeta 'bin/' de este proyecto."
                )
            })
        elif os_name == "linux":
            missing_items.append({
                "item": "FFmpeg" if not is_es else "FFmpeg (Binarios Multimedia)",
                "category": "Video Engine (Critical)" if not is_es else "Motor de Video (Crítico)",
                "urgency": "CRITICAL",
                "reason": "Required for video decoding, audio handling, and 3D video encoding." if not is_es else "Indispensable para decodificar, leer pistas de audio y codificar los videos 3D.",
                "command": "sudo apt update && sudo apt install -y ffmpeg",
                "url": "https://ffmpeg.org/download.html",
                "instructions": (
                    "Ubuntu/Debian: sudo apt install -y ffmpeg\n"
                    "Arch Linux: sudo pacman -S ffmpeg\n"
                    "Fedora: sudo dnf install ffmpeg"
                ) if not is_es else (
                    "En Ubuntu/Debian ejecuta: sudo apt install -y ffmpeg\n"
                    "En Arch Linux ejecuta: sudo pacman -S ffmpeg\n"
                    "En Fedora ejecuta: sudo dnf install ffmpeg"
                )
            })
        else: # macOS
            missing_items.append({
                "item": "FFmpeg",
                "category": "Video Engine (Critical)" if not is_es else "Motor de Video (Crítico)",
                "urgency": "CRITICAL",
                "reason": "Required for video decoding and encoding." if not is_es else "Indispensable para decodificar y codificar video.",
                "command": "brew install ffmpeg",
                "url": "https://formulae.brew.sh/formula/ffmpeg",
                "instructions": "Open Terminal and run 'brew install ffmpeg' using Homebrew." if not is_es else "Abre tu Terminal y ejecuta 'brew install ffmpeg' usando Homebrew."
            })

    # 2. GPU Driver / CUDA missing (RECOMMENDED)
    if gpu_info.get("driver_missing_warning") and gpu_info.get("remedy"):
        remedy = gpu_info["remedy"]
        missing_items.append({
            "item": remedy["title"],
            "category": "Hardware GPU Acceleration" if not is_es else "Aceleración de GPU por Hardware",
            "urgency": "RECOMMENDED",
            "reason": gpu_info["driver_missing_warning"],
            "command": remedy.get("command", ""),
            "url": remedy.get("driver_url", ""),
            "instructions": remedy.get("instructions", f"Install official drivers: {remedy.get('driver_name', '')}")
        })
    elif gpu_info["device_type"] == "cpu" and not gpu_info.get("driver_missing_warning"):
        missing_items.append({
            "item": "Dedicated GPU Accelerator (NVIDIA / Apple Silicon)" if not is_es else "Acelerador Gráfico Dedicado (NVIDIA / Apple Silicon)",
            "category": "Performance (Informational)" if not is_es else "Rendimiento (Informativo)",
            "urgency": "OPTIONAL",
            "reason": "The system is operating in multithreaded CPU mode. 3D conversion will work reliably, but a dedicated GPU will speed up AI inference up to 12x." if not is_es else "El sistema está operando en modo CPU multihilo. La conversión 3D funcionará sin errores, pero una GPU dedicada acelerará la IA hasta 12x veces más rápido.",
            "command": "",
            "url": "https://pytorch.org/get-started/locally/",
            "instructions": "If you have an NVIDIA or AMD GPU, install official drivers and the corresponding PyTorch build." if not is_es else "Si tienes una tarjeta gráfica NVIDIA o AMD, asegúrate de tener instalados sus drivers y la versión correspondiente de PyTorch."
        })

    # 3. Spatial Video CLI missing (OPTIONAL - Apple Vision Pro MV-HEVC)
    if not spatial_path:
        missing_items.append({
            "item": "Spatial Video CLI (Mike Swanson)",
            "category": "Apple Spatial Video MV-HEVC (Optional)" if not is_es else "Formato MV-HEVC Espacial (Opcional)",
            "urgency": "OPTIONAL",
            "reason": "Only required if exporting native Spatial Video for Apple Vision Pro or Meta Quest." if not is_es else "Solo es necesario si vas a exportar en formato Spatial Video nativo para Apple Vision Pro o Meta Quest.",
            "command": "",
            "url": "https://blog.mikeswanson.com/spatial",
            "instructions": (
                f"Download the compiled binary for {os_name} from Mike Swanson's official site and place the '{spatial_name}' executable inside the 'bin/' folder."
            ) if not is_es else (
                f"Descarga la versión compilada para {os_name} del sitio oficial de Mike Swanson y copia el ejecutable '{spatial_name}' dentro de la carpeta 'bin/' del proyecto."
            )
        })

    # Status determination
    has_critical = any(x["urgency"] == "CRITICAL" for x in missing_items)
    has_warning = any(x["urgency"] == "RECOMMENDED" for x in missing_items)

    if has_critical:
        status = "error"
    elif has_warning:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "os": os_name,
        "os_details": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "device": gpu_info["device_type"].upper(),
        "gpu_name": gpu_info["gpu_name"],
        "encoder": best_encoder,
        "encoder_desc": encoder_desc,
        "available_encoders": encoders,
        "ffmpeg_installed": ffmpeg_path is not None,
        "spatial_installed": spatial_path is not None,
        "missing_items": missing_items
    }

    # 3. Spatial Video CLI missing (OPTIONAL - Apple Vision Pro MV-HEVC)
    if not spatial_path:
        missing_items.append({
            "item": "Spatial Video CLI (Mike Swanson)",
            "category": "Formato MV-HEVC Espacial (Opcional)",
            "urgency": "OPTIONAL",
            "reason": "Solo es necesario si vas a exportar en formato Spatial Video nativo para Apple Vision Pro o Meta Quest.",
            "command": "",
            "url": "https://blog.mikeswanson.com/spatial",
            "instructions": (
                f"Descarga la versión compilada para {os_name} del sitio oficial de Mike Swanson y copia el ejecutable '{spatial_name}' dentro de la carpeta 'bin/' del proyecto."
            )
        })

    # Status determination
    has_critical = any(x["urgency"] == "CRITICAL" for x in missing_items)
    has_warning = any(x["urgency"] == "RECOMMENDED" for x in missing_items)

    if has_critical:
        status = "error"
    elif has_warning:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "os": os_name,
        "os_details": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "device": gpu_info["device_type"].upper(),
        "gpu_name": gpu_info["gpu_name"],
        "encoder": best_encoder,
        "encoder_desc": encoder_desc,
        "available_encoders": encoders,
        "ffmpeg_installed": ffmpeg_path is not None,
        "spatial_installed": spatial_path is not None,
        "missing_items": missing_items
    }
