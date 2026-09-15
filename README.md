# 2D to 3D Studio - Offline Stereoscopic & Spatial Engine

A standalone, high-performance, 100% offline 2D-to-3D image and video converter. Transforms conventional 2D photos and videos into stereoscopic 3D (Half-SBS, Full-SBS, Dubois Anaglyph) and Apple Spatial Video (MV-HEVC / Spatial Photos) with zero cloud dependency, no subscriptions, and complete privacy.

Features native hardware acceleration across platforms:
- **Apple Silicon (M1/M2/M3/M4/M5):** Accelerated with **Metal Performance Shaders (MPS)** and **VideoToolbox** hardware HEVC encoding.
- **NVIDIA GeForce / RTX / Quadro:** Accelerated with **CUDA** inference and **NVENC** hardware video encoding.
- **AMD Radeon:** **ROCm** support on Linux and **AMF** hardware encoding on Windows.
- **Intel Arc / Iris Xe:** **QuickSync (QSV)** hardware encoding.
- **CPU Multithreading:** High-accuracy fallback for systems without a dedicated GPU.

---

## Key Features

- **100% Offline & Private:** Zero telemetry, no cloud APIs, no tokens, and no data leaves your machine.
- **State-of-the-Art Depth AI:** Pure PyTorch implementation of **DINOv2 ViT-Small + DPT Depth Anything** running directly on your GPU.
- **Anti-Ghosting Stereo Synthesis (`right_only` mode):** Keeps the original left eye 100% untouched and unwarped. Completely eliminates the line deformation, blurriness, and double-vision ghosting typical of naive DIBR converters.
- **Bilateral Edge Snapping:** Sharpens depth boundaries along high-contrast lines and subtitle edges to prevent warped text.
- **Temporal Coherence Filter:** Eliminates inter-frame depth flickering in video while adapting to fast-moving scene cuts.
- **Automated Black Bar Letterbox Cropping:** Detects and crops letterboxing to prevent black margins from creating artificial depth planes.
- **Fast Depth Re-Export:** Save depth maps (`_depth.mp4` / `_depth.png`) for manual grading in Photoshop, DaVinci Resolve, or After Effects, then re-render stereo in seconds without re-running AI inference.
- **Multi-Format 3D Output:**
  - **Half-SBS (Side-by-Side):** Ready for 3D TVs (LG Cinema 3D, Samsung 3D, Sony) and Jellyfin / Plex / Roku streaming.
  - **Full-SBS:** Full resolution for VR headsets (Meta Quest 2/3/Pro, Pico).
  - **Dubois Anaglyph (Red/Cyan):** Color-accurate Dubois algorithm viewable on any monitor with standard 3D glasses.
  - **Apple Spatial Video (MV-HEVC):** Multi-View HEVC for Apple Vision Pro and Meta Quest 3 via native `spatial` packaging.
  - **Apple Spatial Photo (HEIC):** Native spatial photo format for visionOS and iOS.
- **Multi-Language Support (i18n):** Default language is **English (`en`)**, secondary is **Spanish (`es`)**. Community translation files can be dropped into `locales/` and are automatically discovered by both the Web GUI and CLI.

---

## Hardware Autodetection & Diagnostics

The converter includes a cross-platform diagnostic engine that inspects your GPU, PyTorch acceleration stack, and multimedia binaries on startup.

### Run System Diagnostics via CLI:

```bash
# Default English diagnostics
python convert_3d.py --check-system

# Spanish diagnostics
python convert_3d.py --check-system --lang es
```

**Example Output:**
```text
========================================================================
        2D TO 3D STUDIO - SYSTEM DIAGNOSTICS REPORT
========================================================================
Operating System:   Darwin 23.6.0 (arm64)
AI Accelerator (GPU): Apple Silicon (arm) [Metal MPS] [MPS]
Video Encoder:   Apple VideoToolbox (Hardware Acceleration) (hevc_videotoolbox)
FFmpeg Installed:    ✓ Yes
Spatial CLI:         ✓ Yes
Overall Status:      OK
------------------------------------------------------------------------
✓ Excellent! All components and drivers are configured and optimized.
```

If drivers, CUDA packages, or FFmpeg are missing, the system will output actionable alerts with:
1. Urgency status (`CRITICAL`, `RECOMMENDED`, or `OPTIONAL`).
2. Exact copy-paste terminal command (e.g. `winget install Gyan.FFmpeg` on Windows, `sudo apt install ffmpeg` on Linux, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` for NVIDIA CUDA).
3. Official direct download URLs.
4. Clear step-by-step instructions.

---

## Web Graphical User Interface (GUI)

The project includes a web studio interface with real-time SSE progress streaming, FPS telemetry, ETA calculation, optical sliders, preset selectors, and an interactive Hardware Diagnostics Modal.

### Launching the GUI (Windows, Linux, macOS):

#### 🪟 Windows (10 / 11)
- **Method 1 (One-Click Launcher):**
  - Simply double-click [`start_windows.bat`](file:///Users/jdmarinv/Dev/3D/2D-to-3D/start_windows.bat) in the project folder.
- **Method 2 (Command Prompt / PowerShell):**
  ```cmd
  cd path\to\2D-to-3D
  .venv\Scripts\activate
  python gui.py
  ```

#### 🐧 Linux (Ubuntu, Debian, Fedora, Arch)
- **Method 1 (One-Click / Terminal Launcher):**
  - Make sure the script is executable and run it:
    ```bash
    chmod +x start_linux.sh
    ./start_linux.sh
    ```
- **Method 2 (Terminal Step-by-Step):**
  ```bash
  cd ~/2D-to-3D
  source .venv/bin/activate
  python3 gui.py
  ```

#### 🍎 macOS (Apple Silicon M1–M5 & Intel)
- **Method 1 (One-Click Launcher):**
  - Double-click [`start_mac.command`](file:///Users/jdmarinv/Dev/3D/2D-to-3D/start_mac.command) in the project root or the Desktop shortcut.
- **Method 2 (Terminal):**
  ```bash
  cd ~/Dev/3D/2D-to-3D
  source .venv/bin/activate
  python3 gui.py
  ```

*The GUI will automatically open in your default web browser at **`http://127.0.0.1:8765`**.*

---

## First-Time Installation & Setup (Cross-Platform)

### 🪟 Windows Setup Guide
1. **Install Python:**
   - Download and install Python 3.10, 3.11, or 3.12 from [python.org](https://www.python.org/downloads/).
   - **Important:** Check the box **"Add python.exe to PATH"** during installation.
2. **Install FFmpeg:**
   - Open PowerShell as Administrator and run:
     ```powershell
     winget install Gyan.FFmpeg
     ```
   - Or download builds manually from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) and add the `bin` folder to your System PATH.
3. **Setup Virtual Environment & Dependencies:**
   ```cmd
   cd path\to\2D-to-3D
   python -m venv .venv
   call .venv\Scripts\activate

   :: For NVIDIA GPU acceleration (CUDA 12.1):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   :: Install remaining dependencies:
   pip install -r requirements.txt
   ```
4. **Launch:** Double-click `start_windows.bat`.

---

### 🐧 Linux Setup Guide (Ubuntu / Debian / Fedora / Arch)
1. **Install System Dependencies & FFmpeg:**
   ```bash
   # Ubuntu / Debian / Linux Mint:
   sudo apt update && sudo apt install -y python3 python3-venv python3-pip ffmpeg zenity

   # Fedora / RHEL:
   sudo dnf install -y python3 python3-pip ffmpeg zenity

   # Arch Linux / Manjaro:
   sudo pacman -S python python-pip ffmpeg zenity
   ```
2. **Setup Virtual Environment & Dependencies:**
   ```bash
   cd ~/2D-to-3D
   python3 -m venv .venv
   source .venv/bin/activate

   # If you have an NVIDIA GPU (CUDA 12.1):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # If you have an AMD GPU (ROCm 6.0):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

   # Install all requirements:
   pip install -r requirements.txt
   ```
3. **Launch:** Run `./start_linux.sh` or `python3 gui.py`.

---

### 🍎 macOS Setup Guide
1. **Requirements:** macOS 13.0+ (Ventura, Sonoma, or Sequoia). Metal acceleration (MPS) is built-in.
2. **Setup Virtual Environment & Dependencies:**
   ```bash
   cd ~/Dev/3D/2D-to-3D
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Launch:** Run `./start_mac.command` or `python3 gui.py`.

---

### GUI Highlights:
- **Language Switcher (🌐):** Seamlessly switch between English, Spanish, or any user-contributed language in the header without reloading.
- **Hardware Badge:** Click the badge in the upper-right corner to open the System Diagnostics & Drivers modal.
- **Real-Time Progress:** View frame-by-frame progress, rendering speed, and remaining time.
- **One-Click Media Access:** Open the output folder or play the converted 3D file directly upon completion.

---

## Command Line Interface (CLI)

### 1. Basic Conversion

```bash
cd ~/Dev/3D/2D-to-3D
source .venv/bin/activate

# Convert an image to Half-SBS (3D TV format)
python convert_3d.py -i photo.jpg -f hsbs

# Convert a video to Dubois Anaglyph (Red/Cyan glasses)
python convert_3d.py -i video.mp4 -f anaglyph

# Convert to Apple Vision Pro Spatial Video (MV-HEVC)
python convert_3d.py -i video.mp4 -f spatial
```

### 2. Using Conversion Profiles

Profiles automatically configure stereo divergence, convergence plane, and pop-out bias:

```bash
# Balanced profile (recommended baseline: comfortable parallax, zero ghosting)
python convert_3d.py -i video.mp4 --profile balanced

# Standard stereoscopic profiles available:
# --profile balanced        : Balanced depth for movies and documentaries (0.025).
# --profile cinematic_depth : Enhanced depth sensation and volume for live-action films (0.030).
# --profile subtle_parallax : Gentle parallax for fast motion, text, and comfort (0.020).
# --profile high_contrast   : Snapped depth edges for anime, animation, and high-contrast lines (0.022).
```

### 3. Save Depth Maps & Fast Re-Export

Export grayscale depth maps for inspection or manual retouching in video editing software:

```bash
# 1. Generate 3D video and save the depth map (_depth.mp4):
python convert_3d.py -i video.mp4 -f hsbs --save-depth

# 2. Re-export in seconds using an edited or existing depth map (skips AI inference):
python convert_3d.py -i video.mp4 --custom-depth video_depth.mp4 -f hsbs
```

### 4. Quick Preview Clipping (`-s` and `-t`)

Test depth settings on a short excerpt before processing an entire movie:

```bash
# Convert a 10-second test clip starting at second 30:
python convert_3d.py -i movie.mp4 -f hsbs -s 30 -t 10
```

---

## CLI Options Reference

| Option | Default | Description |
|---|---|---|
| `-i`, `--input` | *None* | Path to input 2D image or video file. |
| `-o`, `--output` | Auto | Destination path for converted 3D file. |
| `-f`, `--format` | `hsbs` | 3D Format: `hsbs` (Half-SBS), `sbs` (Full-SBS), `anaglyph` (Red/Cyan), `spatial` (Apple MV-HEVC / HEIC). |
| `--profile` | *None* | Predefined profile: `balanced`, `cinematic_depth`, `subtle_parallax`, `high_contrast`. |
| `--render-mode` | `right_only` | Stereo rendering mode: `right_only` (left eye pristine, no warping) or `both` (symmetric). |
| `--depth-intensity` | `0.022` | Stereo separation strength (Parallax divergence: `0.010` subtle to `0.060` strong). |
| `--convergence` | `0.50` | Zero-parallax screen plane (`0.0` = pop out in front of screen, `1.0` = deep into screen). |
| `--temporal-smooth` | `0.70` | Temporal anti-flicker smoothing weight across consecutive video frames (`0.0` to `0.95`). |
| `--save-depth` | Disabled | Save estimated depth map as a separate video (`_depth.mp4`) or image (`_depth.png`). |
| `--custom-depth` | *None* | Path to external depth map file. Skips AI model inference for instant stereo synthesis. |
| `-s`, `--start-time`| `0.0` | Start time in seconds for video conversion. |
| `-t`, `--duration` | `0.0` | Duration in seconds to convert (`0.0` = full video). |
| `--no-crop` | Disabled | Disable automatic letterbox black bar detection. |
| `--batch-size` | `1` | Batch size for parallel depth inference on GPU (strict FIFO order). |
| `--depth-stride` | `1` | Compute depth every N frames while preserving 100% genuine RGB frame motion. |
| `--resume` | Disabled | Resume an interrupted conversion from the last frame-accurate checkpoint. |
| `--check-cadence` | Disabled | Audit converted 3D output against 2D input for visual cadence and stutter detection. |
| `--repair-stutter` | *None* | Path for output file to repair an already-defective video with frozen duplicate frames. |
| `--lang` | `en` | UI and diagnostics language (`en` for English, `es` for Spanish, or any custom locale). |
| `--check-system` | Flag | Run hardware, GPU acceleration, and encoder diagnostics without converting. |

---

## Internationalization (i18n) & Contributing Translations

2D to 3D Studio is built from the ground up to be multi-language. All user-facing strings are decoupled into standalone JSON files in the `locales/` directory:

```text
locales/
├── en.json   # English (Default)
└── es.json   # Spanish (Secondary)
```

### How to Add a New Language:

You can contribute a new language without touching any Python or frontend code:

1. **Copy the English template:**
   ```bash
   cp locales/en.json locales/<your_lang_code>.json
   # Example for French:
   cp locales/en.json locales/fr.json
   ```

2. **Edit `__meta__` in your new file:**
   ```json
   {
     "__meta__": {
       "code": "fr",
       "name": "Français",
       "native_name": "Français"
     },
     "app": {
       "title": "2D to 3D Studio",
       "subtitle": "MOTEUR STÉRÉOSCOPIQUE ET SPATIAL HORS LIGNE"
     }
   }
   ```

3. **Translate the string values.** Missing keys automatically fall back to English (`en`).

4. **Launch the app:**
   - The Web GUI will automatically detect your new translation file and add it to the **🌐 Language** dropdown selector.
   - The CLI will immediately accept your language via `--lang <code_lang>` (e.g. `python convert_3d.py --check-system --lang fr`).

Pull requests adding new language translations are warmly welcome!

---

## Automated Verification Suite

To run the end-to-end test suite on synthetic 3D geometry and verify model inference, stereo synthesis, and encoder pipelines:

```bash
python tests/test_pipeline.py
```

Expected output:
```text
ALL TESTS PASSED! OFFLINE 2D-TO-3D PIPELINE IS 100% OPERATIONAL.
```
