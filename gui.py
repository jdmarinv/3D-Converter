#!/usr/bin/env python3
"""
Launcher script for 2D to 3D Studio GUI.
Starts the local server and automatically opens the interface in your browser.
"""
import sys
import webbrowser
import threading
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def open_browser(port: int = 8765):
    time.sleep(1.2)
    url = f"http://127.0.0.1:{port}"
    print(f"\n[GUI] Abriendo interfaz en el navegador: {url}")
    webbrowser.open(url)

def main():
    import uvicorn
    port = 8765
    print("=" * 65)
    print("        2D TO 3D STUDIO - INICIANDO INTERFAZ GRÁFICA")
    print("=" * 65)
    print(f"Dirección local: http://127.0.0.1:{port}")
    print("Presiona Ctrl + C en esta ventana para cerrar el servidor.")
    print("=" * 65)

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    uvicorn.run("gui.server:app", host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    main()
