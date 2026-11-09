"""Instala todo lo necesario. Ejecútalo con tu Python de siempre (3.11 o más reciente):

    python instalar.py

1. Crea el entorno virtual .venv e instala las dependencias (con GPU si hay una NVIDIA).
2. Descarga los modelos de detección y borrado (~410 MB) en models/.
3. Descarga las fuentes (licencia OFL) en fonts/.
4. Descarga Kindle Comic Converter (solo Windows) en tools/kcc/.
5. Comprueba que Claude Code (`claude`) esté instalado.

Se puede volver a ejecutar: solo descarga lo que falte.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(AQUI, ".venv")
PY = os.path.join(VENV, "Scripts" if os.name == "nt" else "bin", "python" + (".exe" if os.name == "nt" else ""))

MODELOS = {
    # detección de globos (YOLOv8-seg, GPL-3.0) — https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble
    "models/bubble_seg_model_dynamic.onnx":
        "https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/model_dynamic.onnx",
    # detección de texto (comic-text-detector) — https://github.com/zyddnys/manga-image-translator
    "models/comictextdetector.onnx":
        "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
    # reconstrucción de fondo (LaMa, Apache-2.0) — https://huggingface.co/Carve/LaMa-ONNX
    "models/lama_fp32.onnx":
        "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
}
FUENTES_BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
FUENTES = {
    "fonts/ComicRelief-Bold.ttf": "comicrelief/ComicRelief-Bold.ttf",
    "fonts/Bangers-Regular.ttf": "bangers/Bangers-Regular.ttf",
    "fonts/GochiHand-Regular.ttf": "gochihand/GochiHand-Regular.ttf",
    "fonts/Oswaldwght.ttf": "oswald/Oswald%5Bwght%5D.ttf",
    "fonts/ArchivoNarrowwght.ttf": "archivonarrow/ArchivoNarrow%5Bwght%5D.ttf",
    "fonts/OFL-ComicRelief.txt": "comicrelief/OFL.txt",
    "fonts/OFL-Bangers.txt": "bangers/OFL.txt",
    "fonts/OFL-GochiHand.txt": "gochihand/OFL.txt",
    "fonts/OFL-Oswald.txt": "oswald/OFL.txt",
    "fonts/OFL-ArchivoNarrow.txt": "archivonarrow/OFL.txt",
}
KCC_VERSION = "12.1.5"
KCC_EXE = f"tools/kcc/tool_cli_c2e_{KCC_VERSION}_AMD64.exe"
KCC_URL = f"https://github.com/ciromattia/kcc/releases/download/v{KCC_VERSION}/tool_cli_c2e_{KCC_VERSION}_AMD64.exe"
KCC_SHA256 = "228c9614c93e927465ed628b32edd4ba15f50f8911b9f9702f5f214b95c02abe"


def paso(t):
    print(f"\n== {t}", flush=True)


def correr(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def bajar(url, destino, sha256=None):
    destino = os.path.join(AQUI, destino)
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        print(f"  ya existe {os.path.relpath(destino, AQUI)}")
        return
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    print(f"  bajando {os.path.relpath(destino, AQUI)} ...", flush=True)
    tmp = destino + ".parcial"
    req = urllib.request.Request(url, headers={"User-Agent": "manga-traductor-instalador"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        hecho = 0
        while True:
            bloque = r.read(1 << 20)
            if not bloque:
                break
            f.write(bloque)
            hecho += len(bloque)
            if total > 20 << 20:
                print(f"\r    {hecho >> 20} / {total >> 20} MB", end="", flush=True)
    if total > 20 << 20:
        print()
    if sha256:
        h = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
        if h != sha256:
            os.remove(tmp)
            sys.exit(f"  La suma SHA-256 de {destino} no coincide; descarga cancelada.")
    os.replace(tmp, destino)


def kindlegen():
    """kindlegen (para MOBI por USB) sin instalar Kindle Previewer ni pedir administrador:
    se descarga el instalador oficial con winget (que verifica su SHA-256) y se extrae solo
    la carpeta de kindlegen con 7-Zip. Si falta winget o 7-Zip, se explica la alternativa."""
    destino = os.path.join(AQUI, "tools", "kindlegen")
    if os.path.exists(os.path.join(destino, "kindlegen.exe")):
        print("  ya existe tools\\kindlegen\\kindlegen.exe")
        return
    siete = shutil.which("7z") or next((p for p in (r"C:\Program Files\7-Zip\7z.exe",
                                                    r"C:\Program Files (x86)\7-Zip\7z.exe") if os.path.exists(p)), None)
    if not (shutil.which("winget") and siete):
        print("  Para pasar libros por USB (MOBI) hace falta kindlegen. Opciones:\n"
              "   - instala 7-Zip (https://www.7-zip.org) y vuelve a ejecutar instalar.py, o\n"
              "   - instala Kindle Previewer de Amazon, o\n"
              "   - usa --formato epub y Send to Kindle.")
        return
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        print("  bajando el instalador oficial de Kindle Previewer (~470 MB) para extraer kindlegen...")
        r = subprocess.run(["winget", "download", "--id", "Amazon.KindlePreviewer", "-e", "-d", tmp,
                            "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity"],
                           capture_output=True, text=True)
        exe = next((os.path.join(tmp, f) for f in os.listdir(tmp) if f.lower().endswith(".exe")), None)
        if r.returncode != 0 or not exe:
            print("  no se pudo descargar con winget; usa --formato epub o instala Kindle Previewer")
            return
        subprocess.run([siete, "e", "-tNsis", exe, r"$PLUGINSDIR\KindlePreviewerApp.msix", "-o" + tmp, "-y"],
                       capture_output=True, check=True)
        subprocess.run([siete, "e", os.path.join(tmp, "KindlePreviewerApp.msix"),
                        r"KindlePreviewerApp\resources\KFXGen\bin\*", "-o" + destino, "-y"],
                       capture_output=True, check=True)
    print("  kindlegen listo en tools\\kindlegen\\" if os.path.exists(os.path.join(destino, "kindlegen.exe"))
          else "  no se encontró kindlegen dentro del instalador")


def main():
    if sys.version_info < (3, 11):
        sys.exit("Se necesita Python 3.11 o más reciente.")

    paso("Entorno virtual")
    if not os.path.exists(PY):
        correr([sys.executable, "-m", "venv", VENV])
    correr([PY, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    correr([PY, "-m", "pip", "install", "-q", "-r", os.path.join(AQUI, "requirements.txt")])
    if shutil.which("nvidia-smi"):
        print("  GPU NVIDIA detectada: se instala ONNX Runtime con CUDA")
        correr([PY, "-m", "pip", "install", "-q", "onnxruntime-gpu[cuda,cudnn]"])
    else:
        print("  Sin GPU NVIDIA: se usa la CPU (funciona, pero más lento)")
        correr([PY, "-m", "pip", "install", "-q", "onnxruntime"])

    paso("Modelos")
    for destino, url in MODELOS.items():
        bajar(url, destino)

    paso("Fuentes")
    for destino, ruta in FUENTES.items():
        bajar(FUENTES_BASE + ruta, destino)

    paso("Kindle Comic Converter")
    if os.name == "nt":
        bajar(KCC_URL, KCC_EXE, KCC_SHA256)
        kindlegen()
    else:
        print("  En macOS/Linux instala KCC desde https://github.com/ciromattia/kcc/releases y pasa su"
              " ruta a traducir.py con --kcc (paso «kindle»).")

    paso("Claude Code")
    if shutil.which("claude"):
        print("  claude encontrado. Si nunca iniciaste sesión, abre una terminal, escribe «claude» y usa /login.")
    else:
        print("  NO encuentro «claude». Instala Claude Code (https://claude.com/claude-code) e inicia sesión;"
              " sin él no se puede traducir.")

    print("\nListo. Siguiente paso: lee el README (sección «Traducir un tomo»).")


if __name__ == "__main__":
    main()
