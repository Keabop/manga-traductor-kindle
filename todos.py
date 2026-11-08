"""Procesa una carpeta entera de tomos, uno tras otro, sin intervención.

    .venv\\Scripts\\python todos.py "<carpeta con los EPUB>" --glosario glosarios\\obra.md [--perfil KPW6]

Para cada tomo: analizar, traducir, componer, armar el EPUB y convertir para Kindle.
- Si se acaba la cuota de Claude, espera solo a que se renueve y sigue.
- Se puede cortar y relanzar cuando sea: lo ya traducido no se repite y los tomos
  terminados se saltan.
- Deja un registro en salida\\progreso.log.
"""
import argparse
import contextlib
import json
import os
import re
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import traducir  # noqa: E402

REGISTRO = os.path.join(AQUI, "salida", "progreso.log")


class Tee:
    """Copia la salida en pantalla y en el registro."""

    def __init__(self, *destinos):
        self.destinos = destinos

    def write(self, s):
        for d in self.destinos:
            d.write(s)
            d.flush()

    def flush(self):
        for d in self.destinos:
            d.flush()


def numero(nombre):
    m = re.search(r"vol(?:ume)?\.?\s*(\d+)", nombre, re.I) or re.search(r"(\d+)", nombre)
    return int(m.group(1)) if m else 0


def pendientes_de_traducir(epub):
    """Páginas con texto que aún no tienen traducción (None si el tomo no se ha analizado)."""
    w = traducir.dir_libro(epub)
    du = os.path.join(w, "unidades")
    if not os.path.isdir(du):
        return None
    faltan = 0
    for f in os.listdir(du):
        if f.endswith(".json") and json.load(open(os.path.join(du, f), encoding="utf-8"))["unidades"]:
            faltan += not os.path.exists(os.path.join(w, "traduccion", f))
    return faltan


def terminado(epub):
    nombre = os.path.splitext(os.path.basename(epub))[0] + " [ES]"
    kindle = os.path.join(AQUI, "salida", "kindle")
    hecho = os.path.isdir(kindle) and any(f.startswith(nombre) for f in os.listdir(kindle))
    return hecho and pendientes_de_traducir(epub) == 0


def procesar(epub, glosario, perfil, modelo):
    t0 = time.time()
    traducir.cmd_analizar(epub)
    # dos pasadas: la segunda recoge lotes que fallaron por errores sueltos
    for _ in range(2):
        traducir.cmd_traducir(epub, lote=12, modelo=modelo, glosario_ruta=glosario)
        if not pendientes_de_traducir(epub):
            break
    faltan = pendientes_de_traducir(epub)
    if faltan:
        print(f"  AVISO: {faltan} páginas quedaron sin traducir (se dejan en inglés)")
    traducir.cmd_componer(epub)
    traducir.cmd_empaquetar(epub)
    formato = "mobi" if traducir.buscar_kindlegen() else "epub"
    try:
        traducir.cmd_kindle(epub, perfil, formato=formato)
    except SystemExit as e:
        print(f"  AVISO: la conversión a Kindle falló: {e}")
    print(f"  tomo listo en {(time.time() - t0) / 60:.0f} min")


def sin_suspender():
    """Pide a Windows no suspender el equipo mientras este proceso viva (se libera al salir;
    no cambia la configuración de energía). La pantalla sí se puede apagar."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    sin_suspender()
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--glosario", required=True)
    ap.add_argument("--perfil", default="KPW6")
    ap.add_argument("--modelo", default="opus")
    ap.add_argument("--desde", type=int, default=0, help="número de tomo desde el que empezar")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(REGISTRO), exist_ok=True)
    tomos = sorted((os.path.join(a.carpeta, f) for f in os.listdir(a.carpeta) if f.lower().endswith(".epub")),
                   key=lambda p: numero(os.path.basename(p)))
    tomos = [t for t in tomos if numero(os.path.basename(t)) >= a.desde]
    with open(REGISTRO, "a", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        print(f"\n===== {time.strftime('%Y-%m-%d %H:%M')} · {len(tomos)} tomos · modelo {a.modelo}")
        for i, t in enumerate(tomos, 1):
            nombre = os.path.basename(t)
            if terminado(t):
                print(f"[{i}/{len(tomos)}] {nombre}: ya estaba terminado")
                continue
            print(f"\n[{i}/{len(tomos)}] {nombre}  ({time.strftime('%H:%M')})")
            try:
                procesar(t, a.glosario, a.perfil, a.modelo)
            except SystemExit as e:
                print(f"  ERROR, se pasa al siguiente tomo: {e}")
            except Exception as e:
                print(f"  ERROR {type(e).__name__}: {e} — se pasa al siguiente tomo")
        print(f"\n===== fin {time.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
