"""Herramienta de revisión: corrige bloques sueltos sin volver a traducir.

    .venv\\Scripts\\python revisar.py "<epub>"        abre http://localhost:8790

Muestra el original junto a la traducida, con los bloques sospechosos marcados (letra
chica, eco que se dejó como arte, texto vertical, letra mínima). Por bloque se puede
cambiar el texto, el estilo, la escala de la letra o dejarlo como en el original. Al
guardar se recompone solo esa página (unos segundos); el EPUB se vuelve a armar con
«Armar EPUB».
"""
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import traducir  # noqa: E402

PUERTO = 8790        # 8765 lo usa el companion de la SmallTV
CARPETA = "traduccion"
CAMPOS = ("es", "estilo", "escala", "omitir")
ESTILOS = ("normal", "grito", "cursiva", "manuscrito", "rotulo")

candado = threading.Lock()


class Libro:
    def __init__(self, epub):
        self.epub = epub
        self.w = traducir.dir_libro(epub)
        _, pags = traducir.paginas_del_libro(epub)
        self.pags = {p["page"]: p for p in pags}

    def ruta_tr(self, n):
        return os.path.join(self.w, CARPETA, f"{n:04d}.json")

    def informe(self, n):
        ruta = os.path.join(self.w, "informe_" + CARPETA, f"{n:04d}.json")
        return json.load(open(ruta, encoding="utf-8")) if os.path.exists(ruta) else None

    def resumen(self):
        out = []
        for n in sorted(self.pags):
            inf = self.informe(n)
            avisos = [a for r in (inf or {}).get("bloques", []) for a in r.get("avisos", [])]
            out.append(dict(n=n, traducida=inf is not None, avisos=len(avisos),
                            tipos=sorted(set(avisos)), revisada=self._revisada(n)))
        return out

    def _revisada(self, n):
        ruta = self.ruta_tr(n)
        if not os.path.exists(ruta):
            return False
        return bool(json.load(open(ruta, encoding="utf-8")).get("_revisada"))

    def detalle(self, n):
        inf = self.informe(n) or {}
        tr = json.load(open(self.ruta_tr(n), encoding="utf-8")) if os.path.exists(self.ruta_tr(n)) else {}
        meta = json.load(open(os.path.join(self.w, "unidades", f"{n:04d}.json"), encoding="utf-8"))
        por_inf = {r["id"]: r for r in inf.get("bloques", [])}
        bloques = []
        for u in meta["unidades"]:
            t = tr.get(str(u["id"])) or {}
            if not isinstance(t, dict):
                t = {"es": t}
            r = por_inf.get(u["id"], {})
            if not (t.get("es") or "").strip():
                continue
            bloques.append(dict(id=u["id"], box=u["box"], en=t.get("en", ""), es=t.get("es", ""),
                                estilo=r.get("estilo") or t.get("estilo", "normal"), escala=t.get("escala", 1.0),
                                omitir=bool(t.get("omitir")) or r.get("omitido") == "eco",
                                avisos=r.get("avisos", []), tam=r.get("tam"), ratio=r.get("ratio"),
                                estilo_puesto=r.get("estilo")))
        alto, ancho = inf.get("alto"), inf.get("ancho")
        if not alto:
            import cv2
            im = cv2.imread(os.path.join(self.w, meta["image"]))
            alto, ancho = im.shape[:2]
        return dict(n=n, alto=alto, ancho=ancho, bloques=bloques, revisada=bool(tr.get("_revisada")),
                    traducida=os.path.exists(os.path.join(self.w, traducir.salida_de(CARPETA), f"{n:04d}.png")))

    def guardar(self, n, cambios, revisada):
        ruta = self.ruta_tr(n)
        tr = json.load(open(ruta, encoding="utf-8")) if os.path.exists(ruta) else {}
        for k, c in cambios.items():
            t = tr.get(k) or {}
            if not isinstance(t, dict):
                t = {"es": t}
            if "es" in c:
                t["es"] = str(c["es"]).strip()
            if c.get("estilo") in ESTILOS:
                if c["estilo"] != t.get("estilo"):
                    t["estilo_fijo"] = True      # no volver a adivinar el estilo
                t["estilo"] = c["estilo"]
            if "escala" in c:
                e = max(0.5, min(2.0, float(c["escala"])))
                if abs(e - 1.0) < 0.01:
                    t.pop("escala", None)
                else:
                    t["escala"] = round(e, 2)
            if "omitir" in c:
                t["omitir"] = bool(c["omitir"])
            if t.get("estilo", "normal") != "manuscrito" and t.get("es"):
                t["es"] = t["es"].upper()        # todos los estilos menos el manuscrito van en mayúsculas
            tr[k] = t
        tr["_revisada"] = bool(revisada)
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        json.dump(tr, open(ruta, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with candado:
            traducir.componer_pagina(self.w, self.pags[n], CARPETA)
        return self.detalle(n)

    def imagen(self, cual, n):
        if cual == "es":
            ruta = os.path.join(self.w, traducir.salida_de(CARPETA), f"{n:04d}.png")
            if os.path.exists(ruta):
                return ruta
        return os.path.join(self.w, self.pags[n]["image"])


LIBRO = None


class Manejador(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _enviar(self, codigo, cuerpo, tipo="application/json; charset=utf-8", cache=False):
        if isinstance(cuerpo, (dict, list)):
            cuerpo = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
        elif isinstance(cuerpo, str):
            cuerpo = cuerpo.encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _archivo(self, ruta, cache=False):
        if not os.path.exists(ruta):
            return self._enviar(404, {"error": "no existe"})
        tipo = mimetypes.guess_type(ruta)[0] or "application/octet-stream"
        self._enviar(200, open(ruta, "rb").read(), tipo, cache)

    def do_GET(self):
        p = urlparse(self.path).path
        partes = [unquote(x) for x in p.strip("/").split("/") if x]
        try:
            if not partes:
                return self._archivo(os.path.join(AQUI, "revision", "index.html"))
            if partes[0] == "fuente":
                return self._archivo(os.path.join(AQUI, "fonts", "ArchivoNarrowwght.ttf"), cache=True)
            if partes[0] == "api" and partes[1] == "libro":
                return self._enviar(200, dict(titulo=os.path.splitext(os.path.basename(LIBRO.epub))[0],
                                              paginas=LIBRO.resumen()))
            if partes[0] == "api" and partes[1] == "pagina":
                return self._enviar(200, LIBRO.detalle(int(partes[2])))
            if partes[0] == "img":
                return self._archivo(LIBRO.imagen(partes[1], int(partes[2])))
        except (KeyError, IndexError, ValueError, FileNotFoundError) as e:
            return self._enviar(404, {"error": str(e)})
        self._enviar(404, {"error": "ruta desconocida"})

    def do_POST(self):
        partes = [x for x in urlparse(self.path).path.strip("/").split("/") if x]
        largo = int(self.headers.get("Content-Length") or 0)
        datos = json.loads(self.rfile.read(largo) or b"{}")
        try:
            if partes[:2] == ["api", "pagina"]:
                n = int(partes[2])
                cambios = {str(k): {c: v for c, v in d.items() if c in CAMPOS}
                           for k, d in (datos.get("bloques") or {}).items()}
                return self._enviar(200, LIBRO.guardar(n, cambios, datos.get("revisada", False)))
            if partes[:2] == ["api", "empaquetar"]:
                with candado:
                    traducir.cmd_empaquetar(LIBRO.epub)
                return self._enviar(200, {"ok": True})
        except Exception as e:  # el error se muestra en la página
            return self._enviar(500, {"error": f"{type(e).__name__}: {e}"})
        self._enviar(404, {"error": "ruta desconocida"})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    LIBRO = Libro(sys.argv[1])
    ThreadingHTTPServer.allow_reuse_address = False      # no compartir el puerto con otro programa
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PUERTO), Manejador)
    except OSError:
        sys.exit(f"El puerto {PUERTO} está ocupado por otro programa; cambia PUERTO en revisar.py.")
    url = f"http://localhost:{PUERTO}"
    print(f"Revisión de «{os.path.basename(sys.argv[1])}» en {url}  (Ctrl+C para salir)")
    if "--sin-navegador" not in sys.argv:
        webbrowser.open(url)
    srv.serve_forever()
