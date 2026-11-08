"""Pipeline: EPUB de manga en inglés -> EPUB en español (y versión para Kindle).

    python traducir.py analizar   <epub>           detecta globos y texto, guarda etiquetas
    python traducir.py traducir   <epub> [--glosario glosarios/obra.md] [--modelo opus] [--paginas 5-20]
    python traducir.py componer   <epub> [--paginas 5-20]
    python traducir.py empaquetar <epub>           EPUB traducido a resolución completa
    python traducir.py kindle     <epub> [--perfil KPW6]   EPUB optimizado para Kindle (KCC)
    python traducir.py todo       <epub> --glosario glosarios/obra.md   todo lo anterior

La traducción la hace Claude Code en modo no interactivo (`claude -p`): lee las páginas
etiquetadas y devuelve un JSON con el inglés transcrito y el español de cada bloque. Usa la
cuota de la suscripción de Claude, no una API key.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import epub as epub_mod   # extracción del EPUB y orden de páginas
import pagina

LIBROS = os.path.join(AQUI, "books")
GLOSARIO_GENERAL = os.path.join(AQUI, "glosarios", "_general.md")
KCC = os.path.join(AQUI, "tools", "kcc", "tool_cli_c2e_12.1.5_AMD64.exe")


def leer_glosario(ruta):
    """Reglas generales de estilo + el glosario de la obra (si se indica)."""
    partes = [open(GLOSARIO_GENERAL, encoding="utf-8").read()]
    if ruta:
        if not os.path.exists(ruta):
            sys.exit(f"No existe el glosario {ruta}")
        partes.append(open(ruta, encoding="utf-8").read())
    return "\n\n".join(partes)


# ----------------------------------------------------------------------------- utilidades

def dir_libro(epub):
    return os.path.join(LIBROS, epub_mod.slug(os.path.splitext(os.path.basename(epub))[0]))


def paginas_del_libro(epub):
    meta = epub_mod.cargar(epub)
    return meta, [dict(page=i, image=p["image"]) for i, p in enumerate(meta["pages"]) if p.get("image")]


def rango(txt, todas):
    if not txt:
        return todas
    sel = set()
    for parte in txt.split(","):
        if "-" in parte:
            a, b = parte.split("-")
            sel.update(range(int(a), int(b) + 1))
        else:
            sel.add(int(parte))
    return [p for p in todas if p["page"] in sel]


def _guardar_mascaras(ruta, unidades, gs):
    datos = {}
    for u in unidades:
        datos[f"l{u['id']}"] = np.packbits(u["letras"] > 0)
        datos[f"r{u['id']}"] = np.packbits(u["region"] > 0)
    for i, g in enumerate(gs):
        datos[f"g{i}"] = np.packbits(g["mask"] > 0)
    np.savez_compressed(ruta, **datos)


def cargar_pagina(w, p):
    """Imagen, unidades (con máscaras) y globos tal como quedaron en `analizar`."""
    img = cv2.imread(os.path.join(w, p["image"]))
    H, W = img.shape[:2]
    meta = json.load(open(os.path.join(w, "unidades", f"{p['page']:04d}.json"), encoding="utf-8"))
    m = np.load(os.path.join(w, "unidades", f"{p['page']:04d}.npz"))

    def des(k):
        return (np.unpackbits(m[k])[:H * W].reshape(H, W) * 255).astype(np.uint8)

    unidades = []
    for u in meta["unidades"]:
        u = dict(u)
        u["letras"], u["region"] = des(f"l{u['id']}"), des(f"r{u['id']}")
        unidades.append(u)
    gs = [dict(box=g["box"], mask=des(f"g{i}")) for i, g in enumerate(meta["globos"])]
    return img, unidades, gs


# ----------------------------------------------------------------------------- pasos

def cmd_analizar(epub, sel=None, forzar=False):
    _, pags = paginas_del_libro(epub)
    w = dir_libro(epub)
    os.makedirs(os.path.join(w, "unidades"), exist_ok=True)
    os.makedirs(os.path.join(w, "etiquetas"), exist_ok=True)
    t0 = time.time()
    hechas = 0
    for p in rango(sel, pags):
        js = os.path.join(w, "unidades", f"{p['page']:04d}.json")
        if os.path.exists(js) and not forzar:
            continue
        img = cv2.imread(os.path.join(w, p["image"]))
        if img is None:
            continue
        unidades, gs = pagina.analizar(img)
        _guardar_mascaras(js[:-5] + ".npz", unidades, gs)
        json.dump(dict(page=p["page"], image=p["image"],
                       unidades=[dict(id=u["id"], box=u["box"], globo=u["globo"], hg=u["hg"],
                                      hg_glifo=u.get("hg_glifo"), hg_tinta=u.get("hg_tinta"),
                                      anchos=u.get("anchos"), renglones=u.get("renglones"),
                                      xh_tinta=u.get("xh_tinta"))
                                 for u in unidades],
                       globos=[dict(box=g["box"], conf=g["conf"]) for g in gs]),
                  open(js, "w", encoding="utf-8"), ensure_ascii=False)
        cv2.imwrite(os.path.join(w, "etiquetas", f"p{p['page']:04d}.jpg"),
                    pagina.etiquetar(img, unidades, ancho_max=1100), [cv2.IMWRITE_JPEG_QUALITY, 88])
        hechas += 1
        if hechas % 20 == 0:
            print(f"  {hechas} páginas ({time.time() - t0:.0f}s)", flush=True)
    print(f"analizadas {hechas} páginas en {time.time() - t0:.0f}s")


INSTRUCCIONES = """Eres traductor profesional de manga, del inglés al español.

Te paso páginas de manga ya rotuladas: cada bloque de texto detectado tiene un recuadro rojo
con un número (el id). Lee el inglés DIRECTAMENTE de la imagen y tradúcelo.

Reglas:
1. Un id = un bloque. Traduce exactamente el texto que está dentro de ESE recuadro. Si una
   frase del original quedó partida en dos recuadros, reparte la traducción entre ambos para
   que cada uno diga su parte.
2. Traduce el SENTIDO, no palabra por palabra: reordena según la gramática española
   («SUPER REAL NAME» -> «¡UN NOMBRE SÚPER REAL!», no «SUPER NOMBRE REAL»). Debe sonar
   como lo diría un chico o chica de esa edad.
2b. BREVEDAD: el texto va en el mismo globo que el inglés, y el español suele salir más
   largo, lo que obliga a achicar la letra. Busca la forma más corta y natural: procura no
   pasar del largo del original (en caracteres). En gritos, títulos y textos grandes, lo más
   corto posible («EHHHHHHHH??!!!!!!» -> «¡¿EEEEEEH?!»; «"COSRABA" SUBARUN!!!» ->
   «¡¡SUBARUN DE COSRABA!!»). Nunca sacrifiques el sentido por acortar.
3. Español neutro, tuteo, coloquial; frases cortas para que quepan en el globo. Usa ¡! y ¿?
   de apertura y cierre; combinados van en espejo: «¡¿QUÉ?!» (nunca «¡¿QUÉ!?»). Respeta el
   género de quien habla y de a quien se habla (ver glosario).
4. Mantén el tono: gritos, titubeos (N-NO...), puntos suspensivos, ♥ y ♪.
5. Devuelve "" (cadena vacía) si el recuadro es: una onomatopeya dibujada (WHAM, THUD, BAM,
   katakana...), un signo suelto (?, !, ...) sin palabras, marcas de sudor o ruido del dibujo,
   o texto ilegible. Lo vacío se queda como arte original.
6. Usa el contexto de las páginas anteriores del mismo lote para que el diálogo fluya.
7. Sigue el glosario al pie de la letra.
8. Marca el "estilo" de cada bloque según cómo está ROTULADO el original:
   - "normal": diálogo y narración en letra de cómic corriente.
   - "grito": letra grande, gruesa o condensada, tipo impacto (gritos, exclamaciones fuertes).
   - "cursiva": MAYÚSCULAS inclinadas (pensamientos o susurros rotulados en itálica).
   - "manuscrito": letra a mano con minúsculas (notas, comentarios al margen). Solo si el
     original usa minúsculas; si va todo en mayúsculas inclinadas es "cursiva".
   - "rotulo": carteles, títulos, letreros o pantallas dentro del dibujo.
   En "normal", "grito", "cursiva" y "rotulo" escribe TODO EN MAYÚSCULAS. En "manuscrito"
   escribe en minúsculas con mayúscula inicial, como el original.
9. Texto en vertical (letras apiladas una bajo otra): tradúcelo con una palabra corta del
   mismo tipo («Waaaah~» -> «Guaaa~»), sin espacios, para que se pueda apilar igual.
10. Si un mismo texto aparece repetido detrás de un globo como efecto de eco, tradúcelo igual;
   el programa decide si se deja como arte.

Responde SOLO con JSON válido, sin texto antes ni después, con esta forma:
{"<pagina>": {"<id>": {"en": "<inglés leído>", "es": "<traducción>", "estilo": "normal"}, ...}, ...}
Incluye TODOS los ids de cada página, también los vacíos.
"""


def _pedir_a_claude(rutas, paginas, modelo, glosario, previo):
    lista = "\n".join(f"- página {n}: {r}" for n, r in zip(paginas, rutas))
    prompt = (INSTRUCCIONES + "\n## Glosario\n" + glosario +
              ("\n## Últimas líneas de la página anterior (contexto)\n" + previo if previo else "") +
              "\n## Páginas de este lote (léelas con la herramienta Read, en orden)\n" + lista)
    cmd = ["claude", "-p", "--model", modelo, "--output-format", "json",
           "--allowedTools", "Read", "--no-session-persistence",
           "--add-dir", os.path.dirname(rutas[0])]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                       timeout=900, cwd=os.path.dirname(rutas[0]))
    try:
        sobre = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(r.stderr[-800:] or r.stdout[-800:])
    texto = sobre.get("result", "")
    if sobre.get("is_error"):
        if sobre.get("api_error_status") in (401, 403):
            raise PermissionError(texto)      # problema de cuenta: no tiene caso reintentar
        raise RuntimeError(texto[:500])
    m = re.search(r"\{.*\}", texto, re.S)
    if not m:
        raise RuntimeError("sin JSON en la respuesta: " + texto[:300])
    return json.loads(m.group(0)), sobre.get("total_cost_usd")


def cmd_traducir(epub, sel=None, lote=8, modelo="opus", forzar=False, carpeta="traduccion", glosario_ruta=None):
    _, pags = paginas_del_libro(epub)
    w = dir_libro(epub)
    dtr = os.path.join(w, carpeta)
    os.makedirs(dtr, exist_ok=True)
    glosario = leer_glosario(glosario_ruta)
    pend = []
    for p in rango(sel, pags):
        js = os.path.join(w, "unidades", f"{p['page']:04d}.json")
        if not os.path.exists(js):
            continue
        if not json.load(open(js, encoding="utf-8"))["unidades"]:
            continue
        if os.path.exists(os.path.join(dtr, f"{p['page']:04d}.json")) and not forzar:
            continue
        pend.append(p["page"])
    print(f"{len(pend)} páginas por traducir, lotes de {lote}")
    previo = ""
    for i in range(0, len(pend), lote):
        grupo = pend[i:i + lote]
        rutas = [os.path.join(w, "etiquetas", f"p{n:04d}.jpg") for n in grupo]
        t0 = time.time()
        for intento in range(3):
            try:
                res, costo = _pedir_a_claude(rutas, grupo, modelo, glosario, previo)
                break
            except PermissionError as e:
                sys.exit(f"Claude Code no puede usar la cuenta: {e}")
            except Exception as e:
                print(f"  lote {grupo[0]}-{grupo[-1]} intento {intento + 1} falló: {e}", flush=True)
                time.sleep(10)
        else:
            continue
        for n in grupo:
            datos = res.get(str(n)) or {}
            json.dump(datos, open(os.path.join(dtr, f"{n:04d}.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        ult = res.get(str(grupo[-1])) or {}
        previo = "\n".join(v.get("es", "") for v in ult.values() if isinstance(v, dict) and v.get("es"))[-600:]
        print(f"  páginas {grupo[0]}-{grupo[-1]} listas ({time.time() - t0:.0f}s"
              + (f", ${costo:.3f}" if costo else "") + ")", flush=True)


def traducciones_de(w, n, carpeta="traduccion"):
    ruta = os.path.join(w, carpeta, f"{n:04d}.json")
    if not os.path.exists(ruta):
        return {}
    datos = json.load(open(ruta, encoding="utf-8"))
    return {k: (v if isinstance(v, dict) else {"es": v}) for k, v in datos.items()}


def salida_de(carpeta):
    """Carpeta de páginas compuestas: «es» para la traducción principal, «es_<x>» para otras."""
    return "es" if carpeta == "traduccion" else "es_" + carpeta


def avisos_de(r, u):
    """Motivos para revisar un bloque compuesto (los usa la herramienta de revisión)."""
    av = []
    if r.get("omitido") == "eco":
        av.append("eco")
        return av
    if r.get("omitido"):
        return av
    ref = u.get("hg_tinta") or u.get("hg")
    puesta = r["mayus"]
    xh = u.get("xh_tinta")
    if pagina.usa_minusculas(r["estilo"]) and xh and ref and 0.35 * ref <= xh <= 0.95 * ref:
        ref, puesta = xh, r["tam"] * pagina.altura_x(r["estilo"])
    r["ratio"] = round(puesta / ref, 2) if ref else None
    if r["ratio"] and r["ratio"] < 0.8:
        av.append("chico")
    if r["tam"] <= 10:
        av.append("mínimo")
    if r.get("vertical"):
        av.append("vertical")
    return av


def componer_pagina(w, p, carpeta="traduccion"):
    """Compone una página y guarda la imagen y su informe. Devuelve el informe o None."""
    tr = traducciones_de(w, p["page"], carpeta)
    if not any((v.get("es") or "").strip() for v in tr.values()):
        return None
    img, unidades, gs = cargar_pagina(w, p)
    por = {u["id"]: u for u in unidades}
    informe = []
    out = pagina.componer(img, unidades, gs, tr, informe)
    dout = os.path.join(w, salida_de(carpeta))
    os.makedirs(dout, exist_ok=True)
    cv2.imwrite(os.path.join(dout, f"{p['page']:04d}.png"), out)
    limpio = []
    for r in informe:
        r = {k: v for k, v in r.items() if k not in ("plan",)}
        r["avisos"] = avisos_de(r, por[r["id"]])
        limpio.append(json.loads(json.dumps(r, default=float)))
    dinf = os.path.join(w, "informe_" + carpeta)
    os.makedirs(dinf, exist_ok=True)
    json.dump(dict(page=p["page"], alto=img.shape[0], ancho=img.shape[1], bloques=limpio),
              open(os.path.join(dinf, f"{p['page']:04d}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return limpio


def cmd_componer(epub, sel=None, carpeta="traduccion"):
    _, pags = paginas_del_libro(epub)
    w = dir_libro(epub)
    t0 = time.time()
    n = avisos = 0
    for p in rango(sel, pags):
        inf = componer_pagina(w, p, carpeta)
        if inf is None:
            continue
        n += 1
        avisos += sum(1 for r in inf if r["avisos"])
    print(f"compuestas {n} páginas en {time.time() - t0:.0f}s · {avisos} bloques para revisar")


def cmd_empaquetar(epub, salida=None):
    meta, pags = paginas_del_libro(epub)
    w = dir_libro(epub)
    build = os.path.join(w, "build")
    if os.path.isdir(build):
        shutil.rmtree(build)
    with zipfile.ZipFile(epub) as z:
        z.extractall(build)
    puestas = 0
    for p in pags:
        rp = os.path.join(w, "es", f"{p['page']:04d}.png")
        if not os.path.exists(rp):
            continue
        destino = os.path.join(build, p["image"])
        img = cv2.imread(rp)
        ext = os.path.splitext(destino)[1].lower()
        params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext in (".jpg", ".jpeg") else []
        cv2.imwrite(destino, img, params)
        puestas += 1
    for raiz, _, archivos in os.walk(build):
        for f in archivos:
            if f.endswith(".opf"):
                ruta = os.path.join(raiz, f)
                s = open(ruta, encoding="utf-8-sig", errors="ignore").read()
                s = re.sub(r"<dc:language>[^<]*</dc:language>", "<dc:language>es</dc:language>", s)
                t = re.search(r"<dc:title>(.*?)</dc:title>", s)
                if t and "(Español)" not in t.group(1):
                    s = s.replace(t.group(0), f"<dc:title>{t.group(1)} (Español)</dc:title>")
                open(ruta, "w", encoding="utf-8").write(s)
    salida = salida or os.path.join(AQUI, "salida", os.path.splitext(os.path.basename(epub))[0] + " [ES].epub")
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    if os.path.exists(salida):
        os.remove(salida)
    with zipfile.ZipFile(salida, "w") as z:
        z.write(os.path.join(build, "mimetype"), "mimetype", compress_type=zipfile.ZIP_STORED)
        for raiz, _, archivos in os.walk(build):
            for f in archivos:
                if f == "mimetype":
                    continue
                fp = os.path.join(raiz, f)
                z.write(fp, os.path.relpath(fp, build).replace("\\", "/"),
                        compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    shutil.rmtree(build)
    print(f"{puestas} páginas traducidas -> {salida}")


def _metadatos(epub):
    """(título, autor) del OPF del EPUB original."""
    with zipfile.ZipFile(epub) as z:
        opf = next((n for n in z.namelist() if n.endswith(".opf")), None)
        s = z.read(opf).decode("utf-8", "ignore") if opf else ""
    t = re.search(r"<dc:title[^>]*>(.*?)</dc:title>", s, re.S)
    a = re.search(r"<dc:creator[^>]*>(.*?)</dc:creator>", s, re.S)
    titulo = (t.group(1).strip() if t else os.path.splitext(os.path.basename(epub))[0])
    return titulo, (a.group(1).strip() if a else "")


def buscar_kindlegen():
    """Carpeta que contiene kindlegen(.exe): PATH, C:\\Tools o Kindle Previewer 3."""
    import glob
    exe = "kindlegen.exe" if os.name == "nt" else "kindlegen"
    if shutil.which("kindlegen"):
        return os.path.dirname(shutil.which("kindlegen"))
    candidatos = [r"C:\Tools\kindlegen.exe"]
    for base in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles", ""),
                 os.environ.get("ProgramFiles(x86)", "")):
        if base:
            candidatos += glob.glob(os.path.join(base, "Amazon", "Kindle Previewer 3", "**", exe), recursive=True)
    candidatos += glob.glob(r"C:\Program Files\WindowsApps\KindlePreviewerApp_*\**\kindlegen.exe", recursive=True)
    candidatos += glob.glob("/Applications/Kindle Previewer 3.app/**/kindlegen", recursive=True)
    for c in candidatos:
        if os.path.isfile(c):
            return os.path.dirname(c)
    return None


def cmd_kindle(epub, perfil="KPW6", occidental=False, kcc=None, formato="mobi"):
    """Versión para Kindle con Kindle Comic Converter (KCC): páginas ajustadas a la pantalla
    del modelo y lectura de derecha a izquierda.

    formato «mobi»: para copiar por cable USB a la carpeta documents (necesita kindlegen, que
    viene con Kindle Previewer 3). formato «epub»: para la app o la web de Send to Kindle."""
    kcc = kcc or KCC
    if not os.path.exists(kcc):
        sys.exit("No encuentro KCC. Ejecuta primero:  python instalar.py")
    entorno = dict(os.environ)
    if formato == "mobi":
        kg = buscar_kindlegen()
        if not kg:
            sys.exit("Para MOBI (USB) falta kindlegen: instala Kindle Previewer 3 de Amazon\n"
                     "  https://www.amazon.com/Kindle-Previewer/b?node=21381691011\n"
                     "o usa --formato epub para enviarlo con Send to Kindle.")
        entorno["PATH"] = kg + os.pathsep + entorno.get("PATH", "")
    _, pags = paginas_del_libro(epub)
    w = dir_libro(epub)
    nombre = os.path.splitext(os.path.basename(epub))[0] + " [ES]"
    carpeta = os.path.join(w, "kindle", nombre)       # KCC nombra el archivo como la carpeta
    if os.path.isdir(carpeta):
        shutil.rmtree(carpeta)
    os.makedirs(carpeta)
    traducidas = 0
    for i, p in enumerate(pags):
        es = os.path.join(w, "es", f"{p['page']:04d}.png")
        origen = es if os.path.exists(es) else os.path.join(w, p["image"])
        traducidas += origen == es
        shutil.copy(origen, os.path.join(carpeta, f"{i:04d}{os.path.splitext(origen)[1].lower()}"))
    # Título del nombre del archivo (los metadatos de origen suelen venir sucios)
    titulo = os.path.splitext(os.path.basename(epub))[0] + " (Español)"
    _, autor = _metadatos(epub)
    dsal = os.path.join(AQUI, "salida", "kindle")
    os.makedirs(dsal, exist_ok=True)
    t0 = time.time()
    cmd = [kcc, "-p", perfil, "-f", "MOBI" if formato == "mobi" else "EPUB", "-u",
           "-t", titulo, "-o", dsal, "--language", "es"]
    if formato == "epub":
        cmd += ["--nokepub", "--ts", "190"]
    if autor and autor.lower() != "unknown":
        cmd += ["-a", autor]
    if not occidental:
        cmd.append("-m")                  # manga: derecha a izquierda
    cmd.append(carpeta)
    print(f"KCC ({perfil}, {formato.upper()}): {len(pags)} páginas, {traducidas} traducidas...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=entorno)
    if r.returncode != 0 or "Error" in (r.stdout or ""):
        sys.exit("KCC falló:\n" + ((r.stderr or "") + (r.stdout or ""))[-1500:])
    shutil.rmtree(os.path.dirname(carpeta))
    for f in sorted(os.listdir(dsal)):
        ruta = os.path.join(dsal, f)
        if f.startswith(nombre) and os.path.getmtime(ruta) >= t0 - 1:
            print(f"  -> {ruta} ({os.path.getsize(ruta) / 1e6:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["analizar", "traducir", "componer", "empaquetar", "kindle", "todo"])
    ap.add_argument("epub")
    ap.add_argument("--glosario", help="glosario de la obra (p. ej. glosarios/my-dress-up-darling.md)")
    ap.add_argument("--paginas")
    ap.add_argument("--lote", type=int, default=12)
    ap.add_argument("--modelo", default="opus")
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--carpeta", default="traduccion",
                    help="carpeta de traducciones (p. ej. para comparar modelos)")
    ap.add_argument("--perfil", default="KPW6",
                    help="perfil de Kindle para KCC: KPW6 = Paperwhite 12.ª gen (2024); KPW5 = Paperwhite 11.ª; "
                         "K11 = Kindle básico 11.ª/2024; KS = Scribe")
    ap.add_argument("--occidental", action="store_true", help="lectura de izquierda a derecha (cómic, no manga)")
    ap.add_argument("--kcc", help="ruta a kcc-c2e si no es la que descarga instalar.py (macOS/Linux)")
    ap.add_argument("--formato", choices=["mobi", "epub"], default="mobi",
                    help="mobi: copiar por USB (necesita Kindle Previewer 3); epub: Send to Kindle (app/web)")
    a = ap.parse_args()
    if a.paso in ("analizar", "todo"):
        cmd_analizar(a.epub, a.paginas, a.forzar)
    if a.paso in ("traducir", "todo"):
        cmd_traducir(a.epub, a.paginas, a.lote, a.modelo, a.forzar, a.carpeta, a.glosario)
    if a.paso in ("componer", "todo"):
        cmd_componer(a.epub, a.paginas, a.carpeta)
    if a.paso == "empaquetar" or (a.paso == "todo" and not a.paginas):
        cmd_empaquetar(a.epub)
    if a.paso == "kindle" or (a.paso == "todo" and not a.paginas):
        cmd_kindle(a.epub, a.perfil, a.occidental, a.kcc, a.formato)
