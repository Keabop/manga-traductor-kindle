"""Mide cuánto se parece el tamaño del texto traducido al original.

    python medir.py <carpeta_libro> 5-16

Por cada bloque traducido compara la altura de mayúscula puesta con la altura real de las
letras del original (medida sobre la tinta, no sobre la máscara del modelo) y dice qué
limitó el tamaño: el tope o el espacio disponible.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import pagina
import traducir


def altura_tinta(gray, u, claro):
    """Mediana de la altura de las letras del original, medida en los píxeles de tinta."""
    x, y, w, h = u["box"]
    H, W = gray.shape
    x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(W, x + w + 3), min(H, y + h + 3)
    zona = cv2.dilate(u["letras"], np.ones((5, 5), np.uint8))[y0:y1, x0:x1] > 0
    g = gray[y0:y1, x0:x1]
    tinta = ((g > 150) if claro else (g < 110)) & zona
    n, _, st, _ = cv2.connectedComponentsWithStats(tinta.astype(np.uint8), 8)
    hs = [s[3] for s in st[1:] if s[4] >= 8 and s[3] >= 5]
    if not hs:
        return None
    alto = max(hs)
    hs = [v for v in hs if v >= 0.45 * alto and v <= 1.6 * np.median(hs)]
    return float(np.median(hs)) if hs else None


def medir(w, paginas, carpeta="traduccion"):
    filas = []
    for n in paginas:
        js = os.path.join(w, "unidades", f"{n:04d}.json")
        if not os.path.exists(js):
            continue
        meta = json.load(open(js, encoding="utf-8"))
        img, us, gs = traducir.cargar_pagina(w, dict(page=n, image=meta["image"]))
        tr = traducir.traducciones_de(w, n, carpeta)
        informe = []
        pagina.componer(img, us, gs, tr, informe)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        por_id = {u["id"]: u for u in us}
        d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        for r in informe:
            u = por_id[r["id"]]
            claro, _ = pagina._color_texto(gray, u["letras"])
            orig = altura_tinta(gray, u, claro)
            puesta = r["mayus"]
            if pagina.usa_minusculas(r["estilo"]):
                # manuscrito: se comparan minúsculas con minúsculas
                xo = pagina.altura_x_tinta(gray, u["letras"], u["box"], claro, orig or u["hg"])
                if xo and orig and 0.35 * orig <= xo <= 0.95 * orig:
                    orig, puesta = xo, r["tam"] * pagina.altura_x(r["estilo"])
            f = pagina.fuente(r["tam"], r["estilo"])
            ancho = max(pagina.largo(d, l, f) for l in r["lineas"]) * r.get("cond", 1.0)
            alto = len(r["lineas"]) * r["tam"] * pagina.interlineado(r["estilo"])
            area = (ancho * alto) / max(1, u["box"][2] * u["box"][3])
            filas.append(dict(pag=n, id=r["id"], estilo=r["estilo"], orig=orig, puesta=puesta,
                              ratio=(puesta / orig) if orig else None, area=area,
                              limite="tope" if r["tam"] >= r["tope"] else "espacio",
                              tam=r["tam"], tope=r["tope"], hg=u["hg"], es=r["es"][:30],
                              cond=r.get("cond", 1.0)))
    return filas


if __name__ == "__main__":
    w = sys.argv[1]
    a, b = sys.argv[2].split("-")
    filas = medir(w, range(int(a), int(b) + 1), sys.argv[3] if len(sys.argv) > 3 else "traduccion")
    print(f"{'pág':>3} {'id':>3} {'estilo':<10} {'orig':>5} {'puesta':>6} {'ratio':>5} {'área':>5} {'límite':<8} texto")
    for f in filas:
        print(f"{f['pag']:>3} {f['id']:>3} {f['estilo']:<10} {f['orig'] or 0:>5.1f} {f['puesta']:>6.1f} "
              f"{f['ratio'] or 0:>5.2f} {f['area']:>5.2f} {f['limite']:<8} c{f['cond']:.2f} {f['es']}")
    rs = [f["ratio"] for f in filas if f["ratio"]]
    print(f"\n{len(filas)} bloques · ratio de altura mediana {np.median(rs):.2f} · "
          f"p25 {np.percentile(rs, 25):.2f} · p75 {np.percentile(rs, 75):.2f}")
    for lim in ("tope", "espacio"):
        sub = [f["ratio"] for f in filas if f["limite"] == lim and f["ratio"]]
        if sub:
            print(f"  limitados por {lim}: {len(sub)} · ratio mediana {np.median(sub):.2f}")
