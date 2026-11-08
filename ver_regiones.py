"""Dibuja, sobre la página ya borrada, la región donde se permite escribir cada bloque.

    python ver_regiones.py <carpeta_libro> <pagina> [carpeta_traduccion]

Verde: interior real del globo medido tras borrar. Azul: máscara del detector. Rojo: caja
del texto original.
"""
import json
import os
import sys

import cv2
import numpy as np

import pagina
import traducir

w, n = sys.argv[1], int(sys.argv[2])
carpeta = sys.argv[3] if len(sys.argv) > 3 else "traduccion"
meta = json.load(open(os.path.join(w, "unidades", f"{n:04d}.json"), encoding="utf-8"))
img, us, gs = traducir.cargar_pagina(w, dict(page=n, image=meta["image"]))
tr = traducir.traducciones_de(w, n, carpeta)
informe = []
out = pagina.componer(img, us, gs, tr, informe)
vis = out.copy()
activos = {r["id"] for r in informe}
for u in us:
    if u["id"] not in activos:
        continue
    color = (0, 170, 0) if u.get("interior_real") else (255, 120, 0)
    cs, _ = cv2.findContours(u["region"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cs, -1, color, 2)
    x, y, bw, bh = u["box"]
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 1)
    cv2.putText(vis, str(u["id"]), (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
sal = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pruebas", f"regiones_{n:04d}.png")
cv2.imwrite(sal, vis)
for r in informe:
    print(r["id"], r["estilo"], "tam", r["tam"], "tope", r["tope"], "letra", round(r["tope_letra"]), "caja", r["tope_caja"], r["lineas"])
print(sal)
