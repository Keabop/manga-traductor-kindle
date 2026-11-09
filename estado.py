"""Estado de una traducción por lotes: cuánto lleva, cuánto falta y tiempo estimado.

    .venv\\Scripts\\python estado.py "<carpeta con los EPUB>"
"""
import json
import os
import re
import sys
import time
import zipfile

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import epub as epub_mod  # noqa: E402

REGISTRO = os.path.join(AQUI, "salida", "progreso.log")
SEG_ANALISIS, SEG_COMPOSICION, SEG_KINDLE = 1.3, 2.0, 40     # medidos en el tomo 6 (RTX 5070 Ti)


def paginas_con_imagen(epub):
    with zipfile.ZipFile(epub) as z:
        return sum(1 for n in z.namelist() if re.search(r"\.(jpe?g|png|webp)$", n, re.I)) - 1   # sin portada


def estado_tomo(epub):
    w = os.path.join(AQUI, "books", epub_mod.slug(os.path.splitext(os.path.basename(epub))[0]))
    nombre = os.path.splitext(os.path.basename(epub))[0] + " [ES]"
    kindle = os.path.join(AQUI, "salida", "kindle")
    listo = os.path.isdir(kindle) and any(f.startswith(nombre) for f in os.listdir(kindle))
    du = os.path.join(w, "unidades")
    if not os.path.isdir(du):
        return dict(listo=False, analizado=False, hechas=0, total=paginas_con_imagen(epub))
    con_texto = [f for f in os.listdir(du) if f.endswith(".json")
                 and json.load(open(os.path.join(du, f), encoding="utf-8"))["unidades"]]
    hechas = sum(os.path.exists(os.path.join(w, "traduccion", f)) for f in con_texto)
    return dict(listo=listo and hechas == len(con_texto), analizado=True, hechas=hechas, total=len(con_texto))


def seg_por_pagina():
    """Promedio de traducción por página en la última corrida (sin contar esperas de cuota)."""
    if not os.path.exists(REGISTRO):
        return 13.0
    datos = re.findall(r"páginas (\d+)-(\d+) listas \((\d+)s", open(REGISTRO, encoding="utf-8").read())[-20:]
    pags = sum(int(b) - int(a) + 1 for a, b, _ in datos)
    return sum(int(s) for *_, s in datos) / pags if pags else 13.0


def main():
    carpeta = sys.argv[1]
    tomos = sorted((os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.lower().endswith(".epub")),
                   key=lambda p: int((re.search(r"vol\.?\s*(\d+)", p, re.I) or re.search(r"(\d+)", p)).group(1)))
    spp = seg_por_pagina()
    listos, falta_seg, pags_hechas, pags_total, actual = 0, 0.0, 0, 0, None
    for t in tomos:
        e = estado_tomo(t)
        pags_hechas += e["hechas"]
        pags_total += e["total"]
        if e["listo"]:
            listos += 1
            continue
        if actual is None and e["analizado"]:
            actual = (os.path.basename(t), e)
        restantes = e["total"] - e["hechas"]
        falta_seg += restantes * spp + e["total"] * SEG_COMPOSICION + SEG_KINDLE
        if not e["analizado"]:
            falta_seg += e["total"] * SEG_ANALISIS
    print(f"Tomos listos: {listos} de {len(tomos)}")
    if actual:
        n, e = actual
        print(f"En curso: {n} · {e['hechas']}/{e['total']} páginas traducidas")
    print(f"Páginas traducidas en total: {pags_hechas} de ~{pags_total}")
    h, m = divmod(int(falta_seg // 60), 60)
    fin = time.strftime("%d/%m %H:%M", time.localtime(time.time() + falta_seg))
    print(f"Tiempo estimado: {h} h {m:02d} min de trabajo (≈ {fin}), sin contar esperas por cuota")
    if os.path.exists(REGISTRO):
        ult = [l for l in open(REGISTRO, encoding="utf-8").read().splitlines() if l.strip()][-1]
        if "cuota agotada" in ult:
            print("Ahora: esperando a que se renueve la cuota — " + ult.split("reintento")[-1].strip())


if __name__ == "__main__":
    main()
