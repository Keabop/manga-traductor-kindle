"""Análisis y composición de una página.

analizar(img)  -> unidades de texto (un bloque por globo o por texto suelto)
componer(img, unidades, traducciones) -> página con el inglés borrado y el español puesto

Cada unidad junta los píxeles de texto (máscara de comic-text-detector) que van juntos y sabe
en qué globo cae (máscara del YOLO de globos). El LLM decide luego qué se traduce: lo que
devuelva vacío (onomatopeyas dibujadas, ruido) se queda como arte y no se toca.
"""
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import globos as det_globos
import texto as det_texto
from globos import crear_sesion

AQUI = os.path.dirname(os.path.abspath(__file__))
FUENTE = os.path.join(AQUI, "fonts", "Oswaldwght.ttf")
VARIANTE = "SemiBold"
LAMA = os.path.join(AQUI, "models", "lama_fp32.onnx")


# ----------------------------------------------------------------------------- análisis

def _altura_glifo(binaria):
    n, _, st, _ = cv2.connectedComponentsWithStats(binaria, 8)
    hs = [s[3] for s in st[1:] if s[3] >= 4 and s[4] >= 8]
    return float(np.median(hs)) if hs else 12.0


def _altura_renglon(letras):
    """Altura típica de renglón por proyección horizontal. Más fiable que los glifos cuando
    las letras tienen contorno o están rotas en piezas (gritos grandes)."""
    filas = (letras > 0).sum(axis=1) > 0
    runs, n = [], 0
    for f in filas:
        if f:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
    runs = [r for r in runs if r >= 4]
    return float(np.median(runs)) if runs else 0.0


def _sin_dibujos(letras, gray):
    """Quita de la máscara de texto los dibujos que el detector marcó como texto (un
    alfiler, una engrapadora junto a su etiqueta) para no borrarlos ni agruparlos.

    Se mira la tinta real (trazos oscuros dentro de la máscara): un dibujo es un trazo mucho
    más alto y más grande que las letras que tiene ALREDEDOR. En un grito todas las letras son
    grandes, así que se comparan entre sí y no se tocan."""
    # solo texto oscuro sobre claro: en texto blanco sobre negro el «oscuro» es el fondo
    nm, labm = cv2.connectedComponents((letras > 0).astype(np.uint8), connectivity=8)
    oscuras = np.zeros(letras.shape, np.uint8)
    for j in range(1, nm):
        m = (labm == j).astype(np.uint8) * 255
        claro, _ = _color_texto(gray, m)
        if not claro:
            oscuras |= m
    zona = cv2.dilate(oscuras, np.ones((5, 5), np.uint8)) > 0
    tinta = ((gray < 110) & zona).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(tinta, 8)
    idx = [i for i in range(1, n) if st[i, 4] >= 8 and st[i, 3] >= 4]
    if len(idx) < 4:
        return letras
    hs = np.array([st[i, 3] for i in idx], float)
    areas = np.array([st[i, 2] * st[i, 3] for i in idx], float)
    cs = np.array([cen[i] for i in idx], float)
    out = letras
    for k, i in enumerate(idx):
        h = hs[k]
        # mucho más ancho que alto = palabra con letras unidas (cursiva, letras con contorno)
        if st[i, 2] > 2.2 * h:
            continue
        radio = 3.0 * h
        d = np.hypot(*(cs - cs[k]).T)
        vec = (d <= radio) & (np.arange(len(idx)) != k)
        if vec.sum() < 3:
            continue
        # letra de una palabra grande («EH?»): tiene al lado, en la misma fila y casi pegado,
        # otro trazo de altura parecida. Un dibujo (alfiler, engrapadora) no.
        x, y, w = st[i, 0], st[i, 1], st[i, 2]
        vecina = False
        for j in idx:
            if j == i or not (0.7 * h <= st[j, 3] <= 1.4 * h):
                continue
            ov = min(y + h, st[j, 1] + st[j, 3]) - max(y, st[j, 1])
            gap = max(x, st[j, 0]) - min(x + w, st[j, 0] + st[j, 2])
            if ov >= 0.7 * min(h, st[j, 3]) and gap <= 0.35 * h:
                vecina = True
                break
        if vecina:
            continue
        ref_h = np.percentile(hs[vec], 75)
        ref_a = np.median(areas[vec])
        if h > 1.6 * ref_h and areas[k] > 4 * ref_a:
            comp = (lab == i).astype(np.uint8) * 255
            comp = _fill_holes(cv2.dilate(comp, np.ones((5, 5), np.uint8)))
            if out is letras:
                out = letras.copy()
            out[comp > 0] = 0
    return out


def _fill_holes(mask):
    """Rellena los huecos cerrados de una máscara 0/255."""
    h, w = mask.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    ff[1:-1, 1:-1] = (mask > 0).astype(np.uint8) * 255
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    huecos = cv2.bitwise_not(ff)[1:-1, 1:-1]
    return ((mask > 0) | (huecos > 0)).astype(np.uint8) * 255


def altura_tinta(gray, letras, box, claro):
    """Altura típica de letra del original medida en los píxeles de tinta (no en la máscara
    borrosa del modelo). Es la medición más fiable para diálogo y manuscrito."""
    x, y, w, h = box
    H, W = gray.shape
    x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(W, x + w + 3), min(H, y + h + 3)
    zona = cv2.dilate(letras, np.ones((5, 5), np.uint8))[y0:y1, x0:x1] > 0
    g = gray[y0:y1, x0:x1]
    tinta = ((g > 150) if claro else (g < 110)) & zona
    n, _, st, _ = cv2.connectedComponentsWithStats(tinta.astype(np.uint8), 8)
    hs = [s[3] for s in st[1:] if s[4] >= 8 and s[3] >= 5]
    if len(hs) < 2:
        return None
    alto = max(hs)
    hs = [v for v in hs if 0.45 * alto <= v <= 1.6 * np.median(hs)]
    return float(np.median(hs)) if hs else None


def altura_x_tinta(gray, letras, box, claro, ht):
    """Altura de las minúsculas del original: en cada renglón, la banda de filas donde la
    tinta es densa (el cuerpo de las letras, sin ascendentes ni descendentes)."""
    if not ht:
        return None
    x, y, w, h = box
    H, W = gray.shape
    x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(W, x + w + 3), min(H, y + h + 3)
    zona = cv2.dilate(letras, np.ones((5, 5), np.uint8))[y0:y1, x0:x1] > 0
    g = gray[y0:y1, x0:x1]
    tinta = ((g > 150) if claro else (g < 110)) & zona
    perfil = tinta.sum(axis=1).astype(float)
    filas = perfil > 0
    runs, s = [], None
    for k, f in enumerate(list(filas) + [False]):
        if f and s is None:
            s = k
        elif not f and s is not None:
            runs.append((s, k)); s = None
    bandas = []
    for a, b in runs:
        if b - a < 0.5 * ht:
            continue
        p = perfil[a:b]
        bandas.append(int((p >= 0.45 * p.max()).sum()))
    return float(np.median(bandas)) if bandas else None


def medir_renglones(gray, letras, box, claro, ht):
    """(suma de anchos de renglón, nº de renglones) del texto original, medidos en la tinta.
    Sirve para saber qué tan condensada es la letra del original."""
    if not ht:
        return None, 0
    x, y, w, h = box
    H, W = gray.shape
    x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(W, x + w + 3), min(H, y + h + 3)
    zona = cv2.dilate(letras, np.ones((5, 5), np.uint8))[y0:y1, x0:x1] > 0
    g = gray[y0:y1, x0:x1]
    tinta = ((g > 150) if claro else (g < 110)) & zona
    filas = tinta.sum(axis=1) > 0
    runs, s = [], None
    for k, f in enumerate(list(filas) + [False]):
        if f and s is None:
            s = k
        elif not f and s is not None:
            runs.append([s, k]); s = None
    # unir trozos separados por huecos de 1 px (ruido) y descartar migajas (puntos, comas);
    # la rotulación de manga deja huecos de solo 3-5 px entre renglones
    unidos = []
    for r in runs:
        if unidos and r[0] - unidos[-1][1] <= 1:
            unidos[-1][1] = r[1]
        else:
            unidos.append(r)
    unidos = [r for r in unidos if r[1] - r[0] >= 0.5 * ht]
    if not unidos:
        return None, 0
    total = 0
    for a, b in unidos:
        cols = np.where(tinta[a:b].any(axis=0))[0]
        total += cols.max() - cols.min() + 1
    return float(total), len(unidos)


def _caja(m):
    ys, xs = np.where(m > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def analizar(img):
    """Lista de unidades: dict(id, box, globo, hg, mask_texto, region)."""
    H, W = img.shape[:2]
    gs = det_globos.detectar(img)
    cajas_ctd, seg = det_texto.detectar(img)
    txt = (seg > 90).astype(np.uint8) * 255
    if txt.sum() == 0:
        return [], gs
    # Fuera: trazos largos y delgados (contornos de globos de grito, líneas de efecto) que la
    # máscara marca como texto pero no lo son.
    hg_pag = _altura_glifo(txt)
    n, lab, st, _ = cv2.connectedComponentsWithStats(txt, 8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) > 3 * hg_pag and a < 0.12 * w * h:
            txt[lab == i] = 0

    # Globos fantasma: si la mayor parte de un globo cae dentro de otro más grande, es una
    # detección duplicada y partiría el texto del globo real.
    areas = [int(np.count_nonzero(g["mask"])) for g in gs]
    quitar = set()
    for i, gi in enumerate(gs):
        for j, gj in enumerate(gs):
            if i != j and areas[j] > areas[i] and j not in quitar:
                if np.count_nonzero((gi["mask"] > 0) & (gj["mask"] > 0)) > 0.5 * areas[i]:
                    quitar.add(i)
                    break
    gs = [g for i, g in enumerate(gs) if i not in quitar]

    gray_pag = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cada píxel de texto pertenece a lo sumo a un globo (el más chico que lo contenga).
    id_globo = np.full((H, W), -1, np.int32)
    for g in sorted(range(len(gs)), key=lambda j: -int(np.count_nonzero(gs[j]["mask"]))):
        id_globo[gs[g]["mask"] > 0] = g

    def grupos(letras_bin, limite=None):
        """Bloques de texto: primero renglones, luego renglones apilados y alineados.

        Dos frases sueltas dentro del mismo globo (p. ej. «WA-» arriba a la derecha y «WAIT!»
        abajo a la izquierda) quedan como bloques distintos porque no se alinean como un
        párrafo, aunque estén cerca."""
        letras_bin = _sin_dibujos(letras_bin, gray_pag)
        if not letras_bin.any():
            return
        hg = _altura_glifo(letras_bin)
        # glifos chicos (puntos, comas, marcas de sudor) se apartan para que no hagan de
        # puente entre frases; al final vuelven al bloque que los contenga
        ng0, lab0, sg0, cg0 = cv2.connectedComponentsWithStats(letras_bin, 8)
        chicos = [j for j in range(1, ng0) if sg0[j, 3] < 0.75 * hg and sg0[j, 2] < 0.9 * hg]
        principal = letras_bin.copy()
        if chicos:
            principal[np.isin(lab0, chicos)] = 0
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(hg * 0.85)), max(1, int(hg * 0.2))))
        dil = cv2.dilate(principal, k)
        if limite is not None:
            dil &= limite
        n, lab, st, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
        # altura real de letra de cada renglón (un párrafo apretado o un katakana gigante
        # pueden ser un solo componente muy alto; eso no debe agrandar la tolerancia)
        ng, _, sg, cg = cv2.connectedComponentsWithStats(principal, 8)
        alturas = {}
        for j in range(1, ng):
            if sg[j, 4] < 6:
                continue
            l = lab[int(cg[j, 1]), int(cg[j, 0])]
            if l == 0:
                ys, xs = np.where(_ := (lab[sg[j, 1]:sg[j, 1] + sg[j, 3], sg[j, 0]:sg[j, 0] + sg[j, 2]] > 0))
                if not len(ys):
                    continue
                l = lab[sg[j, 1] + ys[0], sg[j, 0] + xs[0]]
            alturas.setdefault(int(l), []).append(sg[j, 3])
        hl = {i: float(np.median(alturas[i])) if i in alturas else float(st[i, 3]) for i in range(1, n)}
        lineas = [i for i in range(1, n) if st[i, 4] >= 12]
        # marcas chicas (sudor, puntos, comillas sueltas) no sirven de puente entre bloques
        menudo = {i for i in lineas if hl[i] < 0.6 * hg and st[i, 2] < 1.5 * hg}
        padre = {i: i for i in lineas}

        def raiz(a):
            while padre[a] != a:
                padre[a] = padre[padre[a]]
                a = padre[a]
            return a

        for a in lineas:
            if a in menudo:
                continue
            xa, ya, wa, ha = st[a, :4]
            for b in lineas:
                if b == a or b in menudo:
                    continue
                xb, yb, wb, hb = st[b, :4]
                h = min(hl[a], hl[b])
                # misma fila (un renglón cortado al apartar una letra chica): unir si están cerca
                vert = min(ya + ha, yb + hb) - max(ya, yb)
                if vert >= 0.5 * min(ha, hb):
                    sep = max(xa, xb) - min(xa + wa, xb + wb)
                    un_renglon = ha <= 1.6 * hl[a] and hb <= 1.6 * hl[b]
                    if un_renglon and sep <= 1.2 * hg:
                        padre[raiz(a)] = raiz(b)
                    continue
                if yb < ya:            # b debajo de a
                    continue
                hueco = yb - (ya + ha)
                if hueco > 0.8 * h or hueco < -0.6 * h:
                    continue
                solape = min(xa + wa, xb + wb) - max(xa, xb)
                centros = abs((xa + wa / 2) - (xb + wb / 2))
                if solape >= 0.3 * min(wa, wb) or centros <= 0.3 * max(wa, wb):
                    padre[raiz(a)] = raiz(b)
        bloques = {}
        for i in lineas:
            bloques.setdefault(raiz(i), []).append(i)
        salida = []
        for miembros in bloques.values():
            lm = (np.isin(lab, miembros) & (principal > 0)).astype(np.uint8) * 255
            if np.count_nonzero(lm):
                salida.append([lm, _caja(lm)])
        # devolver puntos y comas al bloque cuya caja (algo ampliada) los contiene
        m = 0.35 * hg
        for j in chicos:
            cx, cy = cg0[j]
            for b in salida:
                x, y, w, h = b[1]
                if x - m <= cx <= x + w + m and y - m <= cy <= y + h + m:
                    b[0][lab0 == j] = 255
                    break
        for lm, _ in salida:
            if np.count_nonzero(lm) >= 25:
                yield lm

    unidades = []
    for gi, g in enumerate(gs):
        propio = (id_globo == gi).astype(np.uint8) * 255
        letras = txt & propio
        if np.count_nonzero(letras) < 25:
            continue
        for lm in grupos(letras, propio):
            unidades.append(dict(globo=gi, letras=lm))
    fuera = txt & ((id_globo < 0).astype(np.uint8) * 255)
    # Las cajas del detector de texto sirven de semilla: lo que cae dentro de una caja se
    # agrupa sin mezclarse con lo de afuera (onomatopeyas, líneas de efecto).
    for c in sorted(cajas_ctd, key=lambda c: c["box"][2] * c["box"][3]):
        if c["conf"] < 0.5:
            continue
        x, y, w, h = c["box"]
        m = max(3, int(0.08 * min(w, h)))
        dentro = np.zeros_like(fuera)
        dentro[max(0, y - m):y + h + m, max(0, x - m):x + w + m] = 255
        letras = fuera & dentro
        if np.count_nonzero(letras) < 0.6 * np.count_nonzero(txt & dentro & ((id_globo < 0).astype(np.uint8) * 255)) \
                or np.count_nonzero(letras) < 25:
            continue
        for lm in grupos(letras, dentro):
            unidades.append(dict(globo=-1, letras=lm))
        fuera = fuera & ~dentro
    for lm in grupos(fuera):
        unidades.append(dict(globo=-1, letras=lm))

    # Región disponible para escribir: dentro de un globo, la parte del globo más cercana a
    # este bloque (para globos pegados con dos textos); fuera, la caja del texto ampliada.
    for gi, g in enumerate(gs):
        miembros = [u for u in unidades if u["globo"] == gi]
        if not miembros:
            continue
        if len(miembros) == 1:
            miembros[0]["region"] = g["mask"]
            continue
        semillas = np.zeros((H, W), np.uint8)
        for j, u in enumerate(miembros, 1):
            semillas[u["letras"] > 0] = j
        dist, etiquetas = cv2.distanceTransformWithLabels((semillas == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                          labelType=cv2.DIST_LABEL_PIXEL)
        # etiquetas -> id de semilla
        mapa = np.zeros(etiquetas.max() + 1, np.int32)
        ys, xs = np.where(semillas > 0)
        mapa[etiquetas[ys, xs]] = semillas[ys, xs]
        dueno = mapa[etiquetas]
        for j, u in enumerate(miembros, 1):
            u["region"] = ((dueno == j) & (g["mask"] > 0)).astype(np.uint8) * 255

    # Fragmentos sueltos pegados a un bloque de globo (la máscara del globo es ondulada y a
    # veces deja fuera la última letra): se unen a ese bloque.
    for u in unidades:
        u["box"] = _caja(u["letras"])
        u["hg"] = _altura_glifo(u["letras"])
    # Sin encadenar: se compara con la caja ORIGINAL del bloque, y el fragmento tiene que
    # pegarse al borde del globo (no estar simplemente cerca de su caja).
    cajas0 = {id(b): list(b["box"]) for b in unidades}
    for f in [u for u in unidades if u["globo"] < 0]:
        fx, fy, fw, fh = f["box"]
        nf = np.count_nonzero(f["letras"])
        for b in unidades:
            if b["globo"] < 0 or b.get("absorbido"):
                continue
            bx, by, bw, bh = cajas0[id(b)]
            m = 0.8 * max(f["hg"], b["hg"])
            cerca = fx <= bx + bw + m and bx <= fx + fw + m and fy <= by + bh + m and by <= fy + fh + m
            parecidos = 0.6 <= f["hg"] / max(1.0, b["hg"]) <= 1.6
            chico = nf < 0.25 * np.count_nonzero(b["letras"])
            if not (cerca and parecidos and chico):
                continue
            k = max(5, int(1.3 * b["hg"]) | 1)
            borde = cv2.dilate(gs[b["globo"]]["mask"], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            if np.count_nonzero(f["letras"] & borde) >= 0.5 * nf:
                b["letras"] = b["letras"] | f["letras"]
                b["box"] = _caja(b["letras"])
                f["absorbido"] = True
                break
    unidades = [u for u in unidades if not u.get("absorbido")]

    # Trozos sueltos que caen dentro de la caja de otro bloque del mismo globo (una letra que
    # el agrupador dejó aparte) pertenecen a ese bloque.
    for f in sorted(unidades, key=lambda u: np.count_nonzero(u["letras"])):
        if f.get("absorbido") or f["globo"] < 0:
            continue
        fx, fy, fw, fh = f["box"]
        for b in unidades:
            if b is f or b.get("absorbido") or b["globo"] != f["globo"]:
                continue
            bx, by, bw, bh = b["box"]
            if (np.count_nonzero(f["letras"]) < 0.3 * np.count_nonzero(b["letras"])
                    and fx >= bx - 3 and fy >= by - 3 and fx + fw <= bx + bw + 3 and fy + fh <= by + bh + 3):
                b["letras"] = b["letras"] | f["letras"]
                f["absorbido"] = True
                break
    unidades = [u for u in unidades if not u.get("absorbido")]

    salida = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for u in unidades:
        hg_g = _altura_glifo(u["letras"])
        # renglones pegados sin hueco se verían como uno solo: no dejar que eso infle el tamaño
        u["hg"] = max(hg_g, min(0.85 * _altura_renglon(u["letras"]), 2.2 * hg_g))
        u["hg_glifo"] = hg_g
        claro_u, _ = _color_texto(gray, u["letras"])
        u["hg_tinta"] = altura_tinta(gray, u["letras"], u["box"], claro_u)
        u["anchos"], u["renglones"] = medir_renglones(gray, u["letras"], u["box"], claro_u,
                                                       u["hg_tinta"] or hg_g)
        u["xh_tinta"] = altura_x_tinta(gray, u["letras"], u["box"], claro_u, u["hg_tinta"] or hg_g)
    g32 = gray.astype(np.float32)
    media = cv2.blur(g32, (9, 9))
    textura = np.sqrt(np.maximum(cv2.blur(g32 * g32, (9, 9)) - media * media, 0))
    for u in unidades:
        if "region" not in u:
            # texto suelto: su caja ampliada y, además, el fondo liso que la rodea (sin cruzar
            # líneas de viñeta ni dibujo), sin invadir la de otros bloques
            x, y, w, h = u["box"]
            gx, gy = int(0.12 * w + 4), int(0.12 * h + 4)
            reg = np.zeros((H, W), np.uint8)
            reg[max(0, y - gy):min(H, y + h + gy), max(0, x - gx):min(W, x + w + gx)] = 255
            anillo = cv2.dilate(u["letras"], np.ones((15, 15), np.uint8)) & ~cv2.dilate(u["letras"], np.ones((5, 5), np.uint8))
            fondo = float(np.median(gray[anillo > 0])) if anillo.any() else 255.0
            ex, ey = int(0.45 * w + u["hg"]), int(0.2 * h + u["hg"])
            x0, y0 = max(0, x - ex), max(0, y - ey)
            x1, y1 = min(W, x + w + ex), min(H, y + h + ey)
            liso = ((np.abs(g32[y0:y1, x0:x1] - fondo) < 28) & (textura[y0:y1, x0:x1] < 14)).astype(np.uint8)
            liso[(y - y0):(y - y0 + h), (x - x0):(x - x0 + w)] = 1     # el texto original cuenta como libre
            liso = cv2.erode(liso, np.ones((5, 5), np.uint8))
            n, lab = cv2.connectedComponents(liso, connectivity=4)
            centro = lab[min(lab.shape[0] - 1, y - y0 + h // 2), min(lab.shape[1] - 1, x - x0 + w // 2)]
            if centro > 0:
                reg[y0:y1, x0:x1] |= (lab == centro).astype(np.uint8) * 255
            # el espacio de los demás bloques se descuenta al componer, cuando ya se sabe
            # cuáles se traducen (ver _recortar_libres)
            u["region"] = reg
        salida.append(u)
    # orden de lectura de manga: de arriba abajo, de derecha a izquierda
    salida.sort(key=lambda u: (u["box"][1] // 60, -(u["box"][0] + u["box"][2])))
    for i, u in enumerate(salida):
        u["id"] = i
    return salida, gs


def etiquetar(img, unidades, ancho_max=1400):
    """Imagen con cada unidad recuadrada y numerada, para que el LLM la lea."""
    vis = img.copy()
    for u in unidades:
        x, y, w, h = u["box"]
        cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 0, 255), 2)
        t = str(u["id"])
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y - 6 if y - 6 - th > 0 else y + h + th + 6
        cv2.rectangle(vis, (x - 3, ty - th - 3), (x - 3 + tw + 6, ty + 3), (0, 0, 255), -1)
        cv2.putText(vis, t, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if vis.shape[1] < ancho_max:   # páginas chicas: agrandar para que el texto se lea bien
        s = min(2.0, ancho_max / vis.shape[1])
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return vis


# ----------------------------------------------------------------------------- borrado

_lama = None


def _inpaint_lama(img, mascara):
    """LaMa sobre un recorte cuadrado alrededor de la máscara (el modelo es 512x512 fijo)."""
    global _lama
    if _lama is None:
        _lama = crear_sesion(LAMA)
    H, W = img.shape[:2]
    x, y, w, h = _caja(mascara)
    lado = int(max(w, h) * 1.6 + 32)
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(0, cx - lado // 2), max(0, cy - lado // 2)
    x1, y1 = min(W, x0 + lado), min(H, y0 + lado)
    x0, y0 = max(0, x1 - lado), max(0, y1 - lado)
    rec, mrec = img[y0:y1, x0:x1], mascara[y0:y1, x0:x1]
    rh, rw = rec.shape[:2]
    a = cv2.resize(rec, (512, 512), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mrec, (512, 512), interpolation=cv2.INTER_NEAREST)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    m = (m > 0).astype(np.float32)[None, None]
    out = _lama.run(None, {"image": a, "mask": m})[0][0].transpose(1, 2, 0)
    if out.max() <= 1.5:
        out = out * 255
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_CUBIC)
    sel = mrec > 0
    img[y0:y1, x0:x1][sel] = out[sel]


def _color_texto(gray, letras):
    """(claro, fondo): si las letras son claras sobre fondo oscuro, y el gris del fondo.

    La máscara de texto es borrosa e incluye halo de fondo, así que en vez de la mediana se
    mira cuál extremo de los píxeles de letra (claro u oscuro) se aleja más del fondo."""
    anillo = cv2.dilate(letras, np.ones((11, 11), np.uint8)) & ~cv2.dilate(letras, np.ones((5, 5), np.uint8))
    lb = float(np.median(gray[anillo > 0])) if anillo.any() else 255.0
    v = gray[letras > 0]
    if not len(v):
        return False, lb
    lo, hi = np.percentile(v, 12), np.percentile(v, 88)
    return (hi - lb) > (lb - lo), lb


def borrar(img, u, globo, claro=False):
    """Borra el inglés de la unidad u sobre img (BGR, in-place)."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    letras = u["letras"]
    k = max(3, int(u["hg"] * 0.25) | 1)
    mas = cv2.dilate(letras, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Trazos que la máscara del modelo dejó a medias: componentes del color de la letra que
    # ya están mayormente cubiertos por la máscara se borran completos.
    x, y, w, h = u["box"]
    pad = int(u["hg"] * 0.6) + 4
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    zona = gray[y0:y1, x0:x1]
    cand = ((zona > 170) if claro else (zona < 90)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    mz = mas[y0:y1, x0:x1] > 0
    extra = np.zeros_like(cand)
    for i in range(1, n):
        if st[i, 4] > 0.25 * cand.size:
            continue
        comp = lab == i
        if (comp & mz).sum() >= 0.35 * st[i, 4]:
            extra[comp] = 1
    if extra.any():
        mas[y0:y1, x0:x1] |= cv2.dilate(extra, np.ones((3, 3), np.uint8)) * 255

    if globo is not None:
        interior = cv2.erode(globo["mask"], np.ones((7, 7), np.uint8))
        # el fondo del globo se mide cerca del texto (la máscara del globo a veces se come
        # trama o dibujo del borde y eso no dice nada del fondo real)
        x, y, w, h = u["box"]
        pad = int(u["hg"] * 0.8) + 4
        zona = np.zeros_like(mas)
        zona[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = 255
        fondo = interior & zona & ~cv2.dilate(mas, np.ones((5, 5), np.uint8))
        if np.count_nonzero(fondo) > 150:
            gf = gray[fondo > 0].astype(int)
            med = int(np.median(gf))
            # relleno plano solo si el fondo es de verdad uniforme (blanco o color liso);
            # una trama clara o un degradado se reconstruye con LaMa
            if np.mean(np.abs(gf - med) < 12) > 0.92:
                color = np.median(img[fondo > 0], axis=0).astype(np.uint8)
                oscuro = ((np.abs(gray.astype(int) - med) > 60).astype(np.uint8) * 255) & zona & interior
                total = (mas | cv2.dilate(oscuro, np.ones((3, 3), np.uint8))) & interior
                img[total > 0] = color
                # letras fuera del interior detectado (la máscara del globo es ondulada): si lo
                # que las rodea es del color del globo, también van planas; si no, LaMa
                resto = mas & ~interior
                if np.count_nonzero(resto) > 20:
                    n, lab, st, _ = cv2.connectedComponentsWithStats(resto, 8)
                    lama = np.zeros_like(resto)
                    for i in range(1, n):
                        comp = (lab == i).astype(np.uint8) * 255
                        anillo = cv2.dilate(comp, np.ones((9, 9), np.uint8)) & ~cv2.dilate(comp, np.ones((3, 3), np.uint8)) & ~mas
                        va = gray[anillo > 0].astype(int)
                        if len(va) and np.mean(np.abs(va - med) < 25) > 0.7:
                            img[comp > 0] = color
                        else:
                            lama |= comp
                    if np.count_nonzero(lama) > 20:
                        _inpaint_lama(img, cv2.dilate(lama, np.ones((3, 3), np.uint8)))
                return med          # globo liso: devuelve el gris del fondo
        _inpaint_lama(img, _con_halo(gray, mas, u, claro))
        return None
    _inpaint_lama(img, _con_halo(gray, mas, u, claro))
    return None


def _con_halo(gray, mas, u, claro):
    """Para reconstruir con LaMa, la máscara debe cubrir también el contorno de las letras
    (letra negra con borde blanco, o al revés) y el suavizado; si no, queda un fantasma."""
    k = max(5, int(u["hg"] * 0.3) | 1)
    amplia = cv2.dilate(mas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # píxeles del tono contrario a la letra pegados a ella = contorno: también se cubren
    anillo = amplia & ~mas
    tono = (gray < 90) if claro else (gray > 200)
    halo = (anillo > 0) & tono
    if halo.any():
        amplia = mas | cv2.dilate(halo.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        amplia = cv2.dilate(amplia, np.ones((3, 3), np.uint8))
    return amplia


def _interior_real(gray, u, globo, fondo, vecinos, otros_globos=()):
    """Forma real del interior del globo, medida en la imagen ya borrada.

    La máscara del detector es ondulada y suele quedar más chica que el globo; después de
    borrar el inglés, el interior es una mancha lisa del color de fondo que se puede medir
    tal cual. Se descarta si se «escapa» (globo abierto que se funde con el fondo de página)."""
    H, W = gray.shape
    x, y, w, h = globo["box"]
    pad = 14
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    reg = u["region"][y0:y1, x0:x1] > 0
    liso = (np.abs(gray[y0:y1, x0:x1].astype(int) - int(fondo)) < 30).astype(np.uint8)
    n, lab = cv2.connectedComponents(liso, connectivity=4)
    ids, cnt = np.unique(lab[reg & (liso > 0)], return_counts=True)
    sel = [(c, i) for i, c in zip(ids, cnt) if i != 0]
    if not sel:
        return None
    comp = lab == max(sel)[1]
    if comp.sum() > 1.8 * reg.sum() or comp.sum() < 0.6 * reg.sum():
        return None
    # tapar muescas y huecos chicos (restos de trama, marcas sueltas en el borde)
    k = max(9, int(0.06 * min(w, h)) | 1)
    comp = cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    comp = _fill_holes(comp.astype(np.uint8) * 255) > 0
    # dentro de lo que el detector llamó globo (con holgura) y sin invadir a los vecinos
    holgura = cv2.dilate(reg.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    nuevo = comp & holgura
    for v in vecinos:
        nuevo &= ~(v["region"][y0:y1, x0:x1] > 0)
    # globos pegados o encimados: nunca escribir dentro del vecino (ni sobre su contorno)
    for m in otros_globos:
        nuevo &= ~(cv2.dilate(m[y0:y1, x0:x1], np.ones((9, 9), np.uint8)) > 0)
    nuevo = cv2.erode(nuevo.astype(np.uint8), np.ones((3, 3), np.uint8))
    out = np.zeros((H, W), np.uint8)
    out[y0:y1, x0:x1] = nuevo * 255
    return out


# ----------------------------------------------------------------------------- composición

_fuentes = {}
SIMBOLOS = set("♥♡❤♪♫♬☆★✦✧☀☺♀♂")
FUENTE_SIMBOLOS = r"C:\Windows\Fonts\seguisym.ttf"


# Fuente por estilo de globo: (archivo, variante, paso). El traductor marca el estilo de cada
# bloque según cómo está rotulado el original. «paso» es la distancia entre renglones medida
# en alturas de mayúscula: la rotulación de manga va apretada (~1.25-1.35); un poco más en
# manuscrito, que lleva minúsculas con rasgos descendentes.
ESTILOS = {
    "normal": ("ComicRelief-Bold.ttf", None, 1.34),
    "grito": ("Bangers-Regular.ttf", None, 1.18),
    "manuscrito": ("GochiHand-Regular.ttf", None, 1.5),
    "rotulo": ("Oswaldwght.ttf", "SemiBold", 1.32),   # carteles, títulos y respaldo
    "cursiva": ("ComicRelief-Bold.ttf", None, 1.34),  # mayúsculas inclinadas (se dibujan con sesgo)
}
SESGO = 0.18          # inclinación de la cursiva (desplazamiento horizontal por unidad de alto)


def fuente(tam, estilo="normal"):
    estilo = estilo if estilo in ESTILOS else "normal"
    k = (estilo, tam)
    if k not in _fuentes:
        archivo, variante, _ = ESTILOS[estilo]
        f = ImageFont.truetype(os.path.join(AQUI, "fonts", archivo), tam)
        if variante:
            try:
                f.set_variation_by_name(variante)
            except Exception:
                pass
        f.estilo = estilo
        _fuentes[k] = f
    return _fuentes[k]


_proporcion = {}


def _metrica(estilo):
    """(altura de mayúscula, distancia del origen de dibujo a la parte alta de la «H»),
    ambas en proporción al tamaño de la fuente."""
    if estilo not in _proporcion:
        f = fuente(100, estilo)
        x0, y0, x1, y1 = f.getbbox("H")
        _proporcion[estilo] = ((y1 - y0) / 100.0, y0 / 100.0)
    return _proporcion[estilo]


def altura_mayuscula(estilo):
    """Altura de la «H» en proporción al tamaño de la fuente (cada fuente mide distinto)."""
    return _metrica(estilo)[0]


_proporcion_x = {}


def altura_x(estilo):
    """Altura de la «x» (minúsculas) en proporción al tamaño de la fuente."""
    if estilo not in _proporcion_x:
        f = fuente(100, estilo)
        x0, y0, x1, y1 = f.getbbox("x")
        _proporcion_x[estilo] = (y1 - y0) / 100.0
    return _proporcion_x[estilo]


def usa_minusculas(estilo):
    """El manuscrito va en minúsculas: su tamaño se iguala por la altura de la «x»."""
    return estilo == "manuscrito"


def paso_renglon(estilo):
    """Distancia entre renglones en alturas de mayúscula."""
    return ESTILOS.get(estilo, ESTILOS["normal"])[2]


def interlineado(estilo):
    """Distancia entre renglones en proporción al tamaño de la fuente."""
    return paso_renglon(estilo) * altura_mayuscula(estilo)


def _fuente_sim(tam):
    k = ("sim", tam)
    if k not in _fuentes:
        _fuentes[k] = ImageFont.truetype(FUENTE_SIMBOLOS, tam) if os.path.exists(FUENTE_SIMBOLOS) else fuente(tam)
    return _fuentes[k]


def _tramos(texto, f):
    """Parte el texto en tramos (cadena, fuente): los símbolos que Oswald no tiene van aparte."""
    out, cur, sim = [], "", None
    for c in texto:
        s = c in SIMBOLOS
        if sim is not None and s != sim:
            out.append((cur, _fuente_sim(f.size) if sim else f))
            cur = ""
        cur += c
        sim = s
    if cur:
        out.append((cur, _fuente_sim(f.size) if sim else f))
    return out


def largo(d, texto, font):
    return sum(d.textlength(t, font=fu) for t, fu in _tramos(texto, font))


def dibujar(d, xy, texto, font, **kw):
    x, y = xy
    for t, fu in _tramos(texto, font):
        d.text((x, y), t, font=fu, **kw)
        x += d.textlength(t, font=fu)


def _ancho_en(region, y, cx, margen):
    """Ancho libre de la región en la fila y, alrededor de cx."""
    H, W = region.shape
    y = int(np.clip(y, 0, H - 1))
    fila = region[y] > 0
    cx = int(np.clip(cx, 0, W - 1))
    if not fila[cx]:
        return 0, cx, cx
    i = cx
    while i > 0 and fila[i - 1]:
        i -= 1
    j = cx
    while j < W - 1 and fila[j + 1]:
        j += 1
    return max(0, (j - i) - 2 * margen), i + margen, j - margen


VOCALES = set("AEIOUÁÉÍÓÚÜaeiouáéíóúü")
GRUPOS = {"BL", "BR", "CL", "CR", "DR", "FL", "FR", "GL", "GR", "KL", "KR", "PL", "PR", "TR", "CH", "LL", "RR"}


def silabas_cortes(palabra, minimo=9, lado=4):
    """Posiciones donde se puede partir una palabra española con guion (aproximado).

    Por defecto solo palabras de 9+ letras y 4+ letras a cada lado (¿TER-NURA? no). Los
    gritos admiten cortes más cortos, como el original («HASSE-GAWA», «SUBA-RU»)."""
    p = palabra.upper()
    letras = [i for i, c in enumerate(p) if c.isalpha()]
    if len(letras) < minimo:
        return []
    cortes = []
    i = 1
    while i < len(p) - 2:
        a, b = p[i - 1], p[i]
        if a in VOCALES and b.isalpha() and b not in VOCALES:
            # contar consonantes seguidas
            j = i
            while j < len(p) and p[j].isalpha() and p[j] not in VOCALES:
                j += 1
            if j < len(p) and p[j] in VOCALES:
                n = j - i
                if n == 1:
                    corte = i
                elif n == 2:
                    corte = i if p[i:i + 2] in GRUPOS else i + 1
                else:
                    corte = j - 2 if p[j - 2:j] in GRUPOS else j - 1
                cortes.append(corte)
            i = j
        else:
            i += 1
    def n_letras(s):
        return sum(ch.isalpha() for ch in s)
    return [c for c in cortes if n_letras(p[:c]) >= lado and n_letras(p[c:]) >= lado]


def cortes_palabra(p, estilo, guion):
    """Dónde se puede cortar una palabra al final de renglón: [(posición, añadir_guion)].

    En un guion que ya existe («MARIN-CHAN», «GOJO-KUN») siempre se puede cortar sin añadir
    nada. Por sílaba solo si se permite guion: 8+ letras y 3+ a cada lado; en gritos, desde
    6 letras, como el original («HASSE-GAWA»)."""
    out = [(i + 1, False) for i, ch in enumerate(p) if ch == "-" and 0 < i < len(p) - 1]
    if guion:
        out += [(c, True) for c in (silabas_cortes(p, 6, 2) if estilo == "grito" else silabas_cortes(p, 9, 4))]
    return sorted(set(out))


def _con_puntos(texto, estilo):
    """«...» -> «…» si la fuente lo tiene: un solo carácter en vez de tres."""
    if "..." not in texto:
        return texto
    f = fuente(40, estilo)
    try:
        if f.getmask("…").getbbox() is None:
            return texto
    except Exception:
        return texto
    return texto.replace("...", "…")


NBSP = " "


def _atar_cortas(texto):
    """Une con espacio duro las palabras de 1-2 letras a la siguiente («EL VESTIDOR»,
    «A PROBLEMA», «¡UN NOMBRE»), para que nunca queden solas al final de un renglón."""
    ps = texto.split()
    out = []
    i = 0
    while i < len(ps):
        p = ps[i]
        while (i + 1 < len(ps) and sum(ch.isalpha() for ch in p.split(NBSP)[-1]) <= 2
               and not p.endswith("-") and not p.endswith(("...", "…", "?", "!", ","))):
            i += 1
            p = p + NBSP + ps[i]
        out.append(p)
        i += 1
    return " ".join(out)


def _maquetar(d, texto, tam, region, cx, cy, top, bottom, margen, guion=False, estilo="normal", trazo=0,
              cond=1.0, eje_fijo=False):
    """Reparte las palabras en renglones que caben en la forma del globo. None si no cabe.

    Todo se mide sobre la caja real de las mayúsculas (no sobre el tamaño nominal de la
    fuente): el bloque va de la parte alta del primer renglón a la línea base del último,
    con un margen arriba para las tildes de las mayúsculas (Á, É...)."""
    f = fuente(tam, estilo)
    c_rel, t_rel = _metrica(estilo)
    cap = c_rel * tam
    paso = cap * paso_renglon(estilo)
    tilde = 0.3 * cap
    margen = margen + trazo

    extra = SESGO * tam if estilo == "cursiva" else 0.0

    def largo_c(t):              # ancho ya condensado (y con el sesgo de la cursiva)
        return largo(d, t, f) * cond + extra

    palabras = [p for p in texto.split(" ") if p.strip()]
    for n_lineas in range(1, 2 * len(palabras) + 1):
        alto = (n_lineas - 1) * paso + cap
        y0 = cy - alto / 2 + tilde / 2           # parte alta de la mayúscula del 1.er renglón
        if y0 - tilde < top - 1 or y0 + alto > bottom + 1:
            return None
        anchos, centros = [], []
        for k in range(n_lineas):
            ca = y0 + k * paso
            _, ia, ja = _ancho_en(region, ca + 0.1 * cap, cx, margen)
            _, ib, jb = _ancho_en(region, ca + 0.9 * cap, cx, margen)
            i, j = max(ia, ib), min(ja, jb)      # tramo libre en todo el alto del renglón
            if eje_fijo:                         # todos los renglones centrados en cx
                media = max(0.0, min(cx - i, j - cx))
                anchos.append(2 * media)
                centros.append(cx)
            else:
                anchos.append(max(0, j - i))
                centros.append((i + j) / 2 if j > i else cx)
        lineas, cur, k = [], "", 0
        ok = True
        pend = list(palabras)
        while pend:
            p = pend.pop(0)
            prueba = (cur + " " + p).strip()
            if largo_c(prueba) <= anchos[k]:
                cur = prueba
                continue
            # partir la palabra (en su guion propio o por sílaba) si algo cabe en este renglón
            partido = False
            for c, añadir in reversed(cortes_palabra(p, estilo, guion)):
                t = (cur + " " + p[:c] + ("-" if añadir else "")).strip()
                if largo_c(t) <= anchos[k]:
                    cur = t
                    pend.insert(0, p[c:])
                    partido = True
                    break
            if partido:
                lineas.append(cur); cur = ""; k += 1
                if k >= n_lineas:
                    ok = False
                    break
                continue
            if not cur:
                ok = False
                break
            lineas.append(cur)
            k += 1
            cur = ""
            pend.insert(0, p)
            if k >= n_lineas:
                ok = False
                break
        if ok and cur:
            lineas.append(cur)
        if ok and len(lineas) <= n_lineas:
            lineas = [l.replace(NBSP, " ") for l in lineas]
            # cada renglón va centrado en su tramo libre (sigue la forma del globo); se
            # devuelve el origen de dibujo de PIL, que está por encima de la mayúscula
            return f, lineas, paso, y0 - t_rel * tam, centros[:len(lineas)], cond
    return None


def _con_trazo(globo, claro):
    """Texto suelto (fuera de globo) o claro sobre oscuro lleva contorno para leerse."""
    return not (globo is not None and not claro)


def _mejor_tam(d, es, region, cx, cy, top, bottom, margen, estilo, tope, minimo, trazo_rel, cond=1.0,
               eje_fijo=False):
    """Mayor tamaño (<= tope) que cabe. Preferencias, de mejor a peor: palabras cortas
    atadas y sin guion; luego permitir una palabra corta sola en su renglón; luego
    condensar un 10 % más; luego partir por sílaba. Cada opción menos elegante solo gana si
    da letra notablemente mayor."""
    def mejor(texto, guion, c):
        for tam in range(int(tope), minimo - 1, -1):
            tr = max(2, tam // 7) if trazo_rel else 0
            p = _maquetar(d, texto, tam, region, cx, cy, top, bottom, margen, guion, estilo, tr, c, eje_fijo)
            if p:
                return p
        return None
    atado = _atar_cortas(es)
    plan = mejor(atado, False, cond)
    extra = max(0.62, cond * 0.9)
    for texto, guion, c, ganancia in ((es, False, cond, 1.12), (atado, False, extra, 1.08),
                                      (es, True, cond, 1.15), (es, True, extra, 1.2)):
        if plan is not None and plan[0].size >= tope - 1:
            break
        otro = mejor(texto, guion, c)
        if otro and (plan is None or otro[0].size >= ganancia * plan[0].size):
            plan = otro
    return plan


def condensacion_original(d, u, en, estilo):
    """Factor de ancho (<= 1) para que nuestra letra tenga la misma proporción ancho/alto
    que la del original. Se compara el inglés tal como está rotulado (suma de anchos de sus
    renglones) con lo que mediría ese mismo inglés en nuestra fuente a igual altura."""
    anchos, n = u.get("anchos"), u.get("renglones") or 0
    ht = u.get("hg_tinta")
    en = " ".join((en or "").split())
    if not (anchos and n and ht and len(en) >= 4):
        return 1.0
    # renglones fundidos (texto inclinado o muy apretado) dan una medida falsa
    esperado = u["box"][3] / (1.3 * ht)
    if n < 0.6 * esperado:
        return 1.0
    f = fuente(100, estilo)
    xh = u.get("xh_tinta")
    if usa_minusculas(estilo) and xh and 0.35 * ht <= xh <= 0.95 * ht:
        ref_nuestra, ref_suya = altura_x(estilo) * 100, xh
    else:
        ref_nuestra, ref_suya = altura_mayuscula(estilo) * 100, ht
    nuestro = (largo(d, en, f) - (n - 1) * largo(d, " ", f)) / ref_nuestra
    suyo = anchos / ref_suya
    if nuestro <= 0:
        return 1.0
    return float(np.clip(suyo / nuestro, 0.7, 1.0))


def _tam_por_caja(d, es, u, estilo, desde, trazo_rel, cond=1.0):
    """Tamaño con el que el texto traducido llena la caja que ocupaba el original.

    Así un grito corto («¡¿EEEEH?!») ocupa el mismo espacio que el original aunque la
    altura de letra medida no sea fiable (letras con contorno, cursivas, deformadas)."""
    H, W = u["region"].shape
    x, y, w, h = u["box"]
    ex, ey = int(0.08 * w) + 2, int(0.08 * h) + 2
    caja = np.zeros((H, W), np.uint8)
    caja[max(0, y - ey):min(H, y + h + ey), max(0, x - ex):min(W, x + w + ex)] = 255
    arriba, abajo = max(0, y - ey), min(H, y + h + ey)
    plan = _mejor_tam(d, es, caja, x + w / 2, y + h / 2, arriba, abajo, 0, estilo, desde, 9, trazo_rel, cond)
    return plan[0].size if plan else 0


def planear(d, u, es, globo, estilo="normal", claro=False, tope_fijo=None, en="", escala=1.0):
    """Decide tamaño, renglones y posición. No dibuja. `escala` viene de la revisión manual."""
    cond = condensacion_original(d, u, en, estilo)
    es = _con_puntos(" ".join(es.split()), estilo)      # saltos y espacios dobles -> un espacio
    region = u["region"]
    x, y, w, h = _caja(region)
    tx, ty, tw, th = u["box"]
    cx = tx + tw / 2
    # Globo de un solo bloque: centrado en el globo. Globo compartido por varias frases
    # (WAIT! ... WA-): cada frase se queda donde estaba la original.
    compartido = globo is not None and u.get("compartido", False)
    cy = y + h / 2 if globo is not None and not compartido else ty + th / 2
    # si el centro del texto original cae fuera de la región, usar el centro de la región
    if region[int(np.clip(cy, 0, region.shape[0] - 1)), int(np.clip(cx, 0, region.shape[1] - 1))] == 0:
        cx, cy = x + w / 2, y + h / 2
    margen = max(3, int((0.04 if u.get("interior_real") else 0.055) * w)) if globo is not None else 2
    trazo = _con_trazo(globo, claro)

    # Tope de tamaño: lo mayor entre (a) la altura de letra del original y (b) el tamaño con
    # el que el español llena la caja del original. En diálogo (b) se limita a +30 % para que
    # un texto corto no quede gigante; en gritos manda la caja.
    hg = u["hg"]
    ht, hgl = u.get("hg_tinta"), u.get("hg_glifo") or u["hg"]
    fiable = bool(ht and 0.5 * hgl <= ht <= 3 * hgl)
    if fiable:                                   # medición en tinta: la más fiable si es coherente
        hg = ht
    tope_letra = max(12, hg / altura_mayuscula(estilo))
    xh = u.get("xh_tinta")
    if usa_minusculas(estilo) and xh and 0.35 * hg <= xh <= 0.95 * hg:
        # manuscrito: igualar la altura de las minúsculas, que es lo que el ojo compara
        tope_letra = max(12, xh / altura_x(estilo))
    tope_caja = _tam_por_caja(d, es, u, estilo, int(tope_letra * (3 if estilo == "grito" else 1.6)), trazo, cond)
    if estilo == "grito":
        # la caja del original incluye inclinación, signos y aire: llenarla al 100 % deja el
        # grito más grande que el original. Si la altura de letra es fiable, manda (+35 %).
        tope = max(tope_letra, min(tope_caja, 1.35 * tope_letra)) if fiable else max(tope_letra, 0.8 * tope_caja)
    else:
        tope = max(tope_letra, min(tope_caja, 1.15 * tope_letra))
    if tope_fijo:
        tope = min(tope, tope_fijo)
    tope = max(9, tope * escala)
    plan = _mejor_tam(d, es, region, cx, cy, y + 2, y + h - 2, margen, estilo, int(tope), 9, trazo, cond,
                      eje_fijo=globo is None)
    # Texto vertical (letras apiladas, p. ej. «W/a/a/a/h~» a lo largo de un borde): si el
    # bloque original es alto y angosto y el texto es corto, apilar las letras igual.
    vertical = False
    if th > 1.8 * tw and len(es.replace(" ", "")) <= 16:
        vp = _plan_vertical(d, es, region, cx, ty + th / 2, y, y + h, tw, estilo, int(tope))
        if vp and (plan is None or vp[0].size >= 1.15 * plan[0].size):
            plan, vertical = vp, True
    if plan is None:   # no cabe: renglones por ancho de caja, letra mínima
        f = fuente(9, estilo)
        lineas, cur = [], ""
        for p in es.split():
            t = (cur + " " + p).strip()
            if largo(d, t, f) <= max(w, tw) or not cur:
                cur = t
            else:
                lineas.append(cur); cur = p
        lineas.append(cur)
        plan = (f, lineas, 10.0, cy - len(lineas) * 5, [cx] * len(lineas), 1.0)
    return dict(plan=plan, tam=plan[0].size, tope=int(tope), tope_letra=tope_letra, tope_caja=tope_caja,
                estilo=estilo, lineas=plan[1], mayus=plan[0].size * altura_mayuscula(estilo),
                cond=plan[5], cond_original=cond, vertical=vertical)


def _plan_vertical(d, es, region, cx, cy, top, bottom, ancho_orig, estilo, tope):
    """Una letra por renglón, centradas en cx, dentro de la región y del alto disponible.
    Siempre en mayúsculas: apiladas, las minúsculas se confunden (j/i, l/1)."""
    letras = [c for c in es.upper() if c != " "]
    if not letras:
        return None
    c_rel, t_rel = _metrica(estilo)
    for tam in range(int(tope), 8, -1):
        f = fuente(tam, estilo)
        cap = c_rel * tam
        paso = cap * 1.12
        alto = (len(letras) - 1) * paso + cap
        y0 = cy - alto / 2
        y0 = min(max(y0, top + 2), bottom - 2 - alto)
        if y0 < top + 1 or alto > bottom - top - 2:
            continue
        ancho = max(largo(d, c, f) for c in letras)
        if ancho > 1.6 * ancho_orig:
            continue
        ok = all(_ancho_en(region, y0 + k * paso + 0.5 * cap, cx, 0)[0] >= ancho for k in range(len(letras)))
        if ok:
            return f, letras, paso, y0 - t_rel * tam, [cx] * len(letras), 1.0
    return None


def dibujar_plan(pil, d, pl, claro, globo):
    """Dibuja los renglones. Si hay condensación, cada renglón se dibuja aparte a ancho
    completo y se estrecha en horizontal (como hace un rotulista con la escala horizontal)."""
    f, lineas, paso, y0, centros, cond = pl["plan"]
    tinta = (255, 255, 255) if claro else (0, 0, 0)
    borde = (0, 0, 0) if claro else (255, 255, 255)
    trazo = max(2, f.size // 7) if _con_trazo(globo, claro) else 0
    sesgo = SESGO if pl.get("estilo") == "cursiva" else 0.0
    W = pil.size[0]
    m = 4 + trazo                     # margen al borde de la página (incluye el contorno)
    for ln, c in zip(lineas, centros):
        lw = largo(d, ln, f)
        if cond >= 0.995 and not sesgo:
            px = min(max(c - lw / 2, m), W - lw - m)
            dibujar(d, (px, y0), ln, f, fill=tinta, stroke_width=trazo, stroke_fill=borde)
        else:
            # renglón aparte: se estrecha (condensación) y/o se inclina (cursiva)
            pad = trazo + 4
            alto = int(f.size * 1.6) + 2 * pad
            tmp = Image.new("RGBA", (int(lw) + 2 * pad, alto), (0, 0, 0, 0))
            dibujar(ImageDraw.Draw(tmp), (pad, pad), ln, f, fill=tinta + (255,), stroke_width=trazo,
                    stroke_fill=borde + (255,))
            nw = max(1, int(round(tmp.size[0] * cond)))
            tmp = tmp.resize((nw, alto), Image.LANCZOS)
            extra = 0
            if sesgo:
                extra = int(round(sesgo * alto))
                # x_entrada = x + s·y − s·alto: la parte alta del renglón queda a la derecha
                tmp = tmp.transform((nw + extra, alto), Image.AFFINE, (1, sesgo, -sesgo * alto, 0, 1, 0),
                                    resample=Image.BICUBIC)
            lwc = lw * cond + extra
            px = min(max(c - lwc / 2, m), W - lwc - m)
            pil.paste(tmp, (int(round(px - pad * cond)), int(round(y0 - pad))), tmp)
        y0 += paso


def escribir(pil, d, u, es, claro, globo, estilo="normal", en=""):
    pl = planear(d, u, es, globo, estilo, claro, en=en)
    dibujar_plan(pil, d, pl, claro, globo)
    return pl


def _repartir_globos(unidades, gs, activos):
    """Reparte cada globo solo entre los bloques que se van a traducir.

    Lo que se queda como arte (onomatopeyas, signos sueltos) no se lleva media burbuja: solo
    se le reserva su propio espacio para no escribir encima."""
    H, W = gs[0]["mask"].shape if gs else (0, 0)
    for gi, g in enumerate(gs):
        miembros = [u for u in unidades if u["globo"] == gi]
        vivos = [u for u in miembros if u["id"] in activos]
        if not vivos or len(vivos) == len(miembros):
            continue
        libre = g["mask"].copy()
        for u in miembros:
            if u["id"] not in activos:
                x, y, w, h = u["box"]
                s = max(4, int(0.3 * u["hg"]))
                libre[max(0, y - s):y + h + s, max(0, x - s):x + w + s] = 0
        if len(vivos) == 1:
            vivos[0]["region"] = libre
            continue
        semillas = np.zeros((H, W), np.uint8)
        for j, u in enumerate(vivos, 1):
            semillas[u["letras"] > 0] = j
        _, etiquetas = cv2.distanceTransformWithLabels((semillas == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
        mapa = np.zeros(etiquetas.max() + 1, np.int32)
        ys, xs = np.where(semillas > 0)
        mapa[etiquetas[ys, xs]] = semillas[ys, xs]
        dueno = mapa[etiquetas]
        for j, u in enumerate(vivos, 1):
            u["region"] = ((dueno == j) & (libre > 0)).astype(np.uint8) * 255


def _lineas_vineta(gray):
    """Máscara de los bordes de viñeta: trazos oscuros, rectos y largos."""
    H, W = gray.shape
    osc = (gray < 70).astype(np.uint8) * 255
    hor = cv2.morphologyEx(osc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, W // 6), 1)))
    ver = cv2.morphologyEx(osc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, H // 6))))
    return cv2.dilate(hor | ver, np.ones((5, 5), np.uint8))


def _recortar_libres(unidades, activos, gray=None):
    """Texto suelto: no invadir otros bloques traducidos, ni arte que quede fuera del propio
    texto, ni salirse de su viñeta o de la página. Las marcas sin traducir que caen DENTRO
    del texto propio no le quitan espacio."""
    def cruza(a, b):
        return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])

    lineas = _lineas_vineta(gray) if gray is not None else None
    for u in unidades:
        if u["globo"] >= 0 or u["id"] not in activos:
            continue
        reg = u["region"].copy()
        x, y, w, h = u["box"]
        H, W = reg.shape
        if lineas is not None:
            # quedarse en la viñeta del texto: el lado de los bordes de viñeta donde está su centro
            libre = (lineas == 0).astype(np.uint8)
            n, lab = cv2.connectedComponents(libre, connectivity=4)
            cyx = lab[min(H - 1, y + h // 2), min(W - 1, x + w // 2)]
            if cyx > 0:
                panel = lab == cyx
                if panel.sum() < 0.9 * H * W:
                    reg &= panel.astype(np.uint8) * 255
        borde = 6
        reg[:borde] = 0; reg[-borde:] = 0; reg[:, :borde] = 0; reg[:, -borde:] = 0
        for o in unidades:
            if o is u:
                continue
            ox, oy, ow, oh = o["box"]
            if o["id"] in activos:
                s = max(10, int(u["hg"]))
            elif not cruza(u["box"], o["box"]):
                s = max(4, int(0.3 * o["hg"]))
            else:
                continue
            reg[max(0, oy - s):oy + oh + s, max(0, ox - s):ox + ow + s] = 0
        # su propia caja siempre es suya (salvo el margen de página)
        x0, y0 = max(borde, x), max(borde, y)
        reg[y0:min(H - borde, y + h), x0:min(W - borde, x + w)] = 255
        u["region"] = reg


def _palabras(t):
    return re.findall(r"[A-Z0-9']+", (t or "").upper())


def detectar_ecos(unidades, gs, ingles):
    """Ids de bloques que son un eco de otro (el mismo texto repetido detrás o junto a su
    globo, como efecto visual). Se dejan como arte: traducirlos a medias ensucia la página.

    Un eco: 3+ palabras, el 80 % de ellas aparecen en otro bloque más largo o igual, está
    pegado o encimado a ese bloque y no vive en un globo propio distinto."""
    por = {u["id"]: u for u in unidades}
    ecos = set()
    for a in unidades:
        pa = _palabras(ingles.get(a["id"]))
        if len(pa) < 3:
            continue
        for b in unidades:
            if b is a or b["id"] in ecos:
                continue
            pb = _palabras(ingles.get(b["id"]))
            if len(pb) < len(pa):
                continue
            if sum(p in pb for p in pa) < 0.8 * len(pa):
                continue
            if a["globo"] >= 0 and a["globo"] != b["globo"]:
                continue
            zona = (gs[b["globo"]]["mask"].copy() if b["globo"] >= 0
                    else np.zeros(a["letras"].shape, np.uint8))
            x, y, w, h = b["box"]
            zona[y:y + h, x:x + w] = 255
            k = int(1.5 * b["hg"]) | 1
            zona = cv2.dilate(zona, np.ones((k, k), np.uint8))
            if np.count_nonzero(a["letras"] & zona) == 0:
                continue
            # si los dos son igual de largos, el eco es el de menos tinta
            if len(pa) == len(pb):
                ga = float((cv2.erode(a["letras"], np.ones((3, 3), np.uint8)) > 0).sum())
                gb = float((cv2.erode(b["letras"], np.ones((3, 3), np.uint8)) > 0).sum())
                if ga > gb:
                    continue
            ecos.add(a["id"])
            break
    return ecos


def _mayusculas(en):
    letras = [c for c in en if c.isalpha()]
    return len(letras) >= 4 and sum(c.isupper() for c in letras) >= 0.85 * len(letras)


def componer(img, unidades, gs, traducciones, informe=None):
    """traducciones: {id(str): texto} o {id(str): {"es", "en", "estilo", "omitir", "escala"}}.
    «omitir» deja el bloque como en el original; «escala» agranda o achica la letra (revisión
    manual). Devuelve la página traducida (BGR)."""
    lienzo = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def datos(u):
        t = traducciones.get(str(u["id"])) or ""
        return t if isinstance(t, dict) else {"es": t}

    ingles = {u["id"]: datos(u).get("en") or "" for u in unidades}
    ecos = detectar_ecos(unidades, gs, ingles)

    def texto_de(u):
        t = datos(u)
        es, estilo = (t.get("es") or "").strip(), t.get("estilo") or "normal"
        if t.get("omitir") or (u["id"] in ecos and t.get("omitir") is not False):
            return "", estilo
        # manuscrito mal marcado: si el original va en MAYÚSCULAS es cursiva, no letra a mano
        if estilo == "manuscrito" and not t.get("estilo_fijo") and _mayusculas(t.get("en") or ""):
            estilo, es = "cursiva", es.upper()
        return es, estilo

    def ingles_de(u):
        return ingles.get(u["id"], "")

    if informe is not None:
        for u in unidades:
            if datos(u).get("es") and not texto_de(u)[0]:
                informe.append(dict(id=u["id"], omitido="eco" if u["id"] in ecos else "manual",
                                    box=u["box"], globo=u["globo"], es=datos(u).get("es")))

    activos = {u["id"] for u in unidades if texto_de(u)[0]}
    _repartir_globos(unidades, gs, activos)
    _recortar_libres(unidades, activos, gray)
    trabajo = []
    for u in unidades:
        es, estilo = texto_de(u)
        if not es:
            continue
        globo = gs[u["globo"]] if u["globo"] >= 0 else None
        claro, _ = _color_texto(gray, u["letras"])
        fondo = borrar(lienzo, u, globo, claro)
        trabajo.append(dict(u=u, es=es, en=ingles_de(u), claro=claro, globo=globo, estilo=estilo, fondo=fondo,
                            escala=float(datos(u).get("escala") or 1.0)))

    # Globos lisos: usar la forma real del interior (medida tras borrar), no la máscara
    # ondulada del detector, que suele ser más chica.
    gl = cv2.cvtColor(lienzo, cv2.COLOR_BGR2GRAY)
    for t in trabajo:
        u, g = t["u"], t["globo"]
        if g is None:
            continue
        hermanos = [o["u"] for o in trabajo if o["u"]["globo"] == u["globo"] and o["u"] is not u]
        u["compartido"] = bool(hermanos)
        if t["fondo"] is not None:
            vecinos = [o["u"] for o in trabajo if o["u"] is not u]
            # solo compiten los globos vecinos que tienen texto propio a traducir; un
            # «globo» detectado encima sin texto no le quita espacio a este
            con_texto = {o["u"]["globo"] for o in trabajo if o["u"]["globo"] >= 0}
            otros = [o["mask"] for i, o in enumerate(gs) if o is not g and i in con_texto]
            nuevo = _interior_real(gl, u, g, t["fondo"], vecinos, otros)
            if nuevo is not None and np.count_nonzero(nuevo) > 0.5 * np.count_nonzero(u["region"]):
                u["region"] = nuevo
                u["interior_real"] = True

    pil = Image.fromarray(cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for t in trabajo:
        t["pl"] = planear(d, t["u"], t["es"], t["globo"], t["estilo"], t["claro"], en=t["en"], escala=t["escala"])

    # Tamaño parejo: en una página de manga el diálogo va a un tamaño casi constante. Un globo
    # que quedó mucho más grande que el resto (porque le sobra espacio) se baja al tamaño
    # común, salvo que el original también tuviera letra más grande ahí.
    for est in ("normal", "manuscrito", "rotulo"):
        grupo = [t for t in trabajo if t["estilo"] == est]
        if len(grupo) < 3:
            continue
        med_tam = float(np.median([t["pl"]["tam"] for t in grupo]))
        med_letra = float(np.median([t["pl"]["tope_letra"] for t in grupo]))
        for t in grupo:
            if t["escala"] == 1.0 and t["pl"]["tam"] > 1.2 * med_tam and t["pl"]["tope_letra"] < 1.3 * med_letra:
                t["pl"] = planear(d, t["u"], t["es"], t["globo"], t["estilo"], t["claro"],
                                  tope_fijo=int(round(1.2 * med_tam)), en=t["en"])

    for t in trabajo:
        dibujar_plan(pil, d, t["pl"], t["claro"], t["globo"])
        if informe is not None:
            u = t["u"]
            r = {k: v for k, v in t["pl"].items() if k != "plan"}
            r.update(id=u["id"], globo=u["globo"], box=u["box"], hg=u["hg"], es=t["es"])
            informe.append(r)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
