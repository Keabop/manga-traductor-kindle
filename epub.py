"""Lectura de un EPUB de manga: lo extrae y devuelve las imágenes en orden de lectura."""
import json
import os
import re
import zipfile

AQUI = os.path.dirname(os.path.abspath(__file__))
LIBROS = os.path.join(AQUI, "books")


def slug(nombre):
    s = re.sub(r"[^A-Za-z0-9]+", "_", nombre).strip("_")
    return s[:80]


def extraer(epub_path, work):
    if os.path.isdir(work):
        return work
    os.makedirs(work, exist_ok=True)
    with zipfile.ZipFile(epub_path) as z:
        z.extractall(work)
    return work


def leer_opf(opf_path):
    xml = open(opf_path, encoding="utf-8-sig").read()
    items = {}
    for m in re.finditer(r"<item\b([^>]*)/>", xml):
        attrs = dict(re.findall(r'(\w[\w:-]*)="([^"]*)"', m.group(1)))
        if "href" in attrs and "id" in attrs:
            items[attrs["id"]] = attrs
    spine = []
    sm = re.search(r"<spine[^>]*>(.*?)</spine>", xml, re.S)
    if sm:
        for m in re.finditer(r"<itemref\b([^>]*)/>", sm.group(1)):
            attrs = dict(re.findall(r'(\w[\w:-]*)="([^"]*)"', m.group(1)))
            spine.append(attrs.get("idref"))
    return items, spine


def paginas(epub_path, work):
    """Lista de dict(xhtml, image) en el orden del spine; image=None si la página no tiene."""
    extraer(epub_path, work)
    container = open(os.path.join(work, "META-INF", "container.xml"), encoding="utf-8-sig").read()
    opf_rel = re.search(r'full-path="([^"]+)"', container).group(1).replace("/", os.sep)
    items, spine = leer_opf(os.path.join(work, opf_rel))
    base = os.path.dirname(opf_rel)
    out = []
    for idref in spine:
        it = items.get(idref)
        if not it or it.get("media-type") != "application/xhtml+xml":
            continue
        xhtml_rel = os.path.normpath(os.path.join(base, it["href"]))
        xp = os.path.join(work, xhtml_rel)
        if not os.path.exists(xp):
            continue
        html = open(xp, encoding="utf-8-sig", errors="ignore").read()
        m = re.search(r'<img[^>]+src="([^"]+)"', html) or re.search(r'<image[^>]+href="([^"]+)"', html)
        if not m:
            out.append(dict(xhtml=xhtml_rel, image=None))
            continue
        img_rel = os.path.normpath(os.path.join(os.path.dirname(xhtml_rel), m.group(1)))
        out.append(dict(xhtml=xhtml_rel, image=img_rel))
    return out


def cargar(epub_path):
    nombre = os.path.splitext(os.path.basename(epub_path))[0]
    work = os.path.join(LIBROS, slug(nombre))
    meta = dict(epub=epub_path, work=work, name=nombre, pages=paginas(epub_path, work))
    json.dump(meta, open(os.path.join(work, "book.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return meta
