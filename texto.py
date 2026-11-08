"""Detección de texto con comic-text-detector (dmMaze, ONNX de manga-image-translator).

Da dos cosas: bloques de texto (cajas) y una máscara de píxeles de texto. La máscara sirve
para borrar solo las letras, también fuera de los globos (narración, texto sobre el dibujo).
"""
import os
import cv2
import numpy as np

from globos import crear_sesion, _nms

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(AQUI, "models", "comictextdetector.onnx")
TAM = 1024

_sesion = None


def sesion():
    global _sesion
    if _sesion is None:
        _sesion = crear_sesion(MODELO)
    return _sesion


def detectar(img_bgr, conf=0.4):
    """Devuelve (bloques, mascara). bloques: dict(box=[x,y,w,h], conf). mascara: HxW 0..255."""
    H, W = img_bgr.shape[:2]
    r = TAM / max(H, W)
    nh, nw = int(round(H * r)), int(round(W * r))
    lienzo = np.zeros((TAM, TAM, 3), np.uint8)          # relleno abajo/derecha
    lienzo[:nh, :nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    blk, seg, _det = sesion().run(None, {"images": x})

    b = blk[0]
    b = b[b[:, 4] >= conf]
    bloques = []
    if len(b):
        cajas = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                          b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        for i in _nms(cajas, b[:, 4], 0.35):
            x0, y0, x1, y1 = (cajas[i] / r).tolist()
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 >= 6 and y1 - y0 >= 6:
                bloques.append(dict(box=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)], conf=float(b[i, 4])))

    m = seg[0, 0, :nh, :nw]
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    mascara = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    return bloques, mascara


if __name__ == "__main__":
    import sys
    for ruta in sys.argv[1:]:
        img = cv2.imread(ruta)
        bloques, mascara = detectar(img)
        vis = img.copy()
        vis[mascara > 127] = (0.4 * vis[mascara > 127] + 0.6 * np.array([0, 200, 0])).astype(np.uint8)
        for i, bl in enumerate(bloques):
            x, y, w, h = bl["box"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(vis, f"{i}", (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        sal = os.path.join(os.environ.get("SALIDA", "."), os.path.splitext(os.path.basename(ruta))[0] + "_texto.png")
        cv2.imwrite(sal, vis)
        print(ruta, len(bloques), "bloques ->", sal)
