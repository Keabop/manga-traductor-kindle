"""Une imágenes lado a lado con un rótulo encima de cada una.

    python comparar.py salida.png "Rótulo A" a.png "Rótulo B" b.png ...
"""
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw

import pagina


def lado_a_lado(pares, alto=1100):
    piezas = []
    for rotulo, ruta in pares:
        img = cv2.imread(ruta)
        s = alto / img.shape[0]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cab = Image.new("RGB", (img.shape[1], 54), (30, 30, 30))
        d = ImageDraw.Draw(cab)
        f = pagina.fuente(30)
        d.text(((img.shape[1] - pagina.largo(d, rotulo, f)) / 2, 8), rotulo, font=f, fill=(255, 255, 255))
        piezas.append(np.vstack([cv2.cvtColor(np.array(cab), cv2.COLOR_RGB2BGR), img]))
        piezas.append(np.full((alto + 54, 12, 3), 255, np.uint8))
    return np.hstack(piezas[:-1])


if __name__ == "__main__":
    salida, resto = sys.argv[1], sys.argv[2:]
    cv2.imwrite(salida, lado_a_lado(list(zip(resto[::2], resto[1::2]))))
    print(salida)
