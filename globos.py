"""Detección de globos con un YOLOv8-seg (kitsumed/yolov8m_seg-speech-bubble, ONNX).

Devuelve un globo por bocadillo, con su máscara a resolución de página, en lugar de los
fragmentos línea a línea que daba Tesseract.
"""
import os
import cv2
import numpy as np
import onnxruntime as ort

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(AQUI, "models", "bubble_seg_model_dynamic.onnx")
TAM = 640

_sesion = None


def sesion():
    global _sesion
    if _sesion is None:
        _sesion = crear_sesion(MODELO)
    return _sesion


def crear_sesion(ruta):
    """Sesión ONNX en GPU si cuDNN carga; si no, CPU."""
    ort.set_default_logger_severity(3)
    try:
        ort.preload_dlls()
        s = ort.InferenceSession(ruta, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if "CUDAExecutionProvider" in s.get_providers():
            return s
    except Exception:
        pass
    return ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])


def _letterbox(img, tam=TAM):
    h, w = img.shape[:2]
    r = tam / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    lienzo = np.full((tam, tam, 3), 114, np.uint8)
    py, px = (tam - nh) // 2, (tam - nw) // 2
    lienzo[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return lienzo, r, px, py


def _nms(cajas, puntajes, umbral=0.45):
    orden = puntajes.argsort()[::-1]
    quedan = []
    while orden.size:
        i = orden[0]
        quedan.append(i)
        x0 = np.maximum(cajas[i, 0], cajas[orden[1:], 0])
        y0 = np.maximum(cajas[i, 1], cajas[orden[1:], 1])
        x1 = np.minimum(cajas[i, 2], cajas[orden[1:], 2])
        y1 = np.minimum(cajas[i, 3], cajas[orden[1:], 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        a = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
        iou = inter / (a[i] + a[orden[1:]] - inter + 1e-6)
        orden = orden[1:][iou < umbral]
    return quedan


def detectar(img_bgr, conf=0.35):
    """Lista de globos: dict(box=[x,y,w,h], conf, mask=np.uint8 HxW 0/255)."""
    H, W = img_bgr.shape[:2]
    lb, r, px, py = _letterbox(img_bgr)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    out0, protos = sesion().run(None, {"images": x})
    pred = out0[0].T                      # (anclas, 4 + 1 + 32)
    puntajes = pred[:, 4]
    sel = puntajes >= conf
    pred, puntajes = pred[sel], puntajes[sel]
    if not len(pred):
        return []
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    cajas = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    keep = _nms(cajas, puntajes)
    cajas, puntajes, coefs = cajas[keep], puntajes[keep], pred[keep, 5:]

    p = protos[0]                          # (32, mh, mw)
    mh, mw = p.shape[1:]
    mascaras = 1 / (1 + np.exp(-(coefs @ p.reshape(32, -1)))).reshape(-1, mh, mw)
    globos = []
    for caja, s, m in zip(cajas, puntajes, mascaras):
        # máscara del lienzo 640 -> recortar el relleno -> tamaño de página
        m = cv2.resize(m, (TAM, TAM), interpolation=cv2.INTER_LINEAR)
        nh, nw = int(round(H * r)), int(round(W * r))
        m = m[py:py + nh, px:px + nw]
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        x0 = int(np.clip((caja[0] - px) / r, 0, W)); y0 = int(np.clip((caja[1] - py) / r, 0, H))
        x1 = int(np.clip((caja[2] - px) / r, 0, W)); y1 = int(np.clip((caja[3] - py) / r, 0, H))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        bin_ = np.zeros((H, W), np.uint8)
        bin_[y0:y1, x0:x1] = (m[y0:y1, x0:x1] > 0.5).astype(np.uint8) * 255
        if bin_.sum() == 0:
            continue
        globos.append(dict(box=[x0, y0, x1 - x0, y1 - y0], conf=float(s), mask=bin_))
    return globos


if __name__ == "__main__":
    import sys
    for ruta in sys.argv[1:]:
        img = cv2.imread(ruta)
        gs = detectar(img)
        vis = img.copy()
        for i, g in enumerate(gs):
            cs, _ = cv2.findContours(g["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 0, 255), 2)
            x, y, w, h = g["box"]
            cv2.putText(vis, f"{i}:{g['conf']:.2f}", (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        sal = os.path.splitext(ruta)[0] + "_globos.png"
        if len(sys.argv) > 1 and os.environ.get("SALIDA"):
            sal = os.path.join(os.environ["SALIDA"], os.path.basename(sal))
        cv2.imwrite(sal, vis)
        print(ruta, len(gs), "globos ->", sal)
