# Traductor de manga (inglés → español) para Kindle

Toma un manga en EPUB en inglés y devuelve el mismo manga en español, con los globos
borrados y vueltos a rotular, listo para leer en una Kindle.

- **Detecta** cada globo y cada texto de la página (también fuera de los globos y sobre
  fondos oscuros) con modelos de visión que corren en tu PC.
- **Traduce** con Claude (Opus), que lee la página completa: entiende quién habla, el orden
  de lectura, el tono y deja como arte las onomatopeyas japonesas.
- **Borra** el inglés rellenando el globo o reconstruyendo el dibujo (LaMa).
- **Rotula** el español al tamaño del original, siguiendo la forma del globo, con una
  tipografía por estilo: diálogo, grito, cursiva, manuscrito y carteles.
- **Convierte** para Kindle con Kindle Comic Converter (lectura de derecha a izquierda,
  páginas al tamaño de tu pantalla).

> Úsalo con mangas que tengas tú, para leerlos tú. No compartas los archivos traducidos.

## Qué necesitas

| | |
|---|---|
| Windows 10/11 | macOS y Linux funcionan para traducir; la conversión a Kindle necesita instalar KCC aparte. |
| [Python 3.11 o más reciente](https://www.python.org/downloads/) | Marca «Add python.exe to PATH» al instalarlo. |
| [Claude Code](https://claude.com/claude-code) con sesión iniciada | Hace la traducción. Usa tu suscripción de Claude (Pro o Max). |
| Kindle Previewer 3 (opcional) | Solo para pasar el libro **por cable USB**. [Descarga oficial de Amazon](https://www.amazon.com/Kindle-Previewer/b?node=21381691011). |
| GPU NVIDIA (opcional) | Con ella el análisis tarda ~0.4 s por página; sin ella, unos segundos. |

**Costo:** un tomo de ~175 páginas con Opus consume el equivalente a unos US$11 de uso de tu
suscripción (no se cobra aparte, pero cuenta para tus límites). Con `--modelo sonnet` es
aproximadamente la mitad, con traducciones algo menos finas.

## Instalación

```
git clone https://github.com/Keabop/manga-traductor-kindle.git
cd manga-traductor-kindle
python instalar.py
```

`instalar.py` crea el entorno `.venv`, instala las dependencias y descarga los modelos
(~410 MB), las fuentes y Kindle Comic Converter. Si algo se corta, vuelve a ejecutarlo:
solo baja lo que falte.

Si es la primera vez que usas Claude Code, abre una terminal, escribe `claude` e inicia
sesión con `/login`.

## Traducir un tomo

1. **Glosario de la obra.** Copia `glosarios/_plantilla.md` como `glosarios/<obra>.md` y
   escribe los nombres de los personajes y si son chico o chica. Es lo que más mejora la
   traducción (evita «jugador» por «jugadora»). Hay un ejemplo en
   `glosarios/my-dress-up-darling.md`.

2. **Prueba con unas páginas** antes de gastar un tomo entero:

   ```
   .venv\Scripts\python traducir.py analizar "C:\ruta\Mi manga vol 1.epub"
   .venv\Scripts\python traducir.py traducir "C:\ruta\Mi manga vol 1.epub" --glosario glosarios\mi-manga.md --paginas 5-16
   .venv\Scripts\python traducir.py componer "C:\ruta\Mi manga vol 1.epub" --paginas 5-16
   ```

   Las páginas quedan en `books\<tomo>\es\`.

3. **Tomo completo**, todo seguido (unos 40 minutos):

   ```
   .venv\Scripts\python traducir.py todo "C:\ruta\Mi manga vol 1.epub" --glosario glosarios\mi-manga.md
   ```

   Si se interrumpe, vuelve a lanzar el mismo comando: continúa donde se quedó.

## Revisar y corregir

```
.venv\Scripts\python revisar.py "C:\ruta\Mi manga vol 1.epub"
```

Abre en el navegador una vista con el original junto a la traducida y la lista de
bloques que conviene mirar (letra que quedó chica, texto vertical, ecos que se dejaron como
arte). Por cada bloque puedes cambiar el texto, el estilo, el tamaño o dejarlo como en el
original. «Guardar y recomponer» (Ctrl+Enter) rehace solo esa página en segundos, sin
volver a traducir. Al terminar, «Armar EPUB» y luego el paso de Kindle.

## Pasarlo a la Kindle

```
.venv\Scripts\python traducir.py kindle "C:\ruta\Mi manga vol 1.epub" --perfil KPW6
```

El archivo queda en `salida\kindle\`.

- **Por cable USB (recomendado):** genera un `.mobi` (necesita Kindle Previewer 3
  instalado; se detecta solo). Conecta la Kindle y copia el archivo a la carpeta
  `documents`.
- **Por Send to Kindle:** usa `--formato epub` y súbelo desde la app o
  [la web](https://www.amazon.com/sendtokindle) (máx. 200 MB; por correo el límite es 50 MB).

Perfiles de KCC más comunes:

| Kindle | `--perfil` |
|---|---|
| Paperwhite 12.ª gen (2024) | `KPW6` |
| Paperwhite 11.ª gen / Signature | `KPW5` |
| Paperwhite 3/4, Oasis | `KPW34` |
| Kindle básico 11.ª gen / 2024 | `K11` |
| Scribe | `KS` |

Para cómics occidentales (izquierda a derecha) agrega `--occidental`.

## Cómo funciona

| Paso | Archivo | Qué hace |
|---|---|---|
| Lectura | `epub.py` | Extrae las páginas del EPUB en orden de lectura. |
| Detección | `globos.py`, `texto.py` | Globos (YOLOv8-seg) y píxeles de texto (comic-text-detector), en ONNX. |
| Análisis | `pagina.py` | Agrupa letras en bloques, mide la letra del original y numera los bloques. |
| Traducción | `traducir.py` | Manda las páginas numeradas a `claude -p` en lotes de 12. |
| Composición | `pagina.py` | Borra (relleno o LaMa) y rotula ajustando tamaño, renglones y estilo. |
| Revisión | `revisar.py`, `revision/` | Servidor local para corregir bloques. |
| Kindle | `traducir.py kindle` | Kindle Comic Converter. |
| Diagnóstico | `medir.py`, `comparar.py`, `ver_regiones.py` | Tamaño contra el original, comparativas y zonas de escritura. |

## Licencia

El código de este repositorio es MIT (ver `LICENSE`). Lo de terceros conserva su licencia.

## Créditos y licencias de terceros

Se descargan aparte con `instalar.py`; no vienen en este repositorio:

- Detección de globos: [kitsumed/yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble) (GPL-3.0).
- Detección de texto: comic-text-detector de dmMaze, publicado por [manga-image-translator](https://github.com/zyddnys/manga-image-translator) (GPL-3.0).
- Borrado: [LaMa en ONNX](https://huggingface.co/Carve/LaMa-ONNX) (Apache-2.0).
- Conversión: [Kindle Comic Converter](https://github.com/ciromattia/kcc) (ISC).
- Fuentes de Google Fonts (OFL): Comic Relief, Bangers, Gochi Hand, Oswald y Archivo Narrow.
