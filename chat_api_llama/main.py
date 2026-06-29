import os
import re
import time
import io
import unicodedata
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
from keyboard_agent import generate_keyboard_with_history
import httpx
try:
    from PIL import Image as _PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_HTTP = httpx.Client(timeout=8.0)

# Caminho da pasta de teclados do Eugénio — mudar se necessário
EUGENIO_FOLDER  = r"C:\Users\Pedro Nunes\AppData\Roaming\LabSI2-INESC-ID\Eugénio 3.0"
PICTO_FOLDER    = os.path.join(EUGENIO_FOLDER, "CAT_IMG_pic")

app = FastAPI(title="Eugénio — Teclado AAC (Offline)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Elapsed-Seconds"],
)


class HistoryMessage(BaseModel):
    role: str
    content: str


class KeyboardRequest(BaseModel):
    description: str
    reference_keyboard: str = ""
    history: List[HistoryMessage] = []


class SaveRequest(BaseModel):
    content: str
    name: str

class PictogramSaveRequest(BaseModel):
    word: str
    pic_id: int


def fix_encoding(text):
    # O modelo às vezes gera \xe7 ou \xc3\xa7 como texto em vez do caracter real.
    # Aqui converte essas sequências de volta para os caracteres correctos.
    def fix_utf8_seq(m):
        hex_vals = re.findall(r'[0-9a-fA-F]{2}', m.group(0))
        try:
            return bytes.fromhex(''.join(hex_vals)).decode('utf-8')
        except Exception:
            return m.group(0)

    def fix_single(m):
        try:
            return bytes.fromhex(m.group(1)).decode('latin1')
        except Exception:
            return m.group(0)

    text = re.sub(r'(?:\\x[0-9a-fA-F]{2}){2,}', fix_utf8_seq, text)
    text = re.sub(r'\\x([0-9a-fA-F]{2})', fix_single, text)
    text = text.replace('\uFFFD', '')
    text = unicodedata.normalize('NFC', text)
    return text


def to_latin1(text):
    try:
        return text.encode("latin1")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text.encode("latin1", errors="replace")


@app.post("/keyboard")
def create_keyboard(req: KeyboardRequest):
    start = time.time()

    history = [{"role": m.role, "content": m.content} for m in req.history]

    if req.reference_keyboard and not history:
        history = [
            {"role": "user",      "content": "Usa este teclado como base."},
            {"role": "assistant", "content": req.reference_keyboard},
        ]
    elif req.reference_keyboard:
        history.append({"role": "assistant", "content": req.reference_keyboard})

    tec_content = generate_keyboard_with_history(req.description, history)
    tec_content = fix_encoding(tec_content)
    elapsed = round(time.time() - start, 2)

    return Response(
        content=to_latin1(tec_content),
        media_type="text/plain; charset=latin1",
        headers={
            "Content-Disposition": "attachment; filename=teclado.tec",
            "X-Elapsed-Seconds": str(elapsed),
        }
    )


@app.post("/save_keyboard")
def save_keyboard(req: SaveRequest):
    try:
        if not os.path.isdir(EUGENIO_FOLDER):
            return {"ok": False, "error": f"Pasta não encontrada: {EUGENIO_FOLDER}"}

        safe_name = "".join(c for c in req.name if c not in r'\/:*?"<>|').strip() or "teclado"
        filepath = os.path.join(EUGENIO_FOLDER, safe_name + ".tec")
        content = fix_encoding(req.content)

        with open(filepath, "wb") as f:
            f.write(to_latin1(content))

        return {"ok": True, "path": filepath, "name": safe_name + ".tec"}

    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/list_keyboards")
def list_keyboards():
    try:
        if not os.path.isdir(EUGENIO_FOLDER):
            return {"ok": False, "files": []}
        files = [f for f in os.listdir(EUGENIO_FOLDER) if f.endswith(".tec")]
        return {"ok": True, "files": sorted(files)}
    except Exception:
        return {"ok": False, "files": []}


@app.get("/pictogram")
def get_pictogram(q: str):
    try:
        res = _HTTP.get(f"https://api.arasaac.org/v1/pictograms/pt/search/{q}")
        data = res.json()
        if data and isinstance(data, list):
            pic_id = data[0]["_id"]
            return {"id": pic_id, "url": f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_300.png"}
        return {"id": None, "url": None}
    except Exception:
        return {"id": None, "url": None}


@app.post("/save_pictogram")
def save_pictogram(req: PictogramSaveRequest):
    """Descarrega o pictograma ARASAAC, converte para BMP e guarda na pasta CAT_IMG_pic do Eugénio."""
    if not _PIL_OK:
        return {"ok": False, "error": "Pillow não instalado — corre: pip install Pillow"}
    try:
        os.makedirs(PICTO_FOLDER, exist_ok=True)

        # Nome de ficheiro seguro a partir da palavra (sem acentos, lowercase)
        norm = unicodedata.normalize('NFKD', req.word)
        ascii_word = norm.encode('ascii', 'ignore').decode().lower()
        safe = re.sub(r'[^a-z0-9]', '_', ascii_word).strip('_') or f"picto_{req.pic_id}"
        filename = safe + '.bmp'
        filepath = os.path.join(PICTO_FOLDER, filename)

        # Se já existe, devolve sem re-descarregar
        if os.path.isfile(filepath):
            return {"ok": True, "filename": filename, "cached": True}

        # Descarrega PNG do ARASAAC
        url = f"https://static.arasaac.org/pictograms/{req.pic_id}/{req.pic_id}_300.png"
        png_res = _HTTP.get(url, timeout=12.0)
        if png_res.status_code != 200:
            return {"ok": False, "error": f"ARASAAC devolveu {png_res.status_code}"}

        # Converte PNG → BMP (sem canal alpha — BMP não suporta transparência)
        img = _PILImage.open(io.BytesIO(png_res.content)).convert("RGB")
        bmp_buf = io.BytesIO()
        img.save(bmp_buf, format="BMP")

        with open(filepath, "wb") as f:
            f.write(bmp_buf.getvalue())

        return {"ok": True, "filename": filename, "cached": False}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok", "backend": "Ollama (offline)", "pillow": _PIL_OK}