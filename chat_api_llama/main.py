import time
import unicodedata
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
from keyboard_agent import generate_keyboard_with_history
import httpx

_HTTP = httpx.Client(timeout=8.0)

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


def fix_escapes(text):
    """Corrige sequências de escape que o modelo pode gerar em vez dos caracteres reais.
    Ex: \\xe7 em texto → ç  |  \\xc3\\xa7 em texto → ç
    """
    def fix_utf8_seq(m):
        hex_vals = re.findall(r'[0-9a-fA-F]{2}', m.group(0))
        try:
            return bytes.fromhex(''.join(hex_vals)).decode('utf-8')
        except Exception:
            return m.group(0)

    text = re.sub(r'(?:\\x[0-9a-fA-F]{2}){2,}', fix_utf8_seq, text)

    def fix_single(m):
        try:
            return bytes.fromhex(m.group(1)).decode('latin1')
        except Exception:
            return m.group(0)

    text = re.sub(r'\\x([0-9a-fA-F]{2})', fix_single, text)
    return text


def encode_tec(text):
    """Normaliza e codifica o conteúdo .tec para latin1 (formato esperado pelo Eugénio)."""
    text = fix_escapes(text)
    text = text.replace('\uFFFD', '')
    text = unicodedata.normalize('NFC', text)
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
            {"role": "user",      "content": "Use este teclado como base para as próximas modificações."},
            {"role": "assistant", "content": req.reference_keyboard},
        ]
    elif req.reference_keyboard:
        history.append({"role": "assistant", "content": req.reference_keyboard})

    tec_content = generate_keyboard_with_history(req.description, history)
    elapsed = round(time.time() - start, 2)

    return Response(
        content=encode_tec(tec_content),
        media_type="text/plain; charset=latin1",
        headers={
            "Content-Disposition": "attachment; filename=teclado.tec",
            "X-Elapsed-Seconds": str(elapsed),
        }
    )


@app.get("/pictogram")
def get_pictogram(q: str):
    """Proxy para a API ARASAAC — evita problemas de CORS ao abrir chat.html como file://"""
    try:
        res = _HTTP.get(f"https://api.arasaac.org/v1/pictograms/pt/search/{q}")
        data = res.json()
        if data and isinstance(data, list):
            pic_id = data[0]["_id"]
            return {"id": pic_id, "url": f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_300.png"}
        return {"id": None, "url": None}
    except Exception:
        return {"id": None, "url": None}


@app.get("/health")
def health():
    return {"status": "ok", "backend": "Ollama (offline)"}