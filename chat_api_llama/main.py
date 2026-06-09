# chat_api_llama/main.py
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
from keyboard_agent import generate_keyboard_with_history

# Cliente HTTP reutilizável para chamadas à API ARASAAC
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
    role: str      # "user" ou "assistant"
    content: str   # texto do pedido ou conteúdo .tec gerado


class KeyboardRequest(BaseModel):
    description: str
    reference_keyboard: str = ""
    history: List[HistoryMessage] = []


@app.post("/keyboard")
def create_keyboard(req: KeyboardRequest):
    start = time.time()

    # Constrói o histórico para passar ao modelo
    history = [{"role": m.role, "content": m.content} for m in req.history]

    # Se há teclado de referência, injeta como primeira mensagem do assistente
    if req.reference_keyboard and not history:
        history = [
            {"role": "user",      "content": "Use este teclado como base para as próximas modificações."},
            {"role": "assistant", "content": req.reference_keyboard},
        ]
    elif req.reference_keyboard:
        history.append({"role": "assistant", "content": req.reference_keyboard})

    description = req.description

    tec_content = generate_keyboard_with_history(description, history)

    elapsed = round(time.time() - start, 2)

    # ── Corrigir caracteres problemáticos gerados pelo modelo --
    import unicodedata, re

    # 1. Converter sequências de escape literais que o modelo pode gerar
    #    ex: \xe7 (texto) → ç  |  \xc3\xa7 (texto) → ç
    def fix_escapes(text):
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

    tec_content = fix_escapes(tec_content)

    # 2. Remover caracter de substituição U+FFFD (modelo não conseguiu gerar o char)
    tec_content = tec_content.replace('\uFFFD', '')

    # 3. Normalização NFC — c + cedilha combinado → ç único
    tec_content = unicodedata.normalize('NFC', tec_content)

    try:
        content_bytes = tec_content.encode("latin1")
    except (UnicodeEncodeError, UnicodeDecodeError):
        content_bytes = tec_content.encode("latin1", errors="replace")

    return Response(
        content=content_bytes,
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