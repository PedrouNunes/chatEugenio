# chat_api_llama/main.py
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
from keyboard_agent import generate_keyboard_with_history

app = FastAPI(title="Eugénio — Teclado AAC (Offline)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/health")
def health():
    return {"status": "ok", "backend": "Ollama (offline)"}