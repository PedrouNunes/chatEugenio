# chat_api_llama/main.py
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from keyboard_agent import generate_keyboard_file

app = FastAPI(title="Eugénio — Teclado AAC (Offline)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class KeyboardRequest(BaseModel):
    description: str
    reference_keyboard: str = ""

@app.post("/keyboard")
def create_keyboard(req: KeyboardRequest):
    start = time.time()

    if req.reference_keyboard:
        description = f"""
You will receive an existing .tec keyboard file and a modification request.

IMPORTANT RULES:
- Keep ALL existing content from the keyboard below
- Only ADD or MODIFY what the user explicitly asks for
- Do NOT remove any existing sections, groups or keys
- Return the complete keyboard with the changes applied

EXISTING KEYBOARD:
{req.reference_keyboard}

MODIFICATION REQUESTED:
{req.description}
"""
    else:
        description = req.description

    tec_content = generate_keyboard_file(description)
    elapsed = round(time.time() - start, 2)

    # Codifica em latin1 para compatibilidade com o sistema Eugénio
    # Caracteres sem equivalente em latin1 são substituídos por ?
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