# chat_api/main.py
import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from agent import build_agent
from alfred import build_alfred
from keyboard_agent import generate_keyboard_file

app = FastAPI(title="Eugénio Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

simple_agent = build_agent()
alfred_agent = build_alfred()

class ChatRequest(BaseModel):
    message: str
    reset: bool = True

class ChatResponse(BaseModel):
    response: str
    elapsed_seconds: float

class KeyboardRequest(BaseModel):
    description: str
    reference_keyboard: str = ""  # teclado de referência opcional

def run_agent(agent, message: str, reset: bool) -> ChatResponse:
    start = time.time()
    result = agent.run(message, reset=reset)
    elapsed = round(time.time() - start, 2)
    return ChatResponse(response=str(result), elapsed_seconds=elapsed)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return run_agent(simple_agent, req.message, req.reset)

@app.post("/alfred", response_model=ChatResponse)
def alfred(req: ChatRequest):
    return run_agent(alfred_agent, req.message, req.reset)

@app.post("/keyboard")
def create_keyboard(req: KeyboardRequest):
    start = time.time()

    # Se tiver teclado de referência, inclui no prompt
    if req.reference_keyboard:
        description = f"""
Use this keyboard as reference for the format and style:

{req.reference_keyboard}

Now create a new keyboard based on this description:
{req.description}
"""
    else:
        description = req.description

    tec_content = generate_keyboard_file(description)
    elapsed = round(time.time() - start, 2)

    return Response(
        content=tec_content,
        media_type="text/plain",
        headers={
            "Content-Disposition": "attachment; filename=teclado.tec",
            "X-Elapsed-Seconds": str(elapsed),
        }
    )

@app.get("/health")
def health():
    return {"status": "ok"}