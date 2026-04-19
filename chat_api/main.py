# chat_api/main.py
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import build_agent
from alfred import build_alfred

load_dotenv()

app = FastAPI(title="Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa os dois agentes ao rodar
simple_agent = build_agent()
alfred_agent = build_alfred()

class ChatRequest(BaseModel):
    message: str
    reset: bool = True

class ChatResponse(BaseModel):
    response: str
    elapsed_seconds: float

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

@app.get("/health")
def health():
    return {"status": "ok"}