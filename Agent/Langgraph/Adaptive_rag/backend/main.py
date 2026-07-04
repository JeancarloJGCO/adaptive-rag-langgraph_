
import sys
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

THIS_DIR = Path(__file__).parent
PROJECT_DIR = THIS_DIR.parent  # Adaptive_rag/
sys.path.insert(0, str(PROJECT_DIR))

from src.agents.adaptive_rag import AdaptiveRAGAgent
import database as db

app = FastAPI(title="Adaptive RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Agente

_agent: AdaptiveRAGAgent | None = None


def get_agent() -> AdaptiveRAGAgent:
    global _agent
    if _agent is None:
        _agent = AdaptiveRAGAgent()
    return _agent



# Sesiones temporales
# _temporary_sessions: dict[str, list[adv]] = {}
_temporary_sessions: dict[str, list[dict]] = {}


@app.on_event("startup")
def on_startup():
    db.init_db()


# Modelos
class NewChatRequest(BaseModel):
    title: str = "Nuevo chat"


class MessageRequest(BaseModel):
    session_id: str
    message: str
    mode: Literal["temporary", "permanent"]



# Endpoints
@app.post("/api/sessions/temporary")
def new_temporary_session():
    session_id = str(uuid.uuid4())
    _temporary_sessions[session_id] = []
    return {"session_id": session_id, "mode": "temporary"}


@app.get("/api/sessions/temporary/{session_id}/messages")
def get_temporary_messages(session_id: str):
    if session_id not in _temporary_sessions:
        raise HTTPException(404, "Sesión temporal no encontrada (¿reiniciaste el servidor?)")
    return _temporary_sessions[session_id]



# Endpoints
@app.get("/api/chats")
def list_chats():
    return db.list_chats()


@app.post("/api/chats")
def create_chat(req: NewChatRequest):
    return db.create_chat(req.title)


@app.get("/api/chats/{chat_id}/messages")
def get_chat_messages(chat_id: str):
    if db.get_chat(chat_id) is None:
        raise HTTPException(404, "Chat no encontrado")
    return db.get_messages(chat_id)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    if db.get_chat(chat_id) is None:
        raise HTTPException(404, "Chat no encontrado")
    db.delete_chat(chat_id)
    return {"deleted": chat_id}



# Endpoint principal
@app.post("/api/message")
def send_message(req: MessageRequest):
    if req.mode == "permanent" and db.get_chat(req.session_id) is None:
        raise HTTPException(404, "Chat permanente no encontrado")
    if req.mode == "temporary" and req.session_id not in _temporary_sessions:
        raise HTTPException(404, "Sesión temporal no encontrada")

    agent = get_agent()
    generation = agent.invoke(req.message)

    if req.mode == "permanent":
        db.add_message(req.session_id, "user", req.message)
        db.add_message(req.session_id, "assistant", generation)
        db.rename_chat_if_default(req.session_id, req.message)
    else:
        _temporary_sessions[req.session_id].append({"role": "user", "content": req.message})
        _temporary_sessions[req.session_id].append({"role": "assistant", "content": generation})

    return {"session_id": req.session_id, "generation": generation}


# Frontend estático
FRONTEND_DIR = PROJECT_DIR / "frontend"


@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
