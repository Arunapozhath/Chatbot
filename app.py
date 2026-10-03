"""
Milkosoft AI Chatbot — FastAPI server
Run:  python app.py   →   open http://localhost:8000
"""
import json
import asyncio
from pathlib import Path

import mysql.connector
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_pipeline import MilkosoftRAG
from auth import get_current_user, get_optional_user, verify_user, create_token
from feedback import save_feedback, get_stats, export_dpo_pairs, get_recent_feedback
from config import (
    OLLAMA_LLM_MODEL, MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASS, MYSQL_DB,
    GROQ_API_KEY,
)


# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="Milkosoft AI Chatbot", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = MilkosoftRAG()

FRONTEND = Path(__file__).parent / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


# ── Schemas ────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    user_login_id: str
    password: str

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    model:   str = OLLAMA_LLM_MODEL
    session_id: str = ""

class FeedbackRequest(BaseModel):
    session_id:  str
    question:    str
    answer:      str
    rating:      int           # 1 = thumbs up, -1 = thumbs down
    correction:  str = ""      # optional improved answer
    source_type: str = "rag"
    sources:     list = []

class IngestResponse(BaseModel):
    status: str
    chunks: int


# ── MySQL helper ───────────────────────────────────────────────────────────
def get_mysql_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASS,
        database=MYSQL_DB, connect_timeout=5,
    )


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = FRONTEND / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Frontend not found. Place index.html in /frontend/</h2>")


@app.get("/health")
async def health():
    from db_connector import db_is_available
    return {
        "status":        "ok",
        "model":         rag.model_name,
        "chroma_ready":  Path("./chroma_db").exists(),
        "db_connected":  db_is_available(),
    }


@app.get("/models")
async def list_models():
    local_models = [
        {"id": "llama3.2:1b", "label": "LLaMA 3.2 (1B) — Local, Fastest"},
        {"id": "llama3.2",    "label": "LLaMA 3.2 (3B) — Local, Fast"},
        {"id": "mistral",     "label": "Mistral 7B — Local, Better quality"},
        {"id": "phi3:mini",   "label": "Phi-3 Mini — Local, Compact"},
        {"id": "gemma2:2b",   "label": "Gemma 2 (2B) — Local, Google"},
    ]
    groq_models = []
    if GROQ_API_KEY:
        groq_models = [
            {"id": "groq:llama-3.1-8b-instant",    "label": "Groq LLaMA 3.1 8B ⚡ — Cloud, Ultra Fast"},
            {"id": "groq:llama-3.3-70b-versatile", "label": "Groq LLaMA 3.3 70B ⚡ — Cloud, Best quality"},
            {"id": "groq:gemma2-9b-it",            "label": "Groq Gemma 2 9B ⚡ — Cloud, Fast"},
            {"id": "groq:mixtral-8x7b-32768",      "label": "Groq Mixtral 8x7B ⚡ — Cloud, Long context"},
        ]
    return {"models": groq_models + local_models, "groq_enabled": bool(GROQ_API_KEY)}


@app.post("/auth/login")
async def login(req: LoginRequest):
    try:
        conn = get_mysql_conn()
        user = verify_user(req.user_login_id, req.password, conn)
        conn.close()
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user)
    return {"token": token, "user": user}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, current_user: dict = Depends(get_optional_user)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    if req.model != rag.model_name:
        rag.switch_model(req.model)

    user_party_id = current_user.get("party_id") if current_user else None

    async def event_generator():
        async for event in rag.astream(req.message, req.history, user_party_id):
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback")
async def submit_feedback(
    req: FeedbackRequest,
    current_user: dict = Depends(get_optional_user),
):
    user_id = (current_user or {}).get("user_login_id", "anonymous")
    save_feedback(
        session_id  = req.session_id,
        user_id     = user_id,
        question    = req.question,
        answer      = req.answer,
        rating      = req.rating,
        correction  = req.correction or None,
        source_type = req.source_type,
        sources     = req.sources,
    )
    return {"status": "saved"}


@app.get("/feedback/stats")
async def feedback_stats(current_user: dict = Depends(get_current_user)):
    return get_stats()


@app.get("/feedback/export")
async def feedback_export(current_user: dict = Depends(get_current_user)):
    pairs = export_dpo_pairs()
    return {"dpo_pairs": pairs, "count": len(pairs)}


@app.get("/feedback/recent")
async def feedback_recent(current_user: dict = Depends(get_current_user)):
    return get_recent_feedback(50)


@app.post("/ingest", response_model=IngestResponse)
async def run_ingest(current_user: dict = Depends(get_current_user)):
    try:
        from ingest import ingest
        count = await asyncio.to_thread(ingest)
        rag._db = None   # force ChromaDB reload
        return {"status": "success", "chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🥛 Milkosoft AI Chatbot v2 starting...")
    print("   Open http://localhost:8000 in your browser\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
