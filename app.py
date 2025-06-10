from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bot import ask_question, EventStreamHandler
from typing import Optional
import jwt
import uuid
import os
import asyncio

# Load global JWT signing secret
JWT_SECRET = os.getenv("JWT_SECRET", "my-ultra-secure-signing-key")

app = FastAPI()

# Mount Jinja2 templates and static files (including widget.js)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS: allow all during dev (restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming request
class QuestionPayload(BaseModel):
    question: str
    k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    session_id: Optional[str] = None

# JWT auth dependency
def verify_token(authorization: str = Header(...)) -> dict:
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Invalid authorization scheme")

        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])

        if "company_id" not in payload or "bot_id" not in payload:
            raise ValueError("Token must include 'company_id' and 'bot_id'")

        return {
            "company_id": payload["company_id"],
            "bot_id": payload["bot_id"]
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Unauthorized: {e}")

# Route to serve optional in-browser test UI
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Secure API route for question answering with streaming
@app.post("/ask")
async def ask_question_route(
    payload: QuestionPayload,
    token_data: dict = Depends(verify_token)
):
    session_id = payload.session_id or str(uuid.uuid4())
    company_id = token_data['company_id']
    bot_id = token_data['bot_id']
    
    stream_handler = EventStreamHandler()
    task = ask_question(
        question=payload.question,
        company_id=company_id,
        bot_id=bot_id,
        session_id=session_id,
        streaming=True,
        stream_handler=stream_handler,
        k=payload.k,
        similarity_threshold=payload.similarity_threshold,
        verbose=True
    )
    asyncio.create_task(task)
    return StreamingResponse(
        stream_handler.astream(),
        media_type="text/event-stream",
        headers={"X-Session-ID": session_id}
    )

# Serve widget.js at a friendly path (optional)
@app.get("/widget.js")
async def serve_widget():
    widget_path = os.path.join("static", "widget.js")
    if not os.path.exists(widget_path):
        raise HTTPException(status_code=404, detail="widget.js not found")
    return FileResponse(widget_path, media_type="application/javascript")
