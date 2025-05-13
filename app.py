from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bot import ask_question  # Import your existing bot function
import uuid

app = FastAPI()

# Serve templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow testing from any origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request schema
class QuestionPayload(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question_route(payload: QuestionPayload):
    try:
        response = await ask_question(
            company_id=1,
            bot_id=1,
            question=payload.question.strip(),
            session_id="web-session"
        )

        # If response is an object/dict, convert to string or extract relevant field
        if isinstance(response, dict):
            return {"answer": response.get("answer", str(response))}
        return {"answer": str(response)}

    except Exception as e:
        return {"error": str(e)}

