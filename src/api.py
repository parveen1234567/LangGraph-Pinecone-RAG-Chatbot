from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

try:
    from .agent import answer_question
    from .config import Settings
    from .ingest import load_documents
except ImportError:  # pragma: no cover
    from src.agent import answer_question
    from src.config import Settings
    from src.ingest import load_documents

app = FastAPI(title="RAG Agentic AI", version="1.0.0")


@app.get("/", include_in_schema=False)
def open_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask_question(payload: AskRequest) -> dict[str, str]:
    settings = Settings()
    documents = load_documents(settings.data_dir)
    answer = answer_question(payload.question, documents)
    return {"answer": answer}
