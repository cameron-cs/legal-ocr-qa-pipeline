import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from legal_qa_pipeline import build_pipeline
from src.processor.retriever import AnswerRetriever
from src.utils import clean_text

retriever: AnswerRetriever = None
collection_name: str = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, collection_name

    # args from env vars or fallback defaults
    json_path = os.getenv("QA_JSON", "data/ocr/Amber Heard's memorandum.json")
    embedder = os.getenv("QA_EMBEDDER", "all-MiniLM-L6-v2")
    reranker = os.getenv("QA_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    collection_name = os.getenv("QA_COLLECTION", "query-index")
    confidence = float(os.getenv("QA_CONFIDENCE", "0.85"))
    qdrant_uri = os.getenv("QA_QDRANT", ":memory:")

    if not Path(json_path).exists():
        raise FileNotFoundError(f"JSON path not found: {json_path}")

    retriever = build_pipeline(
        json_path=Path(json_path),
        embedder_name=embedder,
        reranker_name=reranker,
        collection_name=collection_name,
        confidence_threshold=confidence,
        qdrant_uri=qdrant_uri
    )

    yield  # allows app to run


app = FastAPI(lifespan=lifespan)


class QARequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: QARequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not ready")

    result = retriever.retrieve_answer(clean_text(req.question), collection_name=collection_name)
    if not result:
        return {"answer": None, "pages": []}

    return {
        "answer": result.answer,
        "pages": result.pages
    }


if __name__ == "__main__":
    import uvicorn

    os.environ.setdefault("QA_JSON", "../data/ocr/Amber Heard's memorandum.json")
    os.environ.setdefault("QA_COLLECTION", "query-index")
    os.environ.setdefault("QA_QDRANT", ":memory:")

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
