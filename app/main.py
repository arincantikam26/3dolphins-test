from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from app.schemas import (
    QuestionRequest,
    APIResponse,
    QuestionData,
    ErrorResponse,
)
from app.service import RAGService, AppException

app = FastAPI(title="RAG System API (Ollama)", version="1.0.0")
rag_service = RAGService()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/api/v1/questions", response_model=APIResponse)
async def ask_question(request: QuestionRequest):
    result = await rag_service.ask(request.question, request.top_k)

    return APIResponse(
        data=QuestionData(
            answer=result["answer"],
            sources=result["sources"],
        ),
        meta={"processing_time_ms": result["processing_time_ms"]},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error={
                "code": exc.status_code,
                "message": exc.message,
                "details": exc.details,
            }
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error={
                "code": 422,
                "message": "Validation error",
                "details": str(exc),
            }
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={
                "code": 500,
                "message": "Internal server error",
                "details": str(exc),
            }
        ).model_dump(),
    )