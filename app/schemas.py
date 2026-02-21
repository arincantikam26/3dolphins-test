from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: Optional[int] = 3


class SourceItem(BaseModel):
    content: str
    score: float


class QuestionData(BaseModel):
    answer: str
    sources: List[SourceItem]


class APIResponse(BaseModel):
    data: QuestionData
    meta: dict


class ErrorDetail(BaseModel):
    code: int
    message: str
    details: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail