import time
import httpx
from fastapi.concurrency import run_in_threadpool
from app.config import (
    embedding_model,
    async_qdrant_client,
    COLLECTION_NAME,
    OLLAMA_URL,
    OLLAMA_MODEL,
)


class AppException(Exception):
    def __init__(self, status_code: int, message: str, details: str | None = None):
        self.status_code = status_code
        self.message = message
        self.details = details


class RAGService:

    async def embed(self, text: str):
        return await run_in_threadpool(
            lambda: embedding_model.encode(text).tolist()
        )

    async def search(self, query_vector, top_k: int):
        response = await async_qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        )

        hits = response.points

        if not hits:
            raise AppException(
                status_code=404,
                message="No relevant documents found",
                details="Vector search returned empty results",
            )
        
        results = []

        for hit in hits:
            # Handle both tuple and object cases
            if isinstance(hit, tuple):
                point = hit[0]
                score = hit[1]
            else:
                point = hit
                score = hit.score

            results.append(
                {
                    "content": point.payload["text"],
                    "score": score,
                }
            )

        return results

    async def generate_answer(self, question: str, context: str):
        prompt = f"""
You are an AI assistant.
Answer strictly based on the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
        timeout = httpx.Timeout(120.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
            )

        if response.status_code != 200:
            raise AppException(
                status_code=500,
                message="LLM generation failed",
                details=response.text,
            )

        return response.json()["response"]

    async def ask(self, question: str, top_k: int):
        start_time = time.time()

        query_vector = await self.embed(question)
        results = await self.search(query_vector, top_k)

        context = "\n".join(r["content"] for r in results)
        answer = await self.generate_answer(question, context)

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "answer": answer.strip(),
            "sources": results,
            "processing_time_ms": processing_time,
        }