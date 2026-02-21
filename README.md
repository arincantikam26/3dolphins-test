# Simple RAG System API

This project implements a Retrieval-Augmented Generation (RAG) system using:

- FastAPI (REST API)
- Qdrant (Vector Database)
- Sentence Transformers (Embeddings)
- OpenAI (LLM)

---

## Setup Instruction

1. Start Infrastructure (Qdrant + Ollama)
```
docker compose up -d
```

2. Pull LLM Model
```
docker compose exec ollama ollama pull mistral
```
Verify:
```
docker compose exec ollama ollama list
```

3. Create Virtual Environment
```
python3 -m venv env
source env/bin/activate
```

4. Install Dependencies
```
pip install -r requirements.txt
```

5. Ingest Knowledge Base
```
python ingest.py
```

6. Run API
```
uvicorn app.main:app –reload
```

API: http://localhost:8000
Docs: http://localhost:8000/docs

---

## API Example
### Endpoint
```
POST /api/v1/questions
```
### Example Request
```
{
  "question": "What is RAG?",
  "top_k": 2
}
```
### Example Response
```
{
  "data": {
    "answer": "RAG stands for Retrieval-Augmented Generation. It is an architecture that combines information retrieval with large language models to improve answer accuracy. Instead of relying only on the model's internal knowledge, RAG retrieves relevant documents and provides them as context. In a typical RAG pipeline, a user question is first converted into an embedding, and then the system searches the vector database for similar documents.",
    "sources": [
      {
        "content": "Retrieval-Augmented Generation (RAG) is an architecture that combines information retrieval with large language models. Instead of relying only on the model’s internal knowledge, RAG retrieves relevant documents and provides them as context to improve answer accuracy.",
        "score": 0.44448313
      },
      {
        "content": "In a typical RAG pipeline, a user question is first converted into an embedding. The system then searches the vector database for similar documents.",
        "score": 0.34962848
      }
    ]
  },
  "meta": {
    "processing_time_ms": 48342
  }
}
```
### Error Response Format
```
{
  "error": {
    "code": 404,
    "message": "No relevant documents found",
    "details": "Vector search returned empty results"
  }
}
```

---
## Note
```
Note:
The processing_time_ms may vary depending on hardware performance, 
especially during the first model warm-up.
```