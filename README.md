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

