import os
from qdrant_client import QdrantClient, AsyncQdrantClient
from sentence_transformers import SentenceTransformer

# =============================
# ENV CONFIGURATION
# =============================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# =============================
# INFRASTRUCTURE INITIALIZATION
# =============================

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
async_qdrant_client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")