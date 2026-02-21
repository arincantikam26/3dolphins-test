import uuid
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import qdrant_client, embedding_model, COLLECTION_NAME


VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output size


def create_collection():
    print("Creating collection...")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )


def load_documents(filepath: str):
    print("Loading documents...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Simple paragraph-based chunking
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks


def ingest(filepath: str):
    create_collection()

    chunks = load_documents(filepath)

    points = []
    for chunk in chunks:
        vector = embedding_model.encode(chunk).tolist()

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk},
            )
        )

    print("Uploading vectors to Qdrant...")
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Ingestion completed. Inserted {len(points)} chunks.")


if __name__ == "__main__":
    ingest("data/knowledge.txt")