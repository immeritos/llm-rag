import os
import json
import requests
import pandas as pd
from io import StringIO
import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from tqdm.auto import tqdm
from dotenv import load_dotenv

from db import init_db

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# BASE_URL = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOCS_PATH = os.path.join(DATA_DIR, "documents-with-ids.json")
GT_PATH = os.path.join(DATA_DIR, "ground-truth-data.csv")

def fetch_documents():
    print(f"Loading documents from {DOCS_PATH} ...")
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"Missing file: {DOCS_PATH}")
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} documents (local)")
    return documents

def fetch_ground_truth():
    print(f"Loading ground truth from {GT_PATH} ...")
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Missing file: {GT_PATH}")
    df = pd.read_csv(GT_PATH)
    df = df[df.course == "machine-learning-zoomcamp"]
    gt = df.to_dict(orient="records")
    print(f"Loaded {len(gt)} ground truth records (local)")
    return gt


def load_embedding_model():
    """Load FastEmbed text embedding model."""
    print(f"Loading embedding model: {DENSE_MODEL_NAME}")
    return TextEmbedding(model_name=DENSE_MODEL_NAME)


def setup_qdrant():
    """Setup Qdrant client and create collection."""
    print("Setting up Qdrant...")
    client = QdrantClient(QDRANT_URL)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass  # Collection might not exist
    
    # Get vector dimension from the model
    embedding_model = load_embedding_model()
    
    # Create a test embedding to get the dimension
    test_text = "test"
    test_embedding = list(embedding_model.embed([test_text]))[0]
    vector_dim = len(test_embedding)
    
    print(f"Vector dimension: {vector_dim}")
    
    # Create collection with both dense and sparse vectors
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            # Dense vector for semantic search
            "dense": models.VectorParams(
                size=vector_dim,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            # Sparse vector for BM25-like search
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )
    
    print(f"Qdrant collection '{COLLECTION_NAME}' created with hybrid search support")
    return client, embedding_model


def index_documents(client, documents, embedding_model, batch_size: int = 256):
    """Index documents into Qdrant collection (batched upsert)."""
    print("Indexing documents...")

    points = []
    count = 0

    for doc in tqdm(documents):
        # Combine question and text for better embeddings
        combined_text = f"{doc.get('question', '')} {doc.get('text', '')}"

        # Generate dense embedding
        dense_embedding = list(embedding_model.embed([combined_text]))[0]

        # Create point (你的 JSON 里 id 已是 UUID；保留你的原始写法)
        point = models.PointStruct(
            id=doc.get("id", str(uuid.uuid4())),  # 注意：用 str(uuid.uuid4())（带连字符）更标准
            vector={
                "dense": dense_embedding.tolist(),
                "sparse": models.Document(
                    text=combined_text,
                    model="Qdrant/bm25",
                ),
            },
            payload={
                "text": doc.get("text", ""),
                "section": doc.get("section", ""),
                "question": doc.get("question", ""),
                "course": doc.get("course", ""),
                "id": doc.get("id", ""),
            },
        )
        points.append(point)

        # 批量 upsert
        if len(points) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            count += len(points)
            points = []

    # 提交剩余未满一批的 points
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        count += len(points)

    print(f"Indexed {count} documents")





def main():
    """Main function to run the indexing process."""
    print("Starting the indexing process with Qdrant...")

    # Fetch data
    documents = fetch_documents()
    ground_truth = fetch_ground_truth()
    
    # Setup Qdrant and embedding model
    client, embedding_model = setup_qdrant()
    
    # Index documents
    index_documents(client, documents, embedding_model)
    
    print("Initializing database...")
    init_db()
    
    print("Indexing process completed successfully!")
    print(f"Collection '{COLLECTION_NAME}' is ready for queries")



if __name__ == "__main__":
    main()