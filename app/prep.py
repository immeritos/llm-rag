import os
import json
import uuid
from typing import Iterable, Dict, Any, List

from tqdm.auto import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # 可为空（本地Qdrant）
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_JSONL = os.getenv(
    "DOCS_JSONL",
    os.path.join(BASE_DIR, "data", "adhd_guideline_preprocessed.jsonl")
)

print(">>> DEBUG: DOCS_JSONL resolved to:", DOCS_JSONL)
# -------------------------
# Data loading
# -------------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# -------------------------
# Embedding model
# -------------------------
def load_embedding_model() -> TextEmbedding:
    print(f"Loading FastEmbed model: {DENSE_MODEL_NAME}")
    return TextEmbedding(model_name=DENSE_MODEL_NAME)

# -------------------------
# Qdrant setup
# -------------------------
def setup_qdrant(embedding_model: TextEmbedding) -> QdrantClient:
    print("Setting up Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    # 通过一次前向得到维度（FastEmbed不一定暴露属性）
    test_vec = list(embedding_model.embed(["test"]))[0]
    vector_dim = len(test_vec)
    print(f"Dense vector dimension: {vector_dim}")

    # 重新创建集合（幂等）
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=vector_dim,
                distance=models.Distance.COSINE,
            ),
        },
        # 使用 Qdrant 服务器端的 BM25 稀疏向量（混合检索）
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,  # BM25/IDF 家族
            )
        },
    )
    print(f"Qdrant collection '{COLLECTION_NAME}' is ready (dense + sparse).")
    return client

# -------------------------
# Utility: build text for embedding
# -------------------------
def build_text(doc: dict) -> str:
    """
    组合用于向量化的文本：
    - 优先使用 highlighted_text，其次 text
    - 在前面拼上 breadcrumb / section 作为轻量上下文
    """
    headline = " ".join([x for x in [doc.get("breadcrumb", ""), doc.get("section", "")] if x]).strip()
    body = (doc.get("highlighted_text") or doc.get("text") or "").strip()
    combined = f"{headline}\n{body}".strip() if headline else body
    return combined

# -------------------------
# Batched embedding + upsert
# -------------------------
def index_documents(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    docs_stream: Iterable[Dict[str, Any]],
    batch_size: int = 256,
):
    print("Indexing documents...")

    buffer_docs: List[Dict[str, Any]] = []

    def flush(batch: List[Dict[str, Any]]):
        if not batch:
            return
        texts = [d["_combined_text"] for d in batch]

        # 批量生成dense向量（FastEmbed为生成器，转 list 保证可重用）
        dense_embeds = list(embedding_model.embed(texts))

        points = []
        for d, vec in zip(batch, dense_embeds):
            # 规范 id：尽量使用原始 id，否则生成 uuid4
            pid = d.get("id") or str(uuid.uuid4())

            # 稀疏向量采用服务器端BM25：直接给 Document 即可
            point = models.PointStruct(
                id=pid,
                vector={
                    "dense": vec.tolist(),
                    "sparse": models.Document(
                        text=d["_combined_text"],
                        model="Qdrant/bm25",
                    ),
                },
                payload={
                    # 主文本：保留原始 text 和 highlighted_text（若存在）
                    "text": doc.get("text", ""),
                    "highlighted_text": doc.get("highlighted_text", ""),

                    # 文档位置信息
                    "source": doc.get("source", ""),               # e.g., "adhd_guideline"
                    "section": doc.get("section", ""),             # e.g., "Overview"
                    "breadcrumb": doc.get("breadcrumb", ""),       # e.g., "Overview"
                    "page_start": doc.get("page_start", None),
                    "page_end": doc.get("page_end", None),

                    # 辅助标签/参考
                    "side_labels": doc.get("side_labels", []),     # list
                    "refs": doc.get("refs", []),                   # list

                    # 便于追踪：原始 id + 用于检索融合的 combined_text（可选）
                    "source_id": doc.get("id", ""),
                    "combined_text": doc.get("_combined_text", ""),  # 可在调试/检索分析时查看
                },

            )
            points.append(point)

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    total = 0
    for doc in tqdm(docs_stream, desc="Reading JSONL"):
        combined = build_text(doc)
        doc["_combined_text"] = combined
        buffer_docs.append(doc)

        if len(buffer_docs) >= batch_size:
            flush(buffer_docs)
            total += len(buffer_docs)
            buffer_docs.clear()

    if buffer_docs:
        flush(buffer_docs)
        total += len(buffer_docs)

    print(f"Indexed {total} documents.")

# -------------------------
# Main
# -------------------------
def main():
    print(f"Loading JSONL from: {DOCS_JSONL}")
    docs = read_jsonl(DOCS_JSONL)
    embedding_model = load_embedding_model()
    client = setup_qdrant(embedding_model)
    index_documents(client, embedding_model, docs, batch_size=256)
    print("Done. Collection is ready for queries.")

if __name__ == "__main__":
    main()
