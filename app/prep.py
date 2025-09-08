import os
import re
import json
import uuid
from typing import Iterable, Dict, Any, List, Union

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
    os.path.join(BASE_DIR, "data", "adhd_guideline.jsonl")
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
    构建向量化文本（极简版）：
    1) 标题：优先使用 title_path；否则用 section_id + section_title（原样，不清洗）
    2) 主体：text 或 content（原样）+ bullets/notes/recommendations（原样逐条拼接）
    """
    # 1) 标题（title_path -> section_id + section_title）
    tp = doc.get("title_path")
    if isinstance(tp, str) and tp.strip():
        headline = tp.strip()
    else:
        sid = doc.get("section_id")
        st  = doc.get("section_title")
        # 仅拼接已有字段（不做分隔符拆分）
        headline = " ".join([str(x) for x in (sid, st) if isinstance(x, str) and x.strip()]).strip()

    # 2) 主体（原样）
    parts = []
    body_main = doc.get("text") or doc.get("content") or ""
    if isinstance(body_main, str) and body_main:
        parts.append(body_main)

    for key in ("bullets", "notes", "recommendations"):
        val = doc.get(key)
        if isinstance(val, list) and val:
            parts.append("\n".join([str(x) for x in val if x is not None]))
        elif isinstance(val, str) and val:
            parts.append(val)

    body = "\n".join(parts).strip()

    # 3) 合并
    combined = f"{headline}\n{body}".strip() if headline else body
    return combined

def build_metadata(doc: dict) -> dict:
    # 保持原样放进 payload，供过滤/排序/展示用
    return {
        "id": doc.get("id"),
        "doc_type": doc.get("doc_type"),
        "source": doc.get("source"),
        "dates": doc.get("dates"),
        "section_id": doc.get("section_id"),
        "section_title": doc.get("section_title"),
        "title_path": doc.get("title_path"),
        "population": doc.get("population") or [],
        "tags": doc.get("tags") or [],
        "retrievable": doc.get("retrievable"),
    }
    
# -------------------------
# Batched embedding + upsert
# -------------------------
def index_documents(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    docs_stream: Iterable[Dict[str, Any]],
    batch_size: int = 256,
    collection_name: str = None,
):
    if collection_name is None:
        collection_name = COLLECTION_NAME
        
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
            meta = build_metadata(d)
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
                    **meta,  # 元数据用于过滤/排序/展示
                },
            )
            points.append(point)

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    total = 0
    for doc in tqdm(docs_stream, desc="Reading JSONL"):
        meta = build_metadata(doc)
        
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
