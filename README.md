### LLM-RAG Course Assistant

A lightweight RAG-based Q&A chatbot for course materials, built with:

Streamlit frontend

OpenAI API for LLMs

Qdrant for vector search

Postgres for storing conversations and feedback

## 🚀 Features

Select different OpenAI models (e.g., gpt-3.5-turbo, gpt-4o, etc.)

Hybrid search with Qdrant (dense + sparse vectors)

Conversation history stored in Postgres

Feedback collection (+1 / -1) for model evaluation

Simple monitoring via feedback stats

## 📦 Requirements

Docker & Docker Compose

An OpenAI API Key

## Quick test plan

1) Build & start services
```
bash

cd app
docker compose up -d --build
# (optional) follow logs
docker compose logs -f
```


You should see containers for qdrant, postgres, and streamlit come up.

2) Initialize database tables (one‑time)
```
bash
docker compose exec streamlit python -c 'from db import init_db; init_db(drop_existing=False); print("DB ready")'
```

This creates conversations and feedback tables in Postgres.

3) Verify env vars are injected into the container
```
bash
docker compose exec streamlit env | egrep 'OPENAI_API_KEY|QDRANT_URL|POSTGRES_HOST|COLLECTION_NAME|DENSE_MODEL_NAME'
```

You should see non-empty values for:

OPENAI_API_KEY

QDRANT_URL (e.g. http://qdrant:6333)

POSTGRES_HOST (e.g. postgres)

COLLECTION_NAME

DENSE_MODEL_NAME

If OPENAI_API_KEY is empty, ensure your .env is in app/ (or start with --env-file ../.env) and recreate the container:

```
bash
docker compose up -d --build --force-recreate streamlit
```

4) Confirm Qdrant collection exists and has data

First time? Ingest documents:

```
bash
# indexes local data/ into Qdrant (dense + sparse), and ensures DB schema exists
docker compose exec streamlit python prep.py
```

Check collection status & point count:

```
bash
docker compose exec streamlit python - <<'PY'
import os
from qdrant_client import QdrantClient

url = os.getenv("QDRANT_URL")
cn  = os.getenv("COLLECTION_NAME")
c = QdrantClient(url=url)

try:
    info = c.get_collection(cn)
    print("Collection status:", getattr(info, "status", "unknown"))
    # vectors_count may not be instantly updated; count() is definitive
    print("Point count:", c.count(cn).count)
except Exception as e:
    print("Qdrant check failed:", e)
PY
```

You should see a positive “Point count”. If it’s 0, rerun prep.py and re-check.

5) Open the frontend and test

Codespaces: open the Ports panel and click the forwarded 8501 link (set visibility Public/Private as needed).

Local Docker: open http://localhost:8501

In the UI:

Choose an OpenAI model (e.g., openai/gpt-4o-mini)

Choose your topic and target group

Ask a question related to your indexed data

See answer, response time, tokens, and (if enabled) relevance evaluation

Give feedback (+1/−1) and check it shows up in “Recent Conversations” & “Feedback Statistics”