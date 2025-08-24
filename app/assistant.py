import os
import time
import json
from typing import List, Dict, Any, Optional

from openai import OpenAI
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.getenv(
    "JSONL_PATH",
    os.path.join(BASE_DIR, "data", "adhd_guideline.jsonl")
)
MAX_EXTRACT_CHARS = 1200
TOPK_USE = None  

# Initialize clients
qdrant_client = QdrantClient(QDRANT_URL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Load embedding model (lazy loading)
_embedding_model = None

def get_embedding_model():
    """Get embedding model with lazy loading."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    return _embedding_model


def _with_retrievable_filter(
    base_must: Optional[list],
    include_unretrievable: bool
) -> list:
    must = list(base_must or [])
    if not include_unretrievable:
        must.append(models.FieldCondition(
            key="retrievable",
            match=models.MatchValue(value=True),
        ))
    return must

def search_documents(
    query: str,
    *,
    section_title: Optional[str] = None,        # e.g. "Baseline assessment"
    tags: Optional[List[str]] = None,           # e.g. ["medication"]
    population: Optional[List[str]] = None,     # e.g. ["adults"]
    limit: int = 5,
    include_unretrievable: bool = False,        # 管理员开关
) -> List[models.ScoredPoint]:
    """RRF (dense+sparse) 混合检索，按新payload字段过滤。"""
    embedding_model = get_embedding_model()
    query_embedding = list(embedding_model.embed([query]))[0]

    must = []

    if section_title:
        must.append(models.FieldCondition(
            key="section_title",
            match=models.MatchValue(value=section_title),
        ))

    if tags:
        must.append(models.FieldCondition(
            key="tags",
            match=models.MatchAny(any=tags),
        ))

    if population:
        must.append(models.FieldCondition(
            key="population",
            match=models.MatchAny(any=population),
        ))

    must = _with_retrievable_filter(must, include_unretrievable)
    search_filter = models.Filter(must=must) if must else None

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=query_embedding.tolist(),
                using="dense",
                limit=limit * 5,
                filter=search_filter,
            ),
            models.Prefetch(
                query=models.Document(text=query, model="Qdrant/bm25"),
                using="sparse",
                limit=limit * 5,
                filter=search_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return results.points

_DOC_STORE = None  # id -> doc(dict)

def _load_doc_store(path: str):
    global _DOC_STORE
    if _DOC_STORE is not None:
        return _DOC_STORE
    store = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            _id = obj.get("id")
            if _id:
                store[_id] = obj
    _DOC_STORE = store
    return _DOC_STORE

def _get_doc_by_id(pid: str):
    store = _load_doc_store(JSONL_PATH)
    return store.get(pid)

def build_text(doc: dict) -> str:

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


def _extract_body(pid: str, limit: int = MAX_EXTRACT_CHARS) -> str:
    doc = _get_doc_by_id(pid)
    if not doc:
        return ""
    try:
        full = build_text(doc)  # 你当前的 build_text：标题+主体，原样，不清洗、不chunk
    except Exception:
        # 兜底：尽力拼一点正文
        full = (doc.get("text") or doc.get("content") or "") or ""
        if isinstance(doc.get("bullets"), list):
            full = f"{full}\n" + "\n".join(str(x) for x in doc["bullets"] if x is not None)
    full = full if isinstance(full, str) else str(full)
    if limit and len(full) > limit:
        return full[:limit].rstrip() + " …"
    return full

def build_prompt(query: str, search_results: List[models.ScoredPoint]) -> str:
    """
    构建 LLM 提示词（能从 payload 拿的都用 payload；正文仅回 JSONL 一次查表）。
    """
    prompt_template = (
        "You are a psychologist specialized in ADHD.\n"
        "Answer the QUESTION based on the CONTEXT from the reference database.\n"
        "Use only the facts from the CONTEXT when answering the QUESTION.\n\n"
        "QUESTION: {question}\n\n"
        "CONTEXT:\n{context}\n"
    )

    blocks = []
    pts = search_results if TOPK_USE is None else search_results[:TOPK_USE]

    for r in pts:
        pid = getattr(r, "id", None) or (r.get("id") if isinstance(r, dict) else None)
        payload = getattr(r, "payload", None) or (r.get("payload") if isinstance(r, dict) else {}) or {}
        if not pid:
            continue

        section_title = payload.get("section_title")
        source = payload.get("source")
        dates = payload.get("dates", {})
        
        # 处理日期信息
        date_info = ""
        if isinstance(dates, dict):
            if dates.get("last_updated"):
                date_info = f"  (updated: {dates['last_updated']})"
            elif dates.get("published"):
                date_info = f"  (published: {dates['published']})"
                
        extract = _extract_body(pid, limit=MAX_EXTRACT_CHARS)

        # 如果没取到正文，就只展示题头元信息
        if not extract:
            extract = "[No extract available in payload; body not found in JSONL]"

        block = (
            f"[TITLE] {section_title}\n"
            f"[SOURCE] {source}{date_info}\n"
            f"[EXTRACT]\n{extract}"
        )
        blocks.append(block)

    context = "\n\n---\n\n".join(blocks) if blocks else "No context."
    return prompt_template.format(question=query.strip(), context=context).strip()


def llm(prompt: str, model_choice: str) -> tuple[str, dict, float]:
    """Call OpenAI LLM with the given prompt."""
    start_time = time.time()
    try:
        if model_choice.startswith('openai/'):
            model_name = model_choice.split('/')[-1]
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
            tokens = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"Error: {str(e)}", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, 0.0
    
    end_time = time.time()
    response_time = end_time - start_time
    
    return answer, tokens, response_time


def evaluate_relevance(question: str, answer: str) -> tuple[str, str, dict]:
    """Evaluate the relevance of the generated answer to the question."""
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)
    evaluation, tokens, _ = llm(prompt, 'openai/gpt-4o-mini')
    
    try:
        json_eval = json.loads(evaluation)
        return json_eval.get('Relevance', 'UNKNOWN'), json_eval.get('Explanation', 'No explanation'), tokens
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation", tokens


def calculate_openai_cost(model_choice: str, tokens: dict) -> float:
    """Calculate the cost for OpenAI API usage."""
    if not model_choice.startswith('openai/'):
        return 0.0
    
    model_name = model_choice.split('/')[-1]
    prompt_tokens = tokens.get('prompt_tokens', 0)
    completion_tokens = tokens.get('completion_tokens', 0)
    
    if model_name == 'gpt-3.5-turbo':
        cost = (prompt_tokens * 0.0015 + completion_tokens * 0.002) / 1000
    elif model_name in ['gpt-4o', 'gpt-4o-mini']:
        cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
    elif model_name.startswith('gpt-4'):
        cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
    else:
        cost = 0.0
    
    return cost


def get_answer(query: str, section_title: str = None, tags: List[str] = None, 
               population: List[str] = None, model_choice: str = "openai/gpt-4o-mini", 
               search_limit: int = 5, evaluate: bool = False) -> Dict[str, Any]:
    """Get answer for a query using RAG pipeline."""
    search_results = search_documents(
        query, 
        section_title=section_title,
        tags=tags,
        population=population,
        limit=search_limit
    )
    
    if not search_results:
        return {
            'answer': 'Sorry, I could not find any relevant information to answer your question.',
            'search_results_count': 0,
            'response_time': 0,
            'relevance': 'NON_RELEVANT',
            'relevance_explanation': 'No search results found',
            'model_used': model_choice,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'openai_cost': 0
        }
    
    prompt = build_prompt(query, search_results)
    answer, tokens, response_time = llm(prompt, model_choice)
    
    result = {
        'answer': answer,
        'search_results_count': len(search_results),
        'response_time': response_time,
        'model_used': model_choice,
        'prompt_tokens': tokens['prompt_tokens'],
        'completion_tokens': tokens['completion_tokens'],
        'total_tokens': tokens['total_tokens'],
        'openai_cost': calculate_openai_cost(model_choice, tokens)
    }
    
    result['search_results'] = [
        {
            'score': float(r.score),
            'id': r.id,
            'source': r.payload.get('source', ''),
            'section_title': r.payload.get('section_title', ''),
            'title_path': r.payload.get('title_path', ''),
            'tags': r.payload.get('tags', []),
            'population': r.payload.get('population', []),
            'retrievable': r.payload.get('retrievable', True)
        }
        for r in search_results
    ]
    
    if evaluate:
        relevance, explanation, eval_tokens = evaluate_relevance(query, answer)
        result.update({
            'relevance': relevance,
            'relevance_explanation': explanation,
            'eval_prompt_tokens': eval_tokens['prompt_tokens'],
            'eval_completion_tokens': eval_tokens['completion_tokens'],
            'eval_total_tokens': eval_tokens['total_tokens'],
            'openai_cost': result['openai_cost'] + calculate_openai_cost('openai/gpt-4o-mini', eval_tokens)
        })
    
    return result