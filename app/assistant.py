import os
import time
import json
from typing import List, Dict, Any

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


def search_documents(query: str, course: str = None, limit: int = 5) -> List[models.ScoredPoint]:
    """Perform RRF (Reciprocal Rank Fusion) hybrid search combining dense and sparse vectors."""
    embedding_model = get_embedding_model()
    query_embedding = list(embedding_model.embed([query]))[0]
    
    search_filter = None
    if course:
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="course",
                    match=models.MatchValue(value=course)
                )
            ]
        )
    
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
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                using="sparse",
                limit=limit * 5,
                filter=search_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    
    return results.points


def build_prompt(query: str, search_results: List[models.ScoredPoint]) -> str:
    """Build prompt for LLM with context from search results."""
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = "\n\n".join([
        f"section: {result.payload.get('section', 'N/A')}\n"
        f"question: {result.payload.get('question', 'N/A')}\n"
        f"answer: {result.payload.get('text', 'N/A')}"
        for result in search_results
    ])
    
    return prompt_template.format(question=query, context=context).strip()


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


def get_answer(query: str, course: str = None, model_choice: str = "openai/gpt-4o-mini", 
               search_limit: int = 5, evaluate: bool = False) -> Dict[str, Any]:
    """Get answer for a query using RAG pipeline."""
    search_results = search_documents(query, course, search_limit)
    
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
            'score': float(result.score),
            'course': result.payload.get('course', 'N/A'),
            'section': result.payload.get('section', 'N/A'),
            'question': result.payload.get('question', 'N/A'),
            'text': result.payload.get('text', 'N/A')
        }
        for result in search_results
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