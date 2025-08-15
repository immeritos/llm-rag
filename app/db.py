import os
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from zoneinfo import ZoneInfo

tz = ZoneInfo("Europe/Berlin")


def get_db_connection():
    """Create a PostgreSQL connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "course_assistant"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )


def init_db(drop_existing=False):
    """Initialize the database schema."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if drop_existing:
                cur.execute("DROP TABLE IF EXISTS feedback")
                cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    course TEXT,
                    model_used TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    relevance TEXT,
                    relevance_explanation TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    eval_prompt_tokens INTEGER,
                    eval_completion_tokens INTEGER,
                    eval_total_tokens INTEGER,
                    openai_cost FLOAT,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
    finally:
        conn.close()


def save_conversation(conversation_id, question, answer_data, course=None, timestamp=None):
    """Save a conversation record."""
    if timestamp is None:
        timestamp = datetime.now(tz)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations 
                (id, question, answer, course, model_used, response_time, relevance, relevance_explanation, 
                prompt_tokens, completion_tokens, total_tokens, eval_prompt_tokens, eval_completion_tokens, 
                eval_total_tokens, openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                conversation_id,
                question,
                answer_data.get("answer"),
                course,
                answer_data.get("model_used"),
                answer_data.get("response_time"),
                answer_data.get("relevance"),
                answer_data.get("relevance_explanation"),
                answer_data.get("prompt_tokens"),
                answer_data.get("completion_tokens"),
                answer_data.get("total_tokens"),
                answer_data.get("eval_prompt_tokens"),
                answer_data.get("eval_completion_tokens"),
                answer_data.get("eval_total_tokens"),
                answer_data.get("openai_cost"),
                timestamp
            ))
        conn.commit()
    finally:
        conn.close()


def save_feedback(conversation_id, feedback, timestamp=None):
    """Save feedback for a conversation."""
    if timestamp is None:
        timestamp = datetime.now(tz)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feedback (conversation_id, feedback, timestamp)
                VALUES (%s, %s, %s)
            """, (conversation_id, feedback, timestamp))
        conn.commit()
    finally:
        conn.close()


def get_recent_conversations(limit=5, relevance=None):
    """Retrieve recent conversations with optional relevance filter."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = """
                SELECT c.*, f.feedback
                FROM conversations c
                LEFT JOIN feedback f ON c.id = f.conversation_id
            """
            params = []
            if relevance:
                query += " WHERE c.relevance = %s"
                params.append(relevance)
            query += " ORDER BY c.timestamp DESC LIMIT %s"
            params.append(limit)
            cur.execute(query, tuple(params))
            return cur.fetchall()
    finally:
        conn.close()


def get_feedback_stats():
    """Get aggregated feedback statistics."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN feedback > 0 THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN feedback < 0 THEN 1 ELSE 0 END) as thumbs_down
                FROM feedback
            """)
            return cur.fetchone()
    finally:
        conn.close()
