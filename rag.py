import json
import math
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DB_PATH = BASE_DIR / "rag_data" / "rag.db"

load_dotenv(BASE_DIR / ".env")

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dimensions INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_source
            ON rag_chunks(source)
            """
        )


def split_text(text, chunk_size=260, overlap=40):
    text = text.replace("\r\n", "\n").strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= chunk_size:
            current += "\n\n" + para
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap and len(current) > overlap else current
            current = tail + "\n\n" + para

    if current:
        chunks.append(current)

    return chunks


def embed_text(text):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSIONS,
        encoding_format="float",
    )
    return resp.data[0].embedding


def cosine_similarity(a, b):
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def rebuild_index():
    init_db()

    files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not files:
        raise RuntimeError(f"no markdown files found in {KNOWLEDGE_DIR}")

    with get_conn() as conn:
        conn.execute("DELETE FROM rag_chunks")

        total = 0
        for path in files:
            text = path.read_text(encoding="utf-8")
            chunks = split_text(text)

            for index, chunk in enumerate(chunks):
                embedding = embed_text(chunk)
                conn.execute(
                    """
                    INSERT INTO rag_chunks
                    (source, chunk_index, content, embedding, embedding_model, embedding_dimensions)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        path.name,
                        index,
                        chunk,
                        json.dumps(embedding, ensure_ascii=False),
                        EMBEDDING_MODEL,
                        EMBEDDING_DIMENSIONS,
                    ),
                )
                total += 1

        conn.commit()

    return total



CITY_SOURCE_MAP = {
    "北京": "beijing.md",
    "北京市": "beijing.md",
    "上海": "shanghai.md",
    "上海市": "shanghai.md",
    "成都": "chengdu.md",
    "成都市": "chengdu.md",
}


def search_travel_guide(query, city="", top_k=3):
    init_db()
    query_embedding = embed_text(query)

    source_filter = CITY_SOURCE_MAP.get((city or "").strip())

    sql = """
        SELECT source, chunk_index, content, embedding
        FROM rag_chunks
        WHERE embedding_model = ?
          AND embedding_dimensions = ?
    """
    params = [EMBEDDING_MODEL, EMBEDDING_DIMENSIONS]

    if source_filter:
        sql += " AND source = ?"
        params.append(source_filter)

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    scored = []
    for source, chunk_index, content, embedding_json in rows:
        embedding = json.loads(embedding_json)
        score = cosine_similarity(query_embedding, embedding)
        scored.append(
            {
                "source": source,
                "chunk_index": chunk_index,
                "score": score,
                "content": content,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def format_search_results(results):
    if not results:
        return "没有检索到相关旅行攻略。"

    lines = []
    for index, item in enumerate(results, 1):
        lines.append(
            f"{index}. 来源：{item['source']}，相似度：{item['score']:.4f}\n{item['content']}"
        )

    return "\n\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        total = rebuild_index()
        print(f"rag index rebuilt, chunks={total}")
    elif len(sys.argv) >= 3 and sys.argv[1] == "search":
        city = ""
        args = sys.argv[2:]

        if len(args) >= 2 and args[0] == "--city":
            city = args[1]
            args = args[2:]

        query = " ".join(args)
        results = search_travel_guide(query=query, city=city, top_k=3)
        print(format_search_results(results))
    else:
        print("usage:")
        print("  python3 rag.py build")
        print("  python3 rag.py search 北京雨天适合去哪")
        print("  python3 rag.py search --city 北京 雨天适合去哪")
