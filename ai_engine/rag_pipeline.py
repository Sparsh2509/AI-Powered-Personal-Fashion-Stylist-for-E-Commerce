"""
rag_pipeline.py — ChromaDB retrieval logic (new google-genai SDK).
"""

import logging
from typing import Optional

import chromadb
from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

# ── Single shared Gemini client ────────────────────────────────────
_client = genai.Client(api_key=settings.google_api_key)


def _get_collection():
    """Connect to ChromaDB and return the fashion knowledge collection."""
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    try:
        return client.get_collection(name=settings.chroma_collection_name)
    except Exception:
        raise RuntimeError(
            f"ChromaDB collection '{settings.chroma_collection_name}' not found!\n"
            f"Please build it first: python knowledge_base/builder.py"
        )


def _embed_query(query_text: str) -> list[float]:
    """
    Embed a search query using Gemini Embedding API.

    Uses task_type RETRIEVAL_QUERY (different from RETRIEVAL_DOCUMENT
    used during indexing — this improves retrieval precision).
    """
    result = _client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=query_text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def build_search_query(
    face_shape: str,
    skin_tone: str,
    skin_undertone: str,
    body_type: str,
) -> str:
    """
    Convert a structured user profile into a semantic search query.

    Example:
        Input : face_shape="Round", skin_tone="Medium",
                skin_undertone="Warm", body_type="Pear"
        Output: "Fashion style rules for Round face shape. Color
                 recommendations for Medium skin with Warm undertone.
                 Clothing guidelines for Pear body type."
    """
    query = (
        f"Fashion style rules for {face_shape} face shape. "
        f"Color recommendations for {skin_tone} skin tone "
        f"with {skin_undertone} undertone. "
        f"Clothing and silhouette guidelines for {body_type} body type. "
        f"Best necklines, patterns, and fabrics for this profile."
    )
    logger.info(f"Search query: {query}")
    return query


def retrieve_fashion_rules(
    face_shape: str,
    skin_tone: str,
    skin_undertone: str,
    body_type: str,
    n_results: int = 5,
    category_filter: Optional[str] = None,
) -> dict:
    """
    Search ChromaDB for the most relevant fashion rules.

    Args:
        face_shape     : e.g. "Oval"
        skin_tone      : e.g. "Medium"
        skin_undertone : e.g. "Warm"
        body_type      : e.g. "Pear"
        n_results      : Top-k documents to return (default 5)
        category_filter: Optional metadata filter e.g. "face_shape"

    Returns:
        dict with keys: documents, ids, distances, metadatas,
                        query_used, n_retrieved
    """
    logger.info(
        f"Retrieving rules for: face={face_shape}, "
        f"skin={skin_tone}/{skin_undertone}, body={body_type}"
    )

    query_text = build_search_query(face_shape, skin_tone, skin_undertone, body_type)
    query_embedding = _embed_query(query_text)

    collection = _get_collection()

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if category_filter:
        query_params["where"] = {"category": category_filter}

    results = collection.query(**query_params)

    documents = results["documents"][0]
    distances = results["distances"][0]
    ids       = results["ids"][0]
    metadatas = results["metadatas"][0]

    logger.info(f"Retrieved {len(documents)} documents")
    for i, (doc_id, dist) in enumerate(zip(ids, distances)):
        logger.info(f"  [{i+1}] {doc_id} | distance: {dist:.4f}")

    return {
        "documents": documents,
        "ids": ids,
        "distances": distances,
        "metadatas": metadatas,
        "query_used": query_text,
        "n_retrieved": len(documents),
    }


def format_rules_for_prompt(retrieved: dict) -> str:
    """Format retrieved docs into a numbered string for the LLM prompt."""
    parts = []
    for i, (doc_id, text) in enumerate(zip(retrieved["ids"], retrieved["documents"])):
        parts.append(f"RULE {i+1} [{doc_id}]:\n{text.strip()}")
    return "\n\n".join(parts)


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    results = retrieve_fashion_rules(
        face_shape="Round", skin_tone="Medium",
        skin_undertone="Warm", body_type="Pear", n_results=4
    )
    for i, (doc_id, dist, doc) in enumerate(zip(
        results["ids"], results["distances"], results["documents"]
    )):
        print(f"\n--- Rule {i+1}: {doc_id} (dist={dist:.4f}) ---")
        print(doc[:200] + "...")
