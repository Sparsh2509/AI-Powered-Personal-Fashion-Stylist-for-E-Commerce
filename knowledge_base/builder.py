"""
builder.py — One-time script to build the ChromaDB knowledge base.
Uses the new google-genai SDK.

RUN ONCE before starting the API or Streamlit app:
  python knowledge_base/builder.py

For a full reset:
  python knowledge_base/builder.py --reset
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from google import genai
from google.genai import types

from config import settings
from knowledge_base.fashion_rules import (
    FASHION_DOCUMENTS,
    get_all_texts,
    get_all_ids,
    get_all_metadatas,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Gemini client ──────────────────────────────────────────────────
_client = genai.Client(api_key=settings.google_api_key)


def get_embedding(text: str) -> list[float]:
    """
    Convert text → embedding vector using Gemini text-embedding-004.

    Uses task_type RETRIEVAL_DOCUMENT (for items being stored/indexed).
    At query time we use RETRIEVAL_QUERY — the difference improves
    retrieval quality significantly.
    """
    result = _client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            title="Fashion rule document",
        ),
    )
    return result.embeddings[0].values


def build_knowledge_base(reset: bool = False) -> int:
    """
    Embed all fashion documents and store them in ChromaDB.

    Args:
        reset: If True, wipe the existing collection and rebuild.

    Returns:
        Total document count in the collection after build.
    """
    logger.info("=" * 60)
    logger.info("🏗️  Building Fashion Knowledge Base")
    logger.info("=" * 60)
    logger.info(f"  Embedding model : {settings.gemini_embedding_model}")
    logger.info(f"  ChromaDB path   : {settings.chroma_db_path}")
    logger.info(f"  Collection name : {settings.chroma_collection_name}")

    # Connect to ChromaDB (creates folder if needed)
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    logger.info("✓ ChromaDB connected")

    if reset:
        try:
            client.delete_collection(settings.chroma_collection_name)
            logger.info("✓ Old collection deleted (reset mode)")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"description": "Fashion rules knowledge base for AI stylist"},
    )
    logger.info(f"✓ Collection ready | current docs: {collection.count()}")

    texts     = get_all_texts()
    ids       = get_all_ids()
    metadatas = get_all_metadatas()

    logger.info(f"\n📄 Processing {len(texts)} fashion rule documents...")
    logger.info("-" * 60)

    embeddings_list = []
    for i, (doc_id, text, metadata) in enumerate(zip(ids, texts, metadatas)):
        # Skip docs that already exist (unless reset)
        if not reset and collection.get(ids=[doc_id])["ids"]:
            logger.info(f"  [{i+1}/{len(texts)}] SKIP (exists): {doc_id}")
            continue

        logger.info(f"  [{i+1}/{len(texts)}] Embedding: {doc_id}")

        embedding = get_embedding(text)
        embeddings_list.append({
            "id": doc_id,
            "embedding": embedding,
            "text": text,
            "metadata": metadata,
        })

        # Respect free-tier rate limit (~15 req/min)
        time.sleep(0.5)

    if embeddings_list:
        collection.add(
            ids=[e["id"] for e in embeddings_list],
            embeddings=[e["embedding"] for e in embeddings_list],
            documents=[e["text"] for e in embeddings_list],
            metadatas=[e["metadata"] for e in embeddings_list],
        )
        logger.info(f"\n✅ Added {len(embeddings_list)} documents to ChromaDB")
    else:
        logger.info("\n✅ All documents already in ChromaDB — nothing to add")

    final_count = collection.count()
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🎉 Knowledge Base Ready! Total documents: {final_count}")
    logger.info(f"{'=' * 60}")
    return final_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build ChromaDB fashion knowledge base")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing collection and rebuild")
    args = parser.parse_args()
    if args.reset:
        logger.info("⚠️  Reset mode: existing knowledge base will be deleted!")
    build_knowledge_base(reset=args.reset)
