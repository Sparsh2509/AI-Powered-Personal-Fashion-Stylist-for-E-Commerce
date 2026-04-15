"""
recommender.py — Final recommendation generator (new google-genai SDK).
Orchestrates the full RAG pipeline:
  vision_analyzer → rag_pipeline → Gemini LLM → structured JSON
"""

import re
import json
import logging

from google import genai

from config import settings
from ai_engine.prompt_templates import RecommendationPrompts
from ai_engine.rag_pipeline import retrieve_fashion_rules, format_rules_for_prompt
from ai_engine.vision_analyzer import analyze_photo

logger = logging.getLogger(__name__)

# ── Single shared Gemini client ────────────────────────────────────
_client = genai.Client(api_key=settings.google_api_key)


def _parse_recommendation_response(raw_text: str) -> dict:
    """Strip markdown fences and parse JSON from Gemini LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    cleaned = cleaned.replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse recommendation JSON: {e}")
        raise ValueError(f"Gemini gave non-JSON output. Raw: {raw_text[:400]}")


def generate_recommendation(user_profile: dict, n_rag_results: int = 5) -> dict:
    """
    Generate a fashion recommendation from an already-extracted user profile.

    This is STEP 2+3 of the RAG pipeline (retrieval + generation).

    Args:
        user_profile  : dict from analyze_photo(), with face_shape,
                        skin_tone, skin_undertone, body_type fields.
        n_rag_results : Number of ChromaDB rules to retrieve (default 5).

    Returns:
        dict with: user_profile, rag_metadata, recommendation, model_used
    """
    logger.info("Generating fashion recommendation...")

    # Step 1 — Retrieve relevant rules from ChromaDB (RAG)
    retrieved = retrieve_fashion_rules(
        face_shape     = user_profile.get("face_shape", "unknown"),
        skin_tone      = user_profile.get("skin_tone", "unknown"),
        skin_undertone = user_profile.get("skin_undertone", "unknown"),
        body_type      = user_profile.get("body_type", "unknown"),
        n_results      = n_rag_results,
    )
    logger.info(f"Retrieved {retrieved['n_retrieved']} rules from ChromaDB")

    # Step 2 — Build the prompt: profile + retrieved rules
    profile_summary = (
        f"Face Shape: {user_profile.get('face_shape', 'Unknown')}\n"
        f"Skin Tone: {user_profile.get('skin_tone', 'Unknown')} "
        f"({user_profile.get('skin_undertone', 'Unknown')} undertone)\n"
        f"Body Type: {user_profile.get('body_type', 'Unknown')}"
    )
    formatted_rules = format_rules_for_prompt(retrieved)

    prompt = RecommendationPrompts.ACTIVE.format(
        user_profile  = profile_summary,
        fashion_rules = formatted_rules,
    )

    # Step 3 — Send to Gemini LLM (text-only, new SDK)
    logger.info(f"Sending to Gemini LLM ({settings.gemini_llm_model})...")
    response = _client.models.generate_content(
        model=settings.gemini_llm_model,
        contents=prompt,
    )

    raw_text = response.text
    logger.info("Received recommendation from Gemini")

    recommendation = _parse_recommendation_response(raw_text)

    return {
        "user_profile": user_profile,
        "rag_metadata": {
            "query_used"        : retrieved["query_used"],
            "n_rules_retrieved" : retrieved["n_retrieved"],
            "rule_ids_used"     : retrieved["ids"],
            "retrieval_distances": retrieved["distances"],
        },
        "recommendation": recommendation,
        "model_used": settings.gemini_llm_model,
    }


def run_full_pipeline(
    image_source,
    n_rag_results: int = 5,
    vision_prompt_version: str = "v2",
) -> dict:
    """
    Run the complete end-to-end pipeline from photo to recommendation.

    Steps:
      1. Gemini Vision  → analyze_photo()
      2. ChromaDB RAG   → retrieve_fashion_rules()
      3. Gemini LLM     → generate_recommendation()

    Args:
        image_source          : File path (str) or raw bytes.
        n_rag_results         : How many ChromaDB rules to retrieve.
        vision_prompt_version : "v1", "v2" (default), or "v3".

    Returns:
        Full result dict (profile + rag_metadata + recommendation).

    Example:
        >>> result = run_full_pipeline("photo.jpg")
        >>> result["user_profile"]["face_shape"]   # "Oval"
        >>> result["recommendation"]["color_palette"]
    """
    logger.info("🚀 Starting full fashion stylist pipeline...")

    # STEP 1: Vision analysis
    logger.info("--- STEP 1: Vision Analysis ---")
    user_profile = analyze_photo(image_source, prompt_version=vision_prompt_version)

    # STEP 2+3: RAG retrieval + LLM recommendation
    logger.info("--- STEP 2+3: RAG + Generation ---")
    result = generate_recommendation(user_profile, n_rag_results=n_rag_results)

    logger.info("🎉 Full pipeline complete!")
    return result


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    if not test_image:
        print("Usage: python ai_engine/recommender.py <path_to_image>")
        sys.exit(1)

    print(f"\n🚀 Running full pipeline on: {test_image}")
    result = run_full_pipeline(test_image)
    print("\n✅ FULL RESULT:")
    print(json.dumps(result, indent=2))
