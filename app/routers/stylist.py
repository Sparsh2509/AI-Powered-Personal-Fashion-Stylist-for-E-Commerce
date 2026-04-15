"""
stylist.py — FastAPI router with all AI stylist endpoints.

ENDPOINTS:
  GET  /api/v1/stylist/health
       → Check if ChromaDB and Gemini are working

  POST /api/v1/stylist/analyze-and-recommend
       → Full pipeline: upload photo → get recommendation
       → Upload as multipart/form-data with file field "photo"

  POST /api/v1/stylist/recommend-from-profile
       → Skip vision step: send JSON profile → get recommendation
       → Useful for testing without a real photo

  POST /api/v1/stylist/analyze-only
       → Only run vision analysis, no recommendation
       → Useful for testing Gemini Vision separately

WHY SEPARATE ENDPOINTS?
  Your team can test each module independently.
  Backend team can also cache vision results and call
  recommend-from-profile without re-analyzing the photo.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.schemas import (
    ProfileOnlyRequest,
    FullAnalysisResponse,
    HealthResponse,
    ErrorResponse,
    RAGMetadata,
)
from ai_engine.vision_analyzer import analyze_photo
from ai_engine.recommender import generate_recommendation, run_full_pipeline
from config import settings

logger = logging.getLogger(__name__)

# Create router with /api/v1/stylist prefix (defined in main.py)
router = APIRouter()


# ================================================================
# ENDPOINT 1: HEALTH CHECK
# ================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check — verify ChromaDB and Gemini connectivity",
    tags=["System"],
)
async def health_check():
    """
    Check that ChromaDB is reachable and contains documents.
    Quick way for the backend team to verify the AI service is up.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = client.get_collection(name=settings.chroma_collection_name)
        doc_count = collection.count()
        chroma_ok = True
    except Exception as e:
        logger.warning(f"ChromaDB health check failed: {e}")
        doc_count = 0
        chroma_ok = False

    return HealthResponse(
        status="healthy" if chroma_ok else "degraded",
        version="1.0.0",
        chroma_db_connected=chroma_ok,
        chroma_document_count=doc_count,
        gemini_model=settings.gemini_llm_model,
    )


# ================================================================
# ENDPOINT 2: FULL PIPELINE (Photo → Recommendation)
# ================================================================

@router.post(
    "/analyze-and-recommend",
    response_model=FullAnalysisResponse,
    summary="Full pipeline: Upload photo and get personalized recommendation",
    tags=["AI Stylist"],
)
async def analyze_and_recommend(
    photo: UploadFile = File(
        ...,
        description="User's photo (JPEG or PNG). Should show face and body clearly."
    ),
    vision_prompt_version: str = Form(
        default="v2",
        description="Vision prompt version: v1, v2, or v3"
    ),
    n_rag_results: int = Form(
        default=5,
        description="Number of fashion rules to retrieve (1-10)"
    ),
):
    """
    **Full pipeline endpoint** — the main endpoint for the product.

    1. Receives the user's photo
    2. Sends to Gemini Vision API → extracts face shape, skin tone, body type
    3. Queries ChromaDB → retrieves relevant fashion rules
    4. Sends to Gemini LLM → generates personalized recommendation
    5. Returns structured JSON

    **How to call this (for backend team):**
    ```
    curl -X POST "http://localhost:8000/api/v1/stylist/analyze-and-recommend" \\
         -F "photo=@user_photo.jpg" \\
         -F "vision_prompt_version=v2" \\
         -F "n_rag_results=5"
    ```
    """
    # Validate file type
    if photo.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {photo.content_type}. Use JPEG or PNG."
        )

    logger.info(f"Received photo: {photo.filename} ({photo.content_type})")

    try:
        # Read the photo bytes
        image_bytes = await photo.read()

        # Run the full pipeline
        result = run_full_pipeline(
            image_source=image_bytes,
            n_rag_results=n_rag_results,
            vision_prompt_version=vision_prompt_version,
        )

        return FullAnalysisResponse(
            success=True,
            user_profile=result["user_profile"],
            rag_metadata=RAGMetadata(**result["rag_metadata"]),
            recommendation=result["recommendation"],
            model_used=result["model_used"],
        )

    except ValueError as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


# ================================================================
# ENDPOINT 3: RECOMMEND FROM PROFILE (Skip Vision Step)
# ================================================================

@router.post(
    "/recommend-from-profile",
    response_model=FullAnalysisResponse,
    summary="Get recommendation from an already-extracted profile (no photo needed)",
    tags=["AI Stylist"],
)
async def recommend_from_profile(request: ProfileOnlyRequest):
    """
    **Skip the vision step** — useful for:
    - Testing the RAG + LLM pipeline without a real photo
    - When the backend has already cached the vision analysis
    - Letting users manually input their profile

    Send a JSON body with `face_shape`, `skin_tone`, etc.
    """
    user_profile = {
        "face_shape": request.face_shape,
        "skin_tone": request.skin_tone,
        "skin_undertone": request.skin_undertone,
        "body_type": request.body_type,
    }

    logger.info(f"Recommend from profile: {user_profile}")

    try:
        result = generate_recommendation(
            user_profile=user_profile,
            n_rag_results=request.n_rag_results,
        )

        return FullAnalysisResponse(
            success=True,
            user_profile=result["user_profile"],
            rag_metadata=RAGMetadata(**result["rag_metadata"]),
            recommendation=result["recommendation"],
            model_used=result["model_used"],
        )

    except RuntimeError as e:
        # This usually means ChromaDB is not built yet
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ================================================================
# ENDPOINT 4: ANALYZE ONLY (Vision Analysis Only)
# ================================================================

@router.post(
    "/analyze-only",
    summary="Run only vision analysis — returns face shape, skin tone, body type",
    tags=["AI Stylist"],
)
async def analyze_only(
    photo: UploadFile = File(..., description="User's photo"),
    vision_prompt_version: str = Form(default="v2"),
):
    """
    **Vision only** — runs Gemini Vision API and returns the raw user profile.
    Does NOT run RAG or recommendation.

    Useful for:
    - Testing vision accuracy
    - Evaluation/benchmarking
    - When you want to show users their detected profile before recommendations
    """
    if photo.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {photo.content_type}"
        )

    try:
        image_bytes = await photo.read()
        profile = analyze_photo(image_bytes, prompt_version=vision_prompt_version)

        return {
            "success": True,
            "user_profile": profile,
            "model_used": settings.gemini_vision_model,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Vision analysis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
