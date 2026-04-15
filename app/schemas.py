"""
schemas.py — Pydantic models for FastAPI request and response validation.

WHY PYDANTIC:
  FastAPI uses Pydantic to auto-validate incoming request data.
  If a required field is missing or has wrong type, FastAPI 
  automatically returns a 422 error with a clear message — 
  you don't have to write any validation code manually.

  It also auto-generates Swagger documentation at /docs.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ================================================================
# REQUEST MODELS (what the API receives)
# ================================================================

class AnalyzeAndRecommendRequest(BaseModel):
    """
    For the main endpoint: POST /api/v1/stylist/analyze-and-recommend
    The user's photo is uploaded as a file (handled separately),
    but these are optional parameters sent as form data.
    """
    vision_prompt_version: str = Field(
        default="v2",
        description="Gemini Vision prompt version: v1, v2, or v3"
    )
    n_rag_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of fashion rules to retrieve from ChromaDB (1-10)"
    )


class ProfileOnlyRequest(BaseModel):
    """
    For the endpoint: POST /api/v1/stylist/recommend-from-profile
    Skip vision analysis — directly provide the user profile 
    (useful for testing or when analysis is already done).
    """
    face_shape: str = Field(
        ...,
        description="Face shape: Oval, Round, Square, Heart, Diamond, Rectangle, Triangle"
    )
    skin_tone: str = Field(
        ...,
        description="Skin tone: Fair, Light, Medium, Olive, Tan, Deep"
    )
    skin_undertone: str = Field(
        ...,
        description="Skin undertone: Warm, Cool, Neutral"
    )
    body_type: str = Field(
        ...,
        description="Body type: Hourglass, Pear, Apple, Rectangle, Inverted Triangle"
    )
    n_rag_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of fashion rules to retrieve"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "face_shape": "Oval",
                "skin_tone": "Medium",
                "skin_undertone": "Warm",
                "body_type": "Hourglass",
                "n_rag_results": 5
            }
        }


# ================================================================
# RESPONSE MODELS (what the API returns)
# ================================================================

class UserProfile(BaseModel):
    """Extracted user profile from Gemini Vision analysis."""
    face_shape: str
    face_shape_confidence: Optional[float] = None
    skin_tone: str
    skin_undertone: str
    skin_tone_confidence: Optional[float] = None
    body_type: str
    body_type_confidence: Optional[float] = None
    notes: Optional[str] = None
    prompt_version_used: Optional[str] = None


class ColorPalette(BaseModel):
    """Color recommendations."""
    best_colors: list[str]
    colors_to_avoid: list[str]
    color_explanation: str


class ClothingStyles(BaseModel):
    """Clothing style recommendations."""
    recommended: list[str]
    avoid: list[str]
    style_explanation: str


class Patterns(BaseModel):
    """Pattern recommendations."""
    recommended: list[str]
    avoid: list[str]
    pattern_explanation: str


class OutfitIdea(BaseModel):
    """A single outfit suggestion."""
    occasion: str
    outfit: str
    why_it_works: str


class Recommendation(BaseModel):
    """Complete fashion recommendation from Gemini LLM."""
    color_palette: ColorPalette
    clothing_styles: ClothingStyles
    patterns: Patterns
    fabrics: list[str]
    outfit_ideas: list[OutfitIdea]
    stylist_note: str


class RAGMetadata(BaseModel):
    """Metadata about the RAG retrieval step (for transparency/debugging)."""
    query_used: str
    n_rules_retrieved: int
    rule_ids_used: list[str]
    retrieval_distances: list[float]


class FullAnalysisResponse(BaseModel):
    """
    Complete response from the full pipeline endpoint.
    Includes profile, RAG metadata, and recommendation.
    """
    success: bool = True
    user_profile: dict                  # Raw dict (flexible for model changes)
    rag_metadata: RAGMetadata
    recommendation: dict                # Raw dict (nested structure)
    model_used: str


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""
    status: str
    version: str
    chroma_db_connected: bool
    chroma_document_count: int
    gemini_model: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
