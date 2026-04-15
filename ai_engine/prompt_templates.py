"""
prompt_templates.py — All Gemini prompts, versioned and documented.

WHY this file exists:
  Prompt engineering is a KEY contribution in your project report.
  Keeping all prompts here (with version history) lets you:
    1. Show prompt iteration as a skill in viva
    2. Easily A/B test different prompt versions
    3. Document WHY each design decision was made

PROMPT DESIGN PRINCIPLES USED:
  - Role assignment ("You are an expert...")
  - Structured output (JSON schema in prompt)
  - Chain-of-thought ("First analyze X, then...")
  - Constraint setting ("Respond ONLY with valid JSON")
"""


# ================================================================
# MODULE 1: VISION ANALYSIS PROMPTS
# (Sent to Gemini with the user's photo)
# ================================================================

class VisionPrompts:
    """
    Prompts for Gemini Vision API to extract physical attributes
    from the user's uploaded photo.
    """

    # ---- Version 1 (baseline) ----------------------------------
    # Simple direct question. Used as baseline for comparison.
    ANALYZE_V1 = """
    Look at this photo and tell me:
    1. Face shape
    2. Skin tone
    3. Body type
    """

    # ---- Version 2 (structured JSON output) --------------------
    # Added JSON schema so output is machine-readable.
    # This is the version we use in production.
    ANALYZE_V2 = """
    You are an expert fashion analyst and personal stylist with 
    10 years of experience. Analyze the person in this photo carefully.

    Extract the following attributes:

    1. FACE SHAPE: Identify from these categories:
       - Oval, Round, Square, Heart, Diamond, Rectangle, Triangle

    2. SKIN TONE: Identify from these categories:
       - Fair, Light, Medium, Olive, Tan, Deep
       Also identify the undertone: Warm, Cool, or Neutral

    3. BODY TYPE: Identify from these categories:
       - Hourglass, Pear, Apple, Rectangle, Inverted Triangle

    IMPORTANT RULES:
    - If the photo is unclear or a body part is not visible, 
      set that field to "unclear" and explain in the notes field.
    - Respond ONLY with valid JSON. No extra text before or after.
    - Use exactly the field names shown below.

    Respond in this exact JSON format:
    {
        "face_shape": "<shape>",
        "face_shape_confidence": <0.0 to 1.0>,
        "skin_tone": "<tone>",
        "skin_undertone": "<undertone>",
        "skin_tone_confidence": <0.0 to 1.0>,
        "body_type": "<type>",
        "body_type_confidence": <0.0 to 1.0>,
        "notes": "<any issues or extra observations>"
    }
    """

    # ---- Version 3 (chain-of-thought reasoning) ----------------
    # Added step-by-step reasoning before JSON output.
    # Better accuracy but slower — use for evaluation tests.
    ANALYZE_V3 = """
    You are an expert fashion analyst with deep knowledge of 
    body proportions, color science, and personal styling.

    Study the photo carefully and reason step by step:

    STEP 1 — FACE SHAPE ANALYSIS:
    Examine the jawline width vs forehead width vs face length.
    Consider: Is the face longer or wider? Is the jaw angular or rounded?
    Then classify as: Oval, Round, Square, Heart, Diamond, 
                      Rectangle, or Triangle.

    STEP 2 — SKIN TONE ANALYSIS:
    Look at the overall skin color under the photo's lighting.
    Classify tone as: Fair, Light, Medium, Olive, Tan, or Deep.
    Classify undertone as: Warm (yellow/golden), Cool (pink/blue), 
                           or Neutral (mix of both).

    STEP 3 — BODY TYPE ANALYSIS:
    Compare shoulder width, waist definition, and hip width.
    Classify as: Hourglass, Pear, Apple, Rectangle, 
                 or Inverted Triangle.

    After your analysis, output ONLY this JSON (no other text):
    {
        "face_shape": "<shape>",
        "face_shape_confidence": <0.0 to 1.0>,
        "skin_tone": "<tone>",
        "skin_undertone": "<undertone>",
        "skin_tone_confidence": <0.0 to 1.0>,
        "body_type": "<type>",
        "body_type_confidence": <0.0 to 1.0>,
        "notes": "<observations or limitations>"
    }
    """

    # Active version used in production
    ACTIVE = ANALYZE_V2


# ================================================================
# MODULE 2: RECOMMENDATION PROMPTS
# (Sent to Gemini LLM with analysis + retrieved fashion rules)
# ================================================================

class RecommendationPrompts:
    """
    Prompts for the final recommendation generation step.
    These take the vision analysis + RAG-retrieved rules as input.
    """

    # ---- Version 1 (baseline) ----------------------------------
    RECOMMEND_V1 = """
    Based on the user's profile and fashion rules below, 
    give clothing recommendations.

    User Profile: {user_profile}
    Fashion Rules: {fashion_rules}
    """

    # ---- Version 2 (structured, professional) ------------------
    # Used in production. Produces consistent, structured output.
    RECOMMEND_V2 = """
    You are a professional personal stylist creating a customized 
    fashion recommendation report for a client.

    CLIENT PROFILE (extracted from their photo):
    {user_profile}

    RELEVANT FASHION GUIDELINES (retrieved from knowledge base):
    {fashion_rules}

    Based on the above, create a detailed, personalized recommendation.

    Your recommendation MUST cover:
    1. COLORS: Best colors for their skin tone + undertone.
       Explain WHY each color works (e.g., "warm amber enhances 
       your warm undertone by creating harmony").
    2. CLOTHING STYLES: Best cuts and silhouettes for their body type.
       Be specific (e.g., "A-line skirts", "V-neck tops").
    3. PATTERNS: Which patterns work (and which to avoid).
    4. FABRICS: Any fabric recommendations.
    5. WHAT TO AVOID: Gently mention styles that may not flatter.

    TONE: Warm, encouraging, and professional. 
    Never use negative language about the client's body.

    Respond ONLY with valid JSON in this exact format:
    {{
        "color_palette": {{
            "best_colors": ["<color1>", "<color2>", "<color3>", ...],
            "colors_to_avoid": ["<color1>", "<color2>", ...],
            "color_explanation": "<why these colors work>"
        }},
        "clothing_styles": {{
            "recommended": ["<style1>", "<style2>", ...],
            "avoid": ["<style1>", "<style2>", ...],
            "style_explanation": "<why these styles work>"
        }},
        "patterns": {{
            "recommended": ["<pattern1>", "<pattern2>", ...],
            "avoid": ["<pattern1>", ...],
            "pattern_explanation": "<brief explanation>"
        }},
        "fabrics": ["<fabric1>", "<fabric2>", ...],
        "outfit_ideas": [
            {{
                "occasion": "<casual/formal/party>",
                "outfit": "<specific outfit description>",
                "why_it_works": "<explanation>"
            }}
        ],
        "stylist_note": "<one encouraging personalized message>"
    }}
    """

    # Active version used in production
    ACTIVE = RECOMMEND_V2


# ================================================================
# MODULE 3: RAG QUERY PROMPTS
# (Used to build the search query for ChromaDB retrieval)
# ================================================================

class RAGQueryPrompts:
    """
    Prompts to convert user profile into semantic search queries
    for ChromaDB retrieval.
    """

    # Build a natural language query from structured user profile
    BUILD_QUERY_V1 = """
    Convert this user profile into a search query for fashion rules:
    
    Face Shape: {face_shape}
    Skin Tone: {skin_tone} with {skin_undertone} undertone
    Body Type: {body_type}
    
    Write 2-3 sentences describing what fashion rules would help 
    this person. Focus on colors, styles, and silhouettes.
    Be specific and use fashion terminology.
    """

    ACTIVE = BUILD_QUERY_V1


# ================================================================
# PROMPT VERSION REGISTRY
# (For your project report — shows your iteration process)
# ================================================================

PROMPT_CHANGELOG = {
    "vision_analysis": {
        "v1": "Simple question — baseline. Low structured output.",
        "v2": "Added JSON schema + role assignment. Production version.",
        "v3": "Added chain-of-thought reasoning. Higher accuracy, slower.",
    },
    "recommendation": {
        "v1": "Simple template. Unstructured output.",
        "v2": "Structured JSON output + specific guidance per category.",
    },
    "rag_query": {
        "v1": "Converts profile to natural language query for semantic search.",
    }
}
