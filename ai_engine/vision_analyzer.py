"""
vision_analyzer.py — Gemini Vision API integration (new google-genai SDK).
"""

import re
import json
import logging
import io
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from config import settings
from ai_engine.prompt_templates import VisionPrompts

logger = logging.getLogger(__name__)

# ── Single shared client (new SDK pattern) ─────────────────────────
_client = genai.Client(api_key=settings.google_api_key)


def _load_image(image_source) -> Image.Image:
    """Load an image from a file path (str/Path) or raw bytes."""
    if isinstance(image_source, (str, Path)):
        return Image.open(image_source)
    elif isinstance(image_source, bytes):
        return Image.open(io.BytesIO(image_source))
    else:
        raise ValueError("image_source must be a file path (str) or bytes")


def _parse_gemini_json_response(raw_text: str) -> dict:
    """
    Safely extract JSON from Gemini's response.
    Strips markdown code fences (```json ... ```) if present.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    cleaned = cleaned.replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        logger.error(f"Raw response: {raw_text}")
        raise ValueError(
            f"Gemini returned non-JSON response. Raw text: {raw_text[:300]}"
        )


def analyze_photo(image_source, prompt_version: str = "v2") -> dict:
    """
    Send a photo to Gemini Vision and get a structured user profile.

    Args:
        image_source : File path (str/Path) or raw image bytes.
        prompt_version: "v1", "v2" (default), or "v3"

    Returns:
        dict with face_shape, skin_tone, skin_undertone, body_type,
        confidence scores, notes, and prompt_version_used.

    Example:
        >>> result = analyze_photo("photo.jpg")
        >>> result["face_shape"]   # "Oval"
        >>> result["skin_tone"]    # "Medium"
        >>> result["body_type"]    # "Hourglass"
    """
    logger.info(f"Starting photo analysis — prompt version: {prompt_version}")

    # Pick the prompt
    prompt_map = {
        "v1": VisionPrompts.ANALYZE_V1,
        "v2": VisionPrompts.ANALYZE_V2,
        "v3": VisionPrompts.ANALYZE_V3,
    }
    if prompt_version not in prompt_map:
        raise ValueError(f"Invalid prompt_version '{prompt_version}'. Choose v1/v2/v3.")
    prompt = prompt_map[prompt_version]

    # Load image as PIL
    try:
        image = _load_image(image_source)
        logger.info(f"Image loaded: size={image.size}, mode={image.mode}")
    except Exception as e:
        raise ValueError(f"Could not load image: {e}")

    # Send to Gemini Vision (new SDK: pass PIL image + text prompt as list)
    logger.info(f"Sending to Gemini Vision ({settings.gemini_vision_model})...")
    response = _client.models.generate_content(
        model=settings.gemini_vision_model,
        contents=[image, prompt],
    )

    raw_text = response.text
    logger.info("Received response from Gemini")
    logger.debug(f"Raw: {raw_text}")

    result = _parse_gemini_json_response(raw_text)
    result["prompt_version_used"] = prompt_version

    logger.info(
        f"Analysis complete — face: {result.get('face_shape')}, "
        f"skin: {result.get('skin_tone')}, body: {result.get('body_type')}"
    )
    return result


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    if not test_image:
        print("Usage: python ai_engine/vision_analyzer.py <path_to_image>")
        sys.exit(1)
    print(f"\n🔍 Analyzing: {test_image}")
    result = analyze_photo(test_image, prompt_version="v2")
    print("\n✅ RESULT:")
    print(json.dumps(result, indent=2))
