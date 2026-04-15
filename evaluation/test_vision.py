"""
test_vision.py — Evaluate Gemini Vision API accuracy.

WHAT THIS TESTS:
  Sends multiple test images to Gemini Vision analysis
  and scores how accurately it detects face shape, 
  skin tone, and body type.

USE FOR:
  - Your project report's evaluation section
  - Showing accuracy metrics in viva
  - Comparing prompt versions (v1 vs v2 vs v3)

HOW TO RUN:
  1. Create a folder: evaluation/test_images/
  2. Add photos with known labels (see TEST_CASES below)
  3. Run: python evaluation/test_vision.py

OUTPUT:
  - Per-category accuracy scores
  - JSON results saved to evaluation/results/
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.vision_analyzer import analyze_photo

logging.basicConfig(level=logging.WARNING)  # Suppress info noise during eval
logger = logging.getLogger(__name__)


# ================================================================
# TEST CASES
# Define your test images and their ground-truth labels here.
# Add real photos to evaluation/test_images/ folder.
# ================================================================
TEST_CASES = [
    # Format: { "image": "filename.jpg", "expected": { field: value } }
    # Add real test cases here with photos you know the answers to.
    # Example (commented out — add your real test photos):
    #
    # {
    #     "image": "oval_medium_hourglass.jpg",
    #     "expected": {
    #         "face_shape": "Oval",
    #         "skin_tone": "Medium",
    #         "body_type": "Hourglass"
    #     }
    # },
]

# Path to test images folder
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"
RESULTS_DIR = Path(__file__).parent / "results"


def evaluate_vision(
    prompt_version: str = "v2",
    save_results: bool = True
) -> dict:
    """
    Run evaluation across all test cases and compute accuracy.

    Args:
        prompt_version: Which prompt to test ("v1", "v2", or "v3")
        save_results: Whether to save JSON results to disk

    Returns:
        dict with accuracy scores and per-sample results
    """
    if not TEST_CASES:
        print("⚠️  No test cases defined in TEST_CASES list.")
        print("   Add test images to evaluation/test_images/ and define their labels.")
        return {}

    results = []
    correct = {"face_shape": 0, "skin_tone": 0, "body_type": 0}
    total = len(TEST_CASES)

    print(f"\n🧪 Vision Evaluation | Prompt Version: {prompt_version}")
    print(f"   Test cases: {total}")
    print("=" * 60)

    for i, test_case in enumerate(TEST_CASES):
        image_path = TEST_IMAGES_DIR / test_case["image"]

        if not image_path.exists():
            print(f"  [{i+1}/{total}] SKIP (file not found): {test_case['image']}")
            total -= 1
            continue

        print(f"  [{i+1}/{total}] Testing: {test_case['image']}")

        try:
            # Run vision analysis
            prediction = analyze_photo(str(image_path), prompt_version=prompt_version)

            # Compare each field (case-insensitive)
            sample_result = {
                "image": test_case["image"],
                "expected": test_case["expected"],
                "predicted": {
                    "face_shape": prediction.get("face_shape", "unknown"),
                    "skin_tone": prediction.get("skin_tone", "unknown"),
                    "body_type": prediction.get("body_type", "unknown"),
                },
                "correct": {},
                "confidences": {
                    "face_shape": prediction.get("face_shape_confidence"),
                    "skin_tone": prediction.get("skin_tone_confidence"),
                    "body_type": prediction.get("body_type_confidence"),
                }
            }

            # Score each field
            for field in ["face_shape", "skin_tone", "body_type"]:
                expected_val = test_case["expected"].get(field, "").lower()
                predicted_val = prediction.get(field, "").lower()
                is_correct = expected_val == predicted_val
                sample_result["correct"][field] = is_correct
                if is_correct:
                    correct[field] += 1

            # Print per-sample summary
            face_ok = "✅" if sample_result["correct"]["face_shape"] else "❌"
            skin_ok = "✅" if sample_result["correct"]["skin_tone"] else "❌"
            body_ok = "✅" if sample_result["correct"]["body_type"] else "❌"
            print(
                f"       Face: {face_ok} ({prediction.get('face_shape')}) | "
                f"Skin: {skin_ok} ({prediction.get('skin_tone')}) | "
                f"Body: {body_ok} ({prediction.get('body_type')})"
            )

            results.append(sample_result)

        except Exception as e:
            print(f"       ERROR: {e}")
            results.append({"image": test_case["image"], "error": str(e)})

    # Compute accuracy percentages
    if total > 0:
        accuracy = {
            "face_shape_accuracy": round(correct["face_shape"] / total * 100, 1),
            "skin_tone_accuracy": round(correct["skin_tone"] / total * 100, 1),
            "body_type_accuracy": round(correct["body_type"] / total * 100, 1),
            "overall_accuracy": round(
                (correct["face_shape"] + correct["skin_tone"] + correct["body_type"])
                / (total * 3) * 100, 1
            ),
        }
    else:
        accuracy = {}

    print("\n" + "=" * 60)
    print("📊 ACCURACY RESULTS:")
    print(f"   Face Shape : {accuracy.get('face_shape_accuracy', 'N/A')}%")
    print(f"   Skin Tone  : {accuracy.get('skin_tone_accuracy', 'N/A')}%")
    print(f"   Body Type  : {accuracy.get('body_type_accuracy', 'N/A')}%")
    print(f"   Overall    : {accuracy.get('overall_accuracy', 'N/A')}%")
    print(f"   (n={total} images, prompt={prompt_version})")
    print("=" * 60)

    # Build output report
    report = {
        "evaluation_type": "vision_analysis",
        "prompt_version": prompt_version,
        "timestamp": datetime.now().isoformat(),
        "total_samples": total,
        "accuracy": accuracy,
        "per_sample_results": results,
    }

    # Save to disk
    if save_results and total > 0:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = RESULTS_DIR / f"vision_eval_{prompt_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Results saved: {filename}")

    return report


def compare_prompt_versions():
    """
    Compare accuracy across all three prompt versions.
    Great for your project report — shows prompt engineering iteration.
    """
    print("\n🔬 PROMPT VERSION COMPARISON")
    print("   Testing v1, v2, and v3 on the same images...")
    print("=" * 60)

    comparison = {}
    for version in ["v1", "v2", "v3"]:
        result = evaluate_vision(prompt_version=version, save_results=True)
        if result and "accuracy" in result:
            comparison[version] = result["accuracy"]

    print("\n📊 COMPARISON TABLE:")
    print(f"{'Metric':<25} {'v1':>8} {'v2':>8} {'v3':>8}")
    print("-" * 50)
    for metric in ["face_shape_accuracy", "skin_tone_accuracy", "body_type_accuracy", "overall_accuracy"]:
        label = metric.replace("_accuracy", "").replace("_", " ").title()
        v1 = comparison.get("v1", {}).get(metric, "N/A")
        v2 = comparison.get("v2", {}).get(metric, "N/A")
        v3 = comparison.get("v3", {}).get(metric, "N/A")
        print(f"{label:<25} {str(v1)+' %':>8} {str(v2)+' %':>8} {str(v3)+' %':>8}")

    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Gemini Vision accuracy")
    parser.add_argument("--prompt", default="v2", choices=["v1", "v2", "v3"],
                        help="Prompt version to evaluate")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all prompt versions")
    args = parser.parse_args()

    if args.compare:
        compare_prompt_versions()
    else:
        evaluate_vision(prompt_version=args.prompt)
