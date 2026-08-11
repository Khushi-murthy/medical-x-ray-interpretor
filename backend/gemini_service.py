import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if API_KEY and genai is not None:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print("Warning: could not initialize Gemini client:", e)
        client = None

if client is None:
    print(
        "Warning: Gemini AI is unavailable. "
        "X-ray explanation requests will use a fallback summary."
    )


def generate_xray_explanation(predictions, language="en"):
    """Generate a natural-language Gemini explanation for chest X-ray predictions."""

    if not isinstance(predictions, dict) or not predictions:
        return (
            "No prediction data is available to generate an explanation. "
            "Please analyze an X-ray image first."
        )

    prediction_text = ""
    for disease, score in sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        prediction_text += f"{disease}: {score*100:.2f}%\n"

    language_name = "English" if language.lower().startswith("en") else language

    prompt = f"""
You are an expert radiologist.

A DenseNet121 AI model analyzed a chest X-ray and produced the following disease probabilities:

{prediction_text}

Generate a concise explanation in {language_name} that includes:
- the most likely disease,
- any notable secondary probabilities,
- the recommended next step,
- a clear statement that this is not a final diagnosis,
- a reminder that a qualified radiologist must review the case.
"""

    if client is None:
        return _generate_fallback_explanation(predictions, language=language)

    try:
        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=prompt
        )
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        print("Gemini explanation failed:", e)
        return _generate_fallback_explanation(predictions, language=language)


def _generate_fallback_explanation(predictions, language="en"):
    sorted_preds = sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if not sorted_preds:
        return (
            "The AI model produced no predictions. "
            "No explanation can be generated at this time."
        )

    primary, primary_score = sorted_preds[0]
    secondary = [
        f"{d}: {s*100:.2f}%"
        for d, s in sorted_preds[1:4]
        if s >= 0.05
    ]

    explanation = [
        f"Primary finding: {primary} ({primary_score*100:.2f}%).",
    ]

    if secondary:
        explanation.append("Secondary findings include: " + ", ".join(secondary) + ".")
    else:
        explanation.append("No strong secondary findings were identified.")

    explanation.extend([
        "This output is generated from AI probabilities and is not a clinical diagnosis.",
        "A qualified radiologist should review the X-ray and the patient history before any decisions are made."
    ])

    return " ".join(explanation)
