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

print("ENV PATH:", env_path)
print("GEMINI_API_KEY configured:", bool(API_KEY))

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
        "Reports will use a fallback AI summary instead."
    )


# ===========================================
# Generate AI Medical Report
# ===========================================

def generate_ai_report(predictions):

    prediction_text = ""

    for disease, score in sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        prediction_text += f"{disease}: {score*100:.2f}%\n"


    prompt = f"""

You are an experienced radiologist.

A DenseNet121 AI model analyzed a chest X-ray.

Disease probabilities:

{prediction_text}

Generate a professional radiology report using the following format.

------------------------

AI RADIOLOGY REPORT

Primary Finding

Secondary Findings

Clinical Interpretation

Recommendations

Important Disclaimer

------------------------

Rules:

- Mention only diseases having meaningful probability.
- Explain in simple medical language.
- Mention that this is NOT a final diagnosis.
- Mention that a certified radiologist should review the X-ray.
- Keep the report around 250-350 words.
- Make it sound like a real hospital report.

"""


    if client is None:
        return generate_fallback_report(predictions)

    try:
        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=prompt
        )
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        print("Gemini generation failed:", e)
        return generate_fallback_report(predictions)


def generate_fallback_report(predictions):
    sorted_preds = sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if not sorted_preds:
        return (
            "The AI model produced no predictions. "
            "No radiology report can be generated at this time."
        )

    primary, primary_score = sorted_preds[0]
    secondary = [
        f"{d}: {s*100:.2f}%"
        for d, s in sorted_preds[1:3]
        if s >= 0.01
    ]

    lines = [
        "AI RADIOLOGY REPORT",
        "",
        f"Primary finding: {primary} ({primary_score*100:.2f}%)",
    ]

    if secondary:
        lines.append("Secondary findings:")
        lines.extend(secondary)
    else:
        lines.append("Secondary findings: None detected above a low threshold.")

    lines.extend([
        "",
        "Clinical interpretation:",
        (
            f"The AI model analysis indicates that the most likely finding is {primary} "
            f"with a confidence of {primary_score*100:.2f}%. "
            "Predictions should be interpreted in conjunction with clinical context "
            "and reviewed by a qualified radiologist."
        ),
        "",
        "Recommendations:",
        "• Consult a certified radiologist for definitive diagnosis.",
        "• Correlate with patient symptoms and clinical history.",
        "• Consider follow-up imaging or testing when indicated.",
        "",
        "Important Disclaimer:",
        (
            "This report is for educational and decision-support purposes only. "
            "It is not a final medical diagnosis."
        )
    ])

    return "\n".join(lines)
