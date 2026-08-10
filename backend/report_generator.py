from datetime import datetime


def generate_report(predictions):
    """
    Generate a professional medical report from predictions.
    """

    # Sort diseases by confidence
    sorted_predictions = sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    primary_disease, primary_score = sorted_predictions[0]

    # Other findings (>10%)
    secondary = [
        disease
        for disease, score in sorted_predictions[1:]
        if score >= 0.10
    ]

    if primary_score >= 0.80:
        severity = "High"
    elif primary_score >= 0.50:
        severity = "Moderate"
    else:
        severity = "Low"

    report = {
        "date": datetime.now().strftime("%d-%m-%Y"),
        "time": datetime.now().strftime("%I:%M %p"),

        "primary_finding": primary_disease,
        "confidence": round(primary_score * 100, 2),

        "secondary_findings": secondary,

        "severity": severity,

        "interpretation":
        f"The AI model identified {primary_disease} as the most probable finding with a confidence of {round(primary_score*100,2)}%. "
        "These predictions are generated using a DenseNet121 deep learning model trained on chest X-ray images.",

        "recommendations": [

            "Consult a certified radiologist.",

            "Correlate findings with clinical symptoms.",

            "Perform additional investigations if required.",

            "Do not rely solely on AI predictions."

        ],
        "disease_probabilities": [
    (disease, round(score * 100, 2))
    for disease, score in sorted_predictions
],

        "disclaimer":
        "This report is generated automatically by an AI system and is intended only for educational and decision-support purposes. "
        "It should not be considered a final medical diagnosis."
    }

    return report