from disease_knowledge import DISEASE_KNOWLEDGE


# ============================================================
# MEDVISION AI
# PATIENT-FRIENDLY SUMMARY
# ============================================================


def generate_patient_summary(predictions):

    if not predictions:

        return {
            "primary_finding": "No prediction available",
            "score": 0,
            "explanation":
                "The system could not generate a prediction.",
            "recommendation":
                "Please have the X-ray reviewed by a qualified "
                "healthcare professional."
        }


    # --------------------------------------------------------
    # Sort predictions
    # --------------------------------------------------------

    sorted_predictions = sorted(

        predictions.items(),

        key=lambda x: x[1],

        reverse=True

    )


    primary_finding = (
        sorted_predictions[0][0]
    )


    primary_score = float(
        sorted_predictions[0][1]
    )


    # --------------------------------------------------------
    # Knowledge
    # --------------------------------------------------------

    knowledge = DISEASE_KNOWLEDGE.get(

        primary_finding,

        {

            "meaning":
                "The model identified an imaging pattern "
                "associated with this finding.",

            "why_it_matters":
                "The significance cannot be determined from "
                "the AI score alone.",

            "action":
                "Discuss the result with a qualified "
                "healthcare professional."

        }

    )


    # --------------------------------------------------------
    # Model score interpretation
    # --------------------------------------------------------

    percentage = (
        primary_score * 100
    )


    if percentage >= 50:

        score_label = (
            "Higher model score"
        )

    elif percentage >= 20:

        score_label = (
            "Intermediate model score"
        )

    else:

        score_label = (
            "Lower model score"
        )


    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation = knowledge[
        "action"
    ]


    # --------------------------------------------------------
    # Emergency warning
    # --------------------------------------------------------

    emergency_warning = (

        "Seek urgent medical attention if you experience "
        "severe difficulty breathing, severe or worsening "
        "chest pain, blue or grey lips or face, fainting, "
        "confusion, or rapidly worsening symptoms."

    )


    # --------------------------------------------------------
    # Return structured information
    # --------------------------------------------------------

    return {

        "primary_finding":
            primary_finding,

        "score":
            percentage,

        "score_label":
            score_label,

        "meaning":
            knowledge["meaning"],

        "why_it_matters":
            knowledge["why_it_matters"],

        "recommendation":
            recommendation,

        "emergency_warning":
            emergency_warning,

        "disclaimer":
            (
                "This is an AI-generated decision-support "
                "summary and not a medical diagnosis. "
                "Model scores are not equivalent to clinical "
                "probabilities or disease severity. "
                "A qualified healthcare professional should "
                "interpret the X-ray."
            )

    }