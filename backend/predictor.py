import os
import numpy as np
import tensorflow as tf

try:
    from preprocess import preprocess_image
except ModuleNotFoundError:
    from .preprocess import preprocess_image

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

THRESHOLDS = {
    "Atelectasis": 0.17988281,
    "Cardiomegaly": 0.092365935,
    "Consolidation": 0.094164275,
    "Edema": 0.111305386,
    "Effusion": 0.3161116,
    "Emphysema": 0.102542125,
    "Fibrosis": 0.056806,
    "Hernia": 0.035011284,
    "Infiltration": 0.2352473,
    "Mass": 0.105771594,
    "Nodule": 0.14657141,
    "Pleural_Thickening": 0.101057604,
    "Pneumonia": 0.036833875,
    "Pneumothorax": 0.18982303,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "final_model.keras"))

print("\n" + "=" * 70)
print("                 MEDVISION AI")
print("          CALIBRATED MODEL LOADING")
print("=" * 70)
print("\nModel path:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✓ Model loaded successfully")

print("\n========== MODEL INFORMATION ==========")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Number of output classes:", model.output_shape[-1])

if model.output_shape[-1] != len(DISEASES):
    raise RuntimeError(
        f"Model has {model.output_shape[-1]} outputs but {len(DISEASES)} diseases are configured."
    )

print("\n========== DISEASE THRESHOLDS ==========")
for disease in DISEASES:
    print(f"{disease:<25} {THRESHOLDS[disease]:.8f}")
print("========================================")


def _to_probabilities(raw_output):
    values = np.asarray(raw_output, dtype=np.float32)
    if values.ndim == 2:
        values = values[0]

    if values.ndim != 1 or len(values) != len(DISEASES):
        raise RuntimeError(f"Unexpected model output shape: {values.shape}")

    # Your tested model returns logits such as -2.36 and +1.47.
    # Calibrated thresholds are probabilities, so sigmoid is required.
    if np.any(values < 0.0) or np.any(values > 1.0):
        print("⚠ Model output detected as LOGITS.")
        print("✓ Applying sigmoid conversion.")
        probabilities = tf.math.sigmoid(values).numpy()
    else:
        print("✓ Model output already appears to be probabilities.")
        probabilities = values

    probabilities = np.nan_to_num(
        probabilities, nan=0.0, posinf=1.0, neginf=0.0
    )
    return np.clip(probabilities, 0.0, 1.0).astype(np.float32)


def predict_image(image_path):
    print("\n" + "=" * 70)
    print("                  NEW X-RAY ANALYSIS")
    print("=" * 70)
    print("Image:", image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"X-ray image not found:\n{image_path}")

    print("\n[1] Preprocessing image...")
    image_tensor = preprocess_image(image_path)
    print("✓ Tensor shape:", image_tensor.shape)

    print("\n[2] Running model...")
    raw_output = model.predict(image_tensor, verbose=0)
    raw_output = np.asarray(raw_output, dtype=np.float32)

    raw_logits = raw_output[0] if raw_output.ndim == 2 else raw_output
    probabilities = _to_probabilities(raw_output)

    predictions = {
        disease: float(probabilities[i])
        for i, disease in enumerate(DISEASES)
    }

    sorted_predictions = sorted(
        predictions.items(), key=lambda item: item[1], reverse=True
    )

    raw_top_disease, raw_top_probability = sorted_predictions[0]

    detected_findings = []
    for disease in DISEASES:
        probability = predictions[disease]
        threshold = THRESHOLDS[disease]
        if probability >= threshold:
            detected_findings.append({
                "disease": disease,
                "probability": probability,
                "threshold": threshold
            })

    detected_findings.sort(
        key=lambda item: item["probability"], reverse=True
    )

    if detected_findings:
        primary = detected_findings[0]
        top_disease = primary["disease"]
        top_probability = primary["probability"]
        primary_threshold = primary["threshold"]
        calibrated_detection = True
    else:
        top_disease = None
        top_probability = raw_top_probability
        primary_threshold = THRESHOLDS[raw_top_disease]
        calibrated_detection = False

    if calibrated_detection:
        if top_probability >= 0.75:
            confidence_level = "HIGH"
        elif top_probability >= 0.50:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"
    else:
        confidence_level = "VERY_LOW"

    print("\n" + "=" * 70)
    print("                 MODEL PROBABILITIES")
    print("=" * 70)

    for disease, probability in sorted_predictions:
        threshold = THRESHOLDS[disease]
        status = "DETECTED" if probability >= threshold else "below threshold"
        print(
            f"{disease:<25}{probability * 100:8.4f}%"
            f" | threshold {threshold * 100:8.4f}% | {status}"
        )

    print("\n" + "=" * 70)
    print("                 CALIBRATED FINDINGS")
    print("=" * 70)

    if detected_findings:
        for finding in detected_findings:
            print(
                f"✓ {finding['disease']} | "
                f"score={finding['probability'] * 100:.4f}% | "
                f"threshold={finding['threshold'] * 100:.4f}%"
            )
    else:
        print("✓ No disease exceeded its calibrated threshold.")
        print(
            f"Highest raw output: {raw_top_disease} "
            f"({raw_top_probability * 100:.4f}%)"
        )

    print("=" * 70)

    return {
        "predictions": predictions,
        "detected_findings": detected_findings,
        "calibrated_detection": calibrated_detection,
        "top_disease": top_disease,
        "top_probability": float(top_probability),
        "raw_top_disease": raw_top_disease,
        "raw_top_probability": float(raw_top_probability),
        "threshold": float(primary_threshold),
        "top_percentage": float(top_probability * 100.0),
        "confidence_level": confidence_level,
        "raw_logits": {
            disease: float(raw_logits[i])
            for i, disease in enumerate(DISEASES)
        },
        "model_output_type": (
            "logits_converted_with_sigmoid"
            if np.any(raw_logits < 0.0) or np.any(raw_logits > 1.0)
            else "probabilities"
        )
    }


if __name__ == "__main__":
    print("\nMEDVISION AI predictor loaded.")
    print("\nDiseases:")
    for disease in DISEASES:
        print(f"- {disease}")
    print("\nCalibrated thresholds loaded successfully.")
    print("\nRun predict_image(image_path) to analyze an X-ray.")