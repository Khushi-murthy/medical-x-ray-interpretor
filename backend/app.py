import os
import uuid
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename


# ============================================================
# MEDVISION AI
# FLASK BACKEND
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

UPLOAD_FOLDER = os.path.join(PROJECT_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(PROJECT_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)


# ============================================================
# FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# IMPORT PREDICTOR
# ============================================================

print("\nLoading MedVision AI predictor...")

try:

    from predictor import (
        predict_image,
        DISEASES
    )

    print("✓ Predictor imported successfully")
    print("✓ Number of diseases:", len(DISEASES))

except Exception:

    print("\n❌ Predictor import failed")
    traceback.print_exc()
    raise


# ============================================================
# OPTIONAL GRAD-CAM
# ============================================================

GRADCAM_AVAILABLE = False
generate_gradcam = None

try:

    from gradcam import generate_gradcam

    GRADCAM_AVAILABLE = True

    print("✓ Grad-CAM module available")

except Exception as e:

    GRADCAM_AVAILABLE = False
    generate_gradcam = None

    print("⚠ Grad-CAM unavailable:")
    print(e)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "online",
        "service": "MedVision AI",
        "model": "CheXNet DenseNet121",
        "classes": len(DISEASES),
        "message": "MedVision AI backend is running."
    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "gradcam_available": GRADCAM_AVAILABLE
    })


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    print("\n")
    print("=" * 70)
    print("              /predict REQUEST")
    print("=" * 70)

    try:

        # ====================================================
        # CHECK IMAGE
        # ====================================================

        if "image" not in request.files:

            print("❌ No image field in request")

            return jsonify({
                "success": False,
                "error": "No image uploaded. Use field name 'image'."
            }), 400


        file = request.files["image"]


        if file.filename == "":

            print("❌ Empty filename")

            return jsonify({
                "success": False,
                "error": "No image selected."
            }), 400


        if not allowed_file(file.filename):

            print(
                "❌ Unsupported file:",
                file.filename
            )

            return jsonify({
                "success": False,
                "error": (
                    "Unsupported image format. "
                    "Use PNG, JPG or JPEG."
                )
            }), 400


        # ====================================================
        # SAVE IMAGE
        # ====================================================

        original_filename = secure_filename(
            file.filename
        )

        extension = os.path.splitext(
            original_filename
        )[1].lower()

        unique_filename = (
            uuid.uuid4().hex +
            extension
        )

        upload_path = os.path.join(
            UPLOAD_FOLDER,
            unique_filename
        )


        file.save(upload_path)


        print(
            "✓ Image received:",
            original_filename
        )

        print(
            "✓ Saved:",
            upload_path
        )


        if not os.path.exists(upload_path):

            raise RuntimeError(
                "Uploaded file could not be saved."
            )


        # ====================================================
        # RUN MODEL
        # ====================================================

        print("\nRunning MedVision AI...")

        result = predict_image(
            upload_path
        )

        print("✓ Prediction complete")


        # ====================================================
        # GET PREDICTIONS
        # ====================================================

        predictions = result.get(
            "predictions",
            {}
        )

        detected_findings = result.get(
            "detected_findings",
            []
        )

        calibrated_detection = result.get(
            "calibrated_detection",
            False
        )

        top_disease = result.get(
            "top_disease"
        )

        top_probability = float(
            result.get(
                "top_probability",
                0.0
            )
        )

        threshold = result.get(
            "threshold"
        )


        # ====================================================
        # SORT ALL PREDICTIONS
        # ====================================================

        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )


        # ====================================================
        # CREATE FRONTEND-FRIENDLY PREDICTION LIST
        # ====================================================

        prediction_list = []

        for disease, probability in sorted_predictions:

            probability = float(probability)

            prediction_list.append({

                "disease": disease,

                "name": disease,

                "probability": probability,

                "percentage": round(
                    probability * 100,
                    2
                ),

                "threshold": (
                    float(
                        result.get(
                            "thresholds",
                            {}
                        ).get(
                            disease,
                            0
                        )
                    )
                    if result.get("thresholds")
                    else None
                )

            })


        # ====================================================
        # PRIMARY FINDING
        # ====================================================

        if calibrated_detection and top_disease:

            finding_detected = True

            top_percentage = round(
                top_probability * 100,
                2
            )

            if top_probability >= 0.75:

                confidence_level = "High"

            elif top_probability >= 0.50:

                confidence_level = "Moderate"

            else:

                confidence_level = "Low"

        else:

            finding_detected = False

            top_percentage = round(
                top_probability * 100,
                2
            )

            confidence_level = "Low"


        # ====================================================
        # GRAD-CAM
        # ====================================================

        gradcam_url = None
        gradcam_error = None


        if (
            GRADCAM_AVAILABLE
            and finding_detected
            and top_disease
        ):

            try:

                print("\nGenerating Grad-CAM...")

                gradcam_filename = (
                    "gradcam_"
                    + uuid.uuid4().hex
                    + ".png"
                )

                gradcam_path = os.path.join(
                    OUTPUT_FOLDER,
                    gradcam_filename
                )


                from preprocess import preprocess_image

                image_tensor = preprocess_image(
                    upload_path
                )


                class_index = DISEASES.index(
                    top_disease
                )


                # Import the actual loaded CheXNet model
                try:

                    from chexnet_model import model as gradcam_model

                except Exception:

                    gradcam_model = None


                if gradcam_model is not None:

                    generate_gradcam(
                        model=gradcam_model,
                        image_tensor=image_tensor,
                        original_image_path=upload_path,
                        output_path=gradcam_path,
                        class_index=class_index
                    )


                    if os.path.exists(
                        gradcam_path
                    ):

                        gradcam_url = (
                            "/outputs/"
                            + gradcam_filename
                        )

                        print(
                            "✓ Grad-CAM generated:"
                        )

                        print(
                            gradcam_url
                        )

                else:

                    gradcam_error = (
                        "CheXNet model unavailable "
                        "for Grad-CAM."
                    )

            except Exception as e:

                gradcam_error = str(e)

                print(
                    "⚠ Grad-CAM failed:"
                )

                print(
                    gradcam_error
                )

                traceback.print_exc()


        # ====================================================
        # BACKEND RESPONSE
        # ====================================================

        response = {

            "success": True,

            "message":
                "X-ray analyzed successfully.",

            "filename":
                original_filename,

            "uploaded_file":
                "/uploads/" + unique_filename,


            # ------------------------------------------------
            # DIRECT PREDICTION DATA
            # ------------------------------------------------

            "predictions":
                predictions,

            "prediction_list":
                prediction_list,

            "sorted_predictions":
                sorted_predictions,


            # ------------------------------------------------
            # CALIBRATED RESULTS
            # ------------------------------------------------

            "detected_findings":
                detected_findings,

            "finding_detected":
                finding_detected,

            "calibrated_detection":
                calibrated_detection,

            "top_disease":
                top_disease,

            "top_percentage":
                top_percentage,

            "top_probability":
                top_probability,

            "threshold":
                threshold,

            "confidence_level":
                confidence_level,


            # ------------------------------------------------
            # ALSO KEEP ORIGINAL NESTED STRUCTURE
            # ------------------------------------------------

            "prediction": result,


            # ------------------------------------------------
            # GRAD-CAM
            # ------------------------------------------------

            "gradcam":
                gradcam_url,

            "gradcam_error":
                gradcam_error,


            # ------------------------------------------------
            # MEDICAL DISCLAIMER
            # ------------------------------------------------

            "medical_disclaimer": (
                "This AI output is for research and "
                "educational purposes only and is not "
                "a medical diagnosis. Clinical decisions "
                "must be made by a qualified healthcare "
                "professional."
            )

        }


        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print("\n")
        print("=" * 70)
        print("              FINAL API RESULT")
        print("=" * 70)

        if finding_detected:

            print(
                "Primary finding:",
                top_disease
            )

            print(
                "Model score:",
                f"{top_probability * 100:.2f}%"
            )

            if threshold is not None:

                print(
                    "Threshold:",
                    f"{float(threshold) * 100:.2f}%"
                )

        else:

            print(
                "No calibrated disease finding detected."
            )

            if sorted_predictions:

                print(
                    "Highest raw output:",
                    sorted_predictions[0][0]
                )

                print(
                    "Highest raw score:",
                    f"{sorted_predictions[0][1] * 100:.2f}%"
                )


        print(
            "Detected findings:"
        )

        if detected_findings:

            for finding in detected_findings:

                print(
                    f"  ✓ {finding['disease']}"
                    f" -> "
                    f"{finding['probability'] * 100:.2f}%"
                )

        else:

            print(
                "  None"
            )


        print("=" * 70)


        return jsonify(
            response
        ), 200


    # ========================================================
    # ERROR
    # ========================================================

    except Exception as e:

        print("\n")
        print("=" * 70)
        print("❌ PREDICTION ERROR")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)


        return jsonify({

            "success": False,

            "error": str(e),

            "message":
                "The X-ray could not be analyzed."

        }), 500


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route(
    "/uploads/<filename>",
    methods=["GET"]
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# SERVE GRAD-CAM
# ============================================================

@app.route(
    "/outputs/<filename>",
    methods=["GET"]
)
def output_file(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "success": False,
        "error": "Endpoint not found."
    }), 404


@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "Uploaded file is too large."
    }), 413


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("                 MEDVISION AI")
    print("                 FLASK SERVER")
    print("=" * 70)

    print(
        "\nProject:",
        PROJECT_DIR
    )

    print(
        "Uploads:",
        UPLOAD_FOLDER
    )

    print(
        "Outputs:",
        OUTPUT_FOLDER
    )

    print(
        "Grad-CAM:",
        GRADCAM_AVAILABLE
    )

    print(
        "\nServer:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("\n" + "=" * 70)


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )