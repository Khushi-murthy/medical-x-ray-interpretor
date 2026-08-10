import os
import sys
import glob
import numpy as np

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BACKEND_DIR
)

sys.path.insert(0, BACKEND_DIR)


# ------------------------------------------------------------
# IMPORT MODEL
# ------------------------------------------------------------

from chexnet_model import (
    model,
    DISEASES
)

from preprocess import (
    preprocess_image
)


# ------------------------------------------------------------
# TEST IMAGE DIRECTORY
# ------------------------------------------------------------

IMAGE_DIR = os.path.join(
    PROJECT_DIR,
    "test_images"
)


# ------------------------------------------------------------
# FIND ALL X-RAYS
# ------------------------------------------------------------

extensions = [
    "*.png",
    "*.jpg",
    "*.jpeg"
]

image_paths = []

for extension in extensions:

    image_paths.extend(
        glob.glob(
            os.path.join(
                IMAGE_DIR,
                extension
            )
        )
    )


image_paths = sorted(
    image_paths
)


if not image_paths:

    raise FileNotFoundError(
        f"No images found in:\n{IMAGE_DIR}"
    )


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("                 MEDVISION AI")
print("              CHEXNET BATCH TEST")
print("=" * 80)

print(
    "\nImages found:",
    len(image_paths)
)

for path in image_paths:

    print(
        " -",
        os.path.basename(path)
    )


# ------------------------------------------------------------
# RESULTS STORAGE
# ------------------------------------------------------------

all_results = []


# ------------------------------------------------------------
# PROCESS EVERY IMAGE
# ------------------------------------------------------------

for image_number, image_path in enumerate(
    image_paths,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    print("\n")
    print("=" * 80)
    print(
        f"IMAGE {image_number}/{len(image_paths)}: {filename}"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    try:

        image_tensor = preprocess_image(
            image_path
        )

    except Exception as e:

        print(
            "\n❌ PREPROCESSING FAILED:"
        )

        print(e)

        continue


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            image_tensor,
            verbose=0
        )[0]

    except Exception as e:

        print(
            "\n❌ PREDICTION FAILED:"
        )

        print(e)

        continue


    # --------------------------------------------------------
    # SORT PREDICTIONS
    # --------------------------------------------------------

    results = sorted(
        zip(
            DISEASES,
            prediction
        ),
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # DISPLAY TOP 5
    # --------------------------------------------------------

    print("\nTOP 5 FINDINGS")

    print(
        "-" * 60
    )

    for rank, (
        disease,
        probability
    ) in enumerate(
        results[:5],
        start=1
    ):

        print(
            f"{rank}. "
            f"{disease:<25}"
            f"{float(probability) * 100:>10.4f}%"
        )


    # --------------------------------------------------------
    # RAW VALUES
    # --------------------------------------------------------

    print(
        "\nRAW OUTPUT RANGE:"
    )

    print(
        "Minimum:",
        f"{float(np.min(prediction)):.10f}"
    )

    print(
        "Maximum:",
        f"{float(np.max(prediction)):.10f}"
    )

    print(
        "Mean:",
        f"{float(np.mean(prediction)):.10f}"
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    all_results.append(
        {
            "filename": filename,
            "top_disease": results[0][0],
            "top_probability": float(
                results[0][1]
            ),
            "second_disease": results[1][0],
            "second_probability": float(
                results[1][1]
            ),
            "third_disease": results[2][0],
            "third_probability": float(
                results[2][1]
            ),
            "all_predictions": results
        }
    )


# ------------------------------------------------------------
# FINAL COMPARISON
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("                  FINAL COMPARISON")
print("=" * 80)

print(
    f"{'IMAGE':<18}"
    f"{'TOP FINDING':<25}"
    f"{'SCORE':>12}"
)

print(
    "-" * 60
)


for result in all_results:

    print(
        f"{result['filename']:<18}"
        f"{result['top_disease']:<25}"
        f"{result['top_probability'] * 100:>10.4f}%"
    )


# ------------------------------------------------------------
# CHECK WHETHER MODEL RESPONDS DIFFERENTLY
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("                MODEL BEHAVIOR CHECK")
print("=" * 80)


top_diseases = [
    result["top_disease"]
    for result in all_results
]

top_scores = [
    result["top_probability"]
    for result in all_results
]


unique_diseases = set(
    top_diseases
)


if len(unique_diseases) > 1:

    print(
        "\n✓ MODEL IS RESPONDING DIFFERENTLY"
    )

    print(
        "Different images produced different"
        " top findings."
    )

else:

    print(
        "\n⚠ ALL IMAGES HAVE THE SAME TOP FINDING:"
    )

    print(
        top_diseases[0]
    )


if top_scores:

    score_range = (
        max(top_scores)
        -
        min(top_scores)
    )

    print(
        "\nTop-score range:",
        f"{score_range * 100:.4f} percentage points"
    )


    if score_range > 0.01:

        print(
            "✓ Prediction scores vary between images."
        )

    else:

        print(
            "⚠ Prediction scores are extremely similar."
        )


# ------------------------------------------------------------
# FINISH
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("                  TEST COMPLETE")
print("=" * 80)
print()