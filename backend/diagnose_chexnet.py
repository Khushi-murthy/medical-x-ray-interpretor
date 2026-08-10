import os
import sys
import numpy as np
import tensorflow as tf

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "chexnet_14class.keras"
)

IMAGE_PATH = os.path.join(
    PROJECT_DIR,
    "test_images",
    "test.png"
)

# ------------------------------------------------------------
# DISEASES
# ------------------------------------------------------------

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"
]

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("             MEDVISION AI")
print("          CHEXNET DIAGNOSTIC")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\n✓ MODEL LOADED")

print("Input:", model.input_shape)
print("Output:", model.output_shape)
print("Layers:", len(model.layers))
print("Last layer:", model.layers[-1].name)
print("Activation:", model.layers[-1].activation)

# ------------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMAGE")
print("=" * 70)

print("Image:", IMAGE_PATH)

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

image_bytes = tf.io.read_file(
    IMAGE_PATH
)

image = tf.image.decode_image(
    image_bytes,
    channels=3,
    expand_animations=False
)

image = tf.cast(
    image,
    tf.float32
)

print("Original shape:", image.shape)
print("Original min:", float(tf.reduce_min(image)))
print("Original max:", float(tf.reduce_max(image)))
print("Original mean:", float(tf.reduce_mean(image)))

# ------------------------------------------------------------
# RESIZE
# ------------------------------------------------------------

image = tf.image.resize(
    image,
    (224, 224)
)

# ------------------------------------------------------------
# VARIANT 1
# RAW PIXELS
# ------------------------------------------------------------

raw = tf.expand_dims(
    image,
    axis=0
)

# ------------------------------------------------------------
# VARIANT 2
# 0-1
# ------------------------------------------------------------

scaled = raw / 255.0

# ------------------------------------------------------------
# VARIANT 3
# IMAGENET NORMALIZED
# ------------------------------------------------------------

mean = tf.constant(
    [0.485, 0.456, 0.406],
    dtype=tf.float32
)

std = tf.constant(
    [0.229, 0.224, 0.225],
    dtype=tf.float32
)

imagenet = (
    scaled - mean
) / std

# ------------------------------------------------------------
# PRINT INPUT STATISTICS
# ------------------------------------------------------------

def print_stats(name, tensor):

    print("\n" + name)

    print(
        "Shape:",
        tensor.shape
    )

    print(
        "Min:",
        float(tf.reduce_min(tensor))
    )

    print(
        "Max:",
        float(tf.reduce_max(tensor))
    )

    print(
        "Mean:",
        float(tf.reduce_mean(tensor))
    )

    print(
        "Std:",
        float(tf.math.reduce_std(tensor))
    )


print("\n" + "=" * 70)
print("INPUT VARIANTS")
print("=" * 70)

print_stats(
    "RAW 0-255",
    raw
)

print_stats(
    "SCALED 0-1",
    scaled
)

print_stats(
    "IMAGENET NORMALIZED",
    imagenet
)

# ------------------------------------------------------------
# CREATE FEATURE MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE EXTRACTION")
print("=" * 70)

gap_layer = model.get_layer(
    "global_average_pooling2d"
)

feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=gap_layer.output
)

# ------------------------------------------------------------
# CREATE LOGIT MODEL
# ------------------------------------------------------------

prediction_layer = model.get_layer(
    "predictions"
)

weights, bias = prediction_layer.get_weights()

print(
    "\nClassifier weights:",
    weights.shape
)

print(
    "Classifier bias:",
    bias.shape
)

print(
    "Weight mean:",
    float(np.mean(weights))
)

print(
    "Weight std:",
    float(np.std(weights))
)

print(
    "Bias mean:",
    float(np.mean(bias))
)

# ------------------------------------------------------------
# FUNCTION
# ------------------------------------------------------------

def diagnose_variant(name, tensor):

    print("\n")
    print("-" * 70)
    print(name)
    print("-" * 70)

    # Feature extraction
    features = feature_model(
        tensor,
        training=False
    )

    features_np = features.numpy()[0]

    print("\nFEATURES")

    print(
        "Shape:",
        features.shape
    )

    print(
        "Min:",
        float(np.min(features_np))
    )

    print(
        "Max:",
        float(np.max(features_np))
    )

    print(
        "Mean:",
        float(np.mean(features_np))
    )

    print(
        "Std:",
        float(np.std(features_np))
    )

    # Manually calculate logits
    logits = (
        np.matmul(
            features_np,
            weights
        )
        + bias
    )

    probabilities = (
        1.0 /
        (
            1.0 +
            np.exp(-np.clip(logits, -100, 100))
        )
    )

    print("\nLOGITS")

    print(
        "Min:",
        float(np.min(logits))
    )

    print(
        "Max:",
        float(np.max(logits))
    )

    print(
        "Mean:",
        float(np.mean(logits))
    )

    print("\nPROBABILITIES")

    for disease, probability in sorted(
        zip(DISEASES, probabilities),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{disease:<25}"
            f"{probability:.8f}"
            f"  ({probability * 100:.4f}%)"
        )

    print(
        "\nTOP:",
        DISEASES[
            int(np.argmax(probabilities))
        ]
    )

    print(
        "TOP SCORE:",
        f"{float(np.max(probabilities)) * 100:.4f}%"
    )


# ------------------------------------------------------------
# RUN ALL THREE
# ------------------------------------------------------------

diagnose_variant(
    "VARIANT A - RAW 0-255",
    raw
)

diagnose_variant(
    "VARIANT B - SCALED 0-1",
    scaled
)

diagnose_variant(
    "VARIANT C - IMAGENET NORMALIZED",
    imagenet
)

# ------------------------------------------------------------
# DIRECT MODEL PREDICTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIRECT MODEL PREDICTION")
print("=" * 70)

direct = model.predict(
    imagenet,
    verbose=0
)[0]

for disease, probability in sorted(
    zip(DISEASES, direct),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{disease:<25}"
        f"{float(probability):.8f}"
        f"  ({float(probability) * 100:.4f}%)"
    )

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)