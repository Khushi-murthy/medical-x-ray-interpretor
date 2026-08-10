import os
import sys
import numpy as np
import tensorflow as tf

PROJECT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

WEIGHTS = os.path.join(
    PROJECT,
    "models",
    "brucechou1983_CheXNet_Keras_0.3.0_weights.h5"
)

IMAGE = os.path.join(
    PROJECT,
    "test_images",
    "test.png"
)

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

print("\n==========================================")
print("       REAL CHEXNET DIAGNOSTIC TEST")
print("==========================================")

print("\nWeights:", WEIGHTS)
print("Image:", IMAGE)

if not os.path.exists(WEIGHTS):
    raise FileNotFoundError("CheXNet weights not found")

if not os.path.exists(IMAGE):
    raise FileNotFoundError("test.png not found")

# ------------------------------------------------------------
# EXACT CHEXNET-STYLE ARCHITECTURE
# ------------------------------------------------------------

print("\nBuilding model...")

base = tf.keras.applications.DenseNet121(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg"
)

predictions = tf.keras.layers.Dense(
    14,
    activation="sigmoid",
    name="predictions"
)(base.output)

model = tf.keras.Model(
    inputs=base.input,
    outputs=predictions
)

print("Model output:", model.output_shape)

# ------------------------------------------------------------
# LOAD WEIGHTS
# ------------------------------------------------------------

print("\nLoading weights...")

model.load_weights(WEIGHTS)

print("✓ Weights loaded")

# ------------------------------------------------------------
# INSPECT CLASSIFIER WEIGHTS
# ------------------------------------------------------------

classifier = model.get_layer("predictions")

weights, bias = classifier.get_weights()

print("\nClassifier weight shape:", weights.shape)
print("Classifier bias shape:", bias.shape)

print(
    "Classifier weights mean:",
    float(weights.mean())
)

print(
    "Classifier weights std:",
    float(weights.std())
)

print(
    "Classifier bias mean:",
    float(bias.mean())
)

print(
    "Classifier bias:",
    bias
)

# ------------------------------------------------------------
# LOAD REAL X-RAY
# ------------------------------------------------------------

print("\nLoading real X-ray...")

image_bytes = tf.io.read_file(IMAGE)

image = tf.io.decode_image(
    image_bytes,
    channels=3,
    expand_animations=False
)

image = tf.cast(
    image,
    tf.float32
)

print("Original:", image.shape)

image = tf.image.resize(
    image,
    (224, 224)
)

# IMPORTANT:
# First test the RAW [0,255] image.
# We will NOT apply preprocess_input yet.
image = tf.expand_dims(
    image,
    axis=0
)

print("Input shape:", image.shape)
print(
    "Input min:",
    float(tf.reduce_min(image))
)
print(
    "Input max:",
    float(tf.reduce_max(image))
)
print(
    "Input mean:",
    float(tf.reduce_mean(image))
)

# ------------------------------------------------------------
# PREDICTION A: RAW PIXELS
# ------------------------------------------------------------

print("\nRunning prediction A...")

p1 = model.predict(
    image,
    verbose=0
)[0]

print("\nRAW PIXEL RESULTS")

for disease, probability in sorted(
    zip(DISEASES, p1),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{disease:<22} "
        f"{probability * 100:8.3f}%"
    )

# ------------------------------------------------------------
# PREDICTION B: DENSENET PREPROCESSING
# ------------------------------------------------------------

print("\nRunning prediction B...")

processed = tf.keras.applications.densenet.preprocess_input(
    image
)

p2 = model.predict(
    processed,
    verbose=0
)[0]

print("\nDENSENET PREPROCESS RESULTS")

for disease, probability in sorted(
    zip(DISEASES, p2),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{disease:<22} "
        f"{probability * 100:8.3f}%"
    )

print("\n==========================================")
print("TEST COMPLETE")
print("==========================================")