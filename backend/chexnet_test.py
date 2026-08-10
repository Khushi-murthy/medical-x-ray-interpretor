import os
import sys
import numpy as np
import tensorflow as tf

# ============================================================
# MEDVISION AI - CHEXNET TEST
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

WEIGHTS_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "brucechou1983_CheXNet_Keras_0.3.0_weights.h5"
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

print("\n" + "=" * 70)
print("             MEDVISION AI - CHEXNET")
print("=" * 70)

print("\nWeights:")
print(WEIGHTS_PATH)

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"\nCheXNet weights NOT FOUND:\n{WEIGHTS_PATH}"
    )

print("\n✓ CheXNet weights found")
print(
    "✓ File size:",
    round(os.path.getsize(WEIGHTS_PATH) / (1024 * 1024), 2),
    "MB"
)


# ============================================================
# BUILD EXACT CHEXNET ARCHITECTURE
# ============================================================

print("\nBuilding DenseNet121...")

base_model = tf.keras.applications.DenseNet121(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

x = base_model.output

x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling2d"
)(x)

output = tf.keras.layers.Dense(
    14,
    activation="sigmoid",
    name="predictions"
)(x)

model = tf.keras.Model(
    inputs=base_model.input,
    outputs=output,
    name="CheXNet"
)

print("✓ DenseNet121 created")
print("✓ Output shape:", model.output_shape)


# ============================================================
# LOAD CHEXNET WEIGHTS
# ============================================================

print("\nLoading CheXNet weights...")

try:

    model.load_weights(
        WEIGHTS_PATH
    )

    print("✓ CHEXNET WEIGHTS LOADED SUCCESSFULLY")

except Exception as e:

    print("\n✗ FAILED TO LOAD CHEXNET WEIGHTS")
    print("\nERROR:")
    print(e)

    raise


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n========== MODEL ==========")

print(
    "Input:",
    model.input_shape
)

print(
    "Output:",
    model.output_shape
)

print(
    "Output activation:",
    model.layers[-1].activation
)


# ============================================================
# SAVE AS MODERN KERAS MODEL
# ============================================================

KERAS_OUTPUT = os.path.join(
    PROJECT_DIR,
    "models",
    "chexnet_14class.keras"
)

print("\nSaving converted model:")

print(KERAS_OUTPUT)

model.save(
    KERAS_OUTPUT
)

print("\n✓ MODEL SAVED")
print("✓", KERAS_OUTPUT)


# ============================================================
# VERIFY RELOADING
# ============================================================

print("\nReloading saved .keras model...")

test_model = tf.keras.models.load_model(
    KERAS_OUTPUT,
    compile=False
)

print("✓ .keras MODEL RELOADED")

print(
    "Input:",
    test_model.input_shape
)

print(
    "Output:",
    test_model.output_shape
)


# ============================================================
# RANDOM SANITY TEST
# ============================================================

print("\nRunning sanity prediction...")

dummy_image = np.random.random(
    (1, 224, 224, 3)
).astype(
    np.float32
)

prediction = test_model.predict(
    dummy_image,
    verbose=0
)

print(
    "Prediction shape:",
    prediction.shape
)

print(
    "Prediction range:",
    float(prediction.min()),
    "to",
    float(prediction.max())
)


# ============================================================
# CHECK 14 OUTPUTS
# ============================================================

if prediction.shape[-1] != 14:

    raise RuntimeError(
        f"Expected 14 outputs, "
        f"got {prediction.shape[-1]}"
    )

print("\n✓ 14 disease outputs confirmed")


# ============================================================
# PRINT LABELS
# ============================================================

print("\n========== DISEASE LABELS ==========")

for i, disease in enumerate(DISEASES):

    print(
        f"{i:2d}. {disease}"
    )


print("\n" + "=" * 70)
print("          CHEXNET SETUP COMPLETE")
print("=" * 70)

print(
    "\nYour new model is:"
)

print(
    "models/chexnet_14class.keras"
)

print(
    "\nDO NOT replace predictor.py yet."
)

print(
    "First make sure this test finishes successfully."
)

print("=" * 70)