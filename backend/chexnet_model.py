import os
import tensorflow as tf


# ============================================================
# MEDVISION AI
# DIRECT CHEXNET LOADER
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

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


# ============================================================
# CHECK WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("             MEDVISION AI")
print("       DIRECT CHEXNET MODEL")
print("=" * 70)

print("\nWeights:")
print(WEIGHTS_PATH)

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"\nCheXNet weights not found:\n{WEIGHTS_PATH}"
    )

print(
    "✓ Weights found:",
    round(
        os.path.getsize(WEIGHTS_PATH) / (1024 * 1024),
        2
    ),
    "MB"
)


# ============================================================
# BUILD DENSENET121
# ============================================================

print("\nBuilding DenseNet121...")

base_model = tf.keras.applications.DenseNet121(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

print(
    "✓ DenseNet121 created"
)


# ============================================================
# CHEXNET CLASSIFIER
# ============================================================

print("\nBuilding 14-class classifier...")

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


print(
    "✓ Classifier created"
)

print(
    "Input:",
    model.input_shape
)

print(
    "Output:",
    model.output_shape
)


# ============================================================
# LOAD ORIGINAL H5 WEIGHTS
# ============================================================

print("\nLoading original CheXNet .h5 weights...")

try:

    model.load_weights(
        WEIGHTS_PATH
    )

    print(
        "✓ ORIGINAL CHEXNET WEIGHTS LOADED"
    )

except Exception as e:

    print("\n" + "=" * 70)
    print("CHEXNET WEIGHT LOADING FAILED")
    print("=" * 70)

    print(
        "\nError:",
        repr(e)
    )

    raise


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 70)
print("MODEL VERIFICATION")
print("=" * 70)

print(
    "Input shape:",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)

print(
    "Last layer:",
    model.layers[-1].name
)

print(
    "Activation:",
    model.layers[-1].activation
)

print(
    "Total layers:",
    len(model.layers)
)


if model.output_shape[-1] != 14:

    raise RuntimeError(
        "CheXNet does not have 14 outputs."
    )


# ============================================================
# CHECK CLASSIFIER WEIGHTS
# ============================================================

classifier = model.get_layer(
    "predictions"
)

classifier_weights = classifier.get_weights()

print(
    "\nClassifier kernel:",
    classifier_weights[0].shape
)

print(
    "Classifier bias:",
    classifier_weights[1].shape
)

print(
    "Kernel mean:",
    float(
        tf.reduce_mean(
            classifier_weights[0]
        )
    )
)

print(
    "Kernel std:",
    float(
        tf.math.reduce_std(
            classifier_weights[0]
        )
    )
)

print(
    "Bias mean:",
    float(
        tf.reduce_mean(
            classifier_weights[1]
        )
    )
)


print("\n✓ CHEXNET MODEL READY")
print("=" * 70)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_chexnet(image_tensor):

    """
    Run direct CheXNet prediction.

    image_tensor must have shape:
    (1, 224, 224, 3)

    and use CheXNet/ImageNet preprocessing.
    """

    if image_tensor is None:

        raise ValueError(
            "Image tensor is None."
        )


    if len(image_tensor.shape) != 4:

        raise ValueError(
            f"Invalid tensor shape: "
            f"{image_tensor.shape}"
        )


    if image_tensor.shape[-1] != 3:

        raise ValueError(
            "CheXNet requires 3-channel RGB input."
        )


    output = model.predict(
        image_tensor,
        verbose=0
    )


    probabilities = output[0]


    results = {}

    for disease, probability in zip(
        DISEASES,
        probabilities
    ):

        results[disease] = float(
            probability
        )


    return results