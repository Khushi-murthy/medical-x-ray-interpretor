from PIL import Image
import numpy as np
import tensorflow as tf


IMAGE_SIZE = (224, 224)

# ImageNet statistics used by CheXNet
MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def preprocess_image(image_path):

    # ---------------------------------------------------------
    # 1. Load image
    # ---------------------------------------------------------
    image = Image.open(image_path)

    # ---------------------------------------------------------
    # 2. Force RGB
    # ---------------------------------------------------------
    image = image.convert("RGB")

    # ---------------------------------------------------------
    # 3. Resize to CheXNet input size
    # ---------------------------------------------------------
    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    # ---------------------------------------------------------
    # 4. Convert to float32
    # ---------------------------------------------------------
    image = np.asarray(
        image,
        dtype=np.float32
    )

    # ---------------------------------------------------------
    # 5. Convert 0-255 → 0-1
    # ---------------------------------------------------------
    image = image / 255.0

    # ---------------------------------------------------------
    # 6. ImageNet normalization
    # ---------------------------------------------------------
    image = (
        image - MEAN
    ) / STD

    # ---------------------------------------------------------
    # 7. Add batch dimension
    # ---------------------------------------------------------
    image = np.expand_dims(
        image,
        axis=0
    )

    # ---------------------------------------------------------
    # 8. TensorFlow tensor
    # ---------------------------------------------------------
    image = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )

    # ---------------------------------------------------------
    # DEBUG
    # ---------------------------------------------------------
    print("\nPREPROCESS DEBUG")
    print("-------------------------")
    print("Shape:", image.shape)
    print("Min:", float(tf.reduce_min(image)))
    print("Max:", float(tf.reduce_max(image)))
    print("Mean:", float(tf.reduce_mean(image)))
    print("Std:", float(tf.math.reduce_std(image)))

    return image