import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import cv2


# ============================================================
# MEDVISION AI
# PROPER GRAD-CAM
# ============================================================


def find_last_conv_layer(densenet):

    """
    Find the last convolutional/spatial layer inside
    the nested DenseNet121 model.
    """

    for layer in reversed(densenet.layers):

        try:

            output_shape = layer.output.shape

            if len(output_shape) == 4:

                return layer

        except Exception:

            continue

    return None


# ============================================================
# MAIN GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    model,
    image_tensor,
    original_image_path,
    output_path,
    class_index
):

    print()
    print("=" * 70)
    print("                 MEDVISION AI")
    print("                 GRAD-CAM")
    print("=" * 70)


    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if image_tensor is None:

        raise ValueError(
            "Image tensor is None."
        )


    if len(image_tensor.shape) != 4:

        raise ValueError(
            f"Invalid image tensor shape: "
            f"{image_tensor.shape}"
        )


    print(
        "Image tensor:",
        image_tensor.shape
    )


    # ========================================================
    # GET DENSENET121
    # ========================================================

    try:

        densenet = model.get_layer(
            "densenet121"
        )

    except Exception:

        densenet = None

        for layer in model.layers:

            if "densenet" in layer.name.lower():

                densenet = layer

                break


    if densenet is None:

        raise RuntimeError(
            "DenseNet121 layer not found."
        )


    print(
        "DenseNet:",
        densenet.name
    )


    # ========================================================
    # FIND LAST CONVOLUTIONAL FEATURE LAYER
    # ========================================================

    last_conv = find_last_conv_layer(
        densenet
    )


    if last_conv is None:

        raise RuntimeError(
            "Could not find a convolutional feature layer "
            "inside DenseNet121."
        )


    print(
        "Grad-CAM target:",
        last_conv.name
    )


    print(
        "Target output:",
        last_conv.output.shape
    )


    # ========================================================
    # CREATE INTERNAL DENSENET GRADIENT MODEL
    # ========================================================

    try:

        densenet_grad_model = tf.keras.Model(

            inputs=densenet.input,

            outputs=[
                last_conv.output,
                densenet.output
            ]

        )

    except Exception as e:

        raise RuntimeError(
            "Could not create DenseNet Grad-CAM model: "
            + str(e)
        )


    # ========================================================
    # RUN DENSENET + GRADIENT
    # ========================================================

    with tf.GradientTape() as tape:

        conv_outputs, dense_output = (
            densenet_grad_model(
                image_tensor,
                training=False
            )
        )


        tape.watch(
            conv_outputs
        )


        # ----------------------------------------------------
        # Continue through the outer model
        # ----------------------------------------------------

        x = dense_output


        densenet_index = None


        for i, layer in enumerate(
            model.layers
        ):

            if layer.name == densenet.name:

                densenet_index = i

                break


        if densenet_index is None:

            raise RuntimeError(
                "Could not locate DenseNet in outer model."
            )


        # ----------------------------------------------------
        # Run remaining layers
        # ----------------------------------------------------

        for layer in model.layers[
            densenet_index + 1:
        ]:

            x = layer(
                x,
                training=False
            )


        predictions = x


        print(
            "Final prediction shape:",
            predictions.shape
        )


        # ----------------------------------------------------
        # Validate class index
        # ----------------------------------------------------

        number_of_classes = (
            predictions.shape[-1]
        )


        if class_index >= number_of_classes:

            raise ValueError(

                f"Invalid class index "
                f"{class_index}. "
                f"Model has "
                f"{number_of_classes} classes."

            )


        class_score = predictions[
            :,
            class_index
        ]


    # ========================================================
    # CALCULATE GRADIENT
    # ========================================================

    gradients = tape.gradient(

        class_score,

        conv_outputs

    )


    if gradients is None:

        raise RuntimeError(
            "Gradients are None."
        )


    print(
        "Gradient shape:",
        gradients.shape
    )


    # ========================================================
    # REMOVE BATCH
    # ========================================================

    conv_outputs = conv_outputs[0]

    gradients = gradients[0]


    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    weights = tf.reduce_mean(

        gradients,

        axis=(0, 1)

    )


    print(
        "Weights shape:",
        weights.shape
    )


    # ========================================================
    # WEIGHT FEATURE MAPS
    # ========================================================

    cam = tf.reduce_sum(

        conv_outputs * weights,

        axis=-1

    )


    # ========================================================
    # RELU
    # ========================================================

    cam = tf.maximum(
        cam,
        0
    )


    cam = cam.numpy()


    # ========================================================
    # NORMALIZE
    # ========================================================

    cam_min = cam.min()

    cam_max = cam.max()


    print(
        "CAM min:",
        cam_min
    )


    print(
        "CAM max:",
        cam_max
    )


    if cam_max - cam_min < 1e-8:

        raise RuntimeError(
            "Grad-CAM produced an almost uniform heatmap. "
            "The model gradients contain insufficient "
            "spatial information for this prediction."
        )


    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min
    )


    # ========================================================
    # LOAD ORIGINAL IMAGE
    # ========================================================

    original_path = Path(
        original_image_path
    )


    if not original_path.exists():

        raise FileNotFoundError(
            f"X-ray not found: {original_path}"
        )


    original = cv2.imread(
        str(original_path)
    )


    if original is None:

        raise RuntimeError(
            "Could not read original X-ray."
        )


    height, width = (
        original.shape[:2]
    )


    # ========================================================
    # RESIZE CAM
    # ========================================================

    cam = cv2.resize(

        cam,

        (
            width,
            height
        ),

        interpolation=cv2.INTER_CUBIC

    )


    # ========================================================
    # CONVERT TO 0-255
    # ========================================================

    heatmap_uint8 = np.uint8(

        255 * cam

    )


    # ========================================================
    # APPLY REAL HEATMAP
    # ========================================================

    heatmap = cv2.applyColorMap(

        heatmap_uint8,

        cv2.COLORMAP_JET

    )


    # ========================================================
    # REMOVE VERY LOW ACTIVATION
    # ========================================================

    # This prevents the entire X-ray from being painted.

    threshold = 0.35


    mask = cam >= threshold


    # Create transparency mask

    alpha = np.zeros_like(
        cam,
        dtype=np.float32
    )


    alpha[mask] = (
        cam[mask] - threshold
    ) / (
        1.0 - threshold
    )


    alpha = np.clip(
        alpha,
        0,
        1
    )


    # Make overlay stronger where attention is high

    alpha = (
        alpha * 0.75
    )


    alpha = alpha[..., np.newaxis]


    # ========================================================
    # BLEND
    # ========================================================

    original_rgb = cv2.cvtColor(

        original,

        cv2.COLOR_BGR2RGB

    )


    heatmap_rgb = cv2.cvtColor(

        heatmap,

        cv2.COLOR_BGR2RGB

    )


    blended = (

        original_rgb.astype(
            np.float32
        ) * (
            1 - alpha
        )

        +

        heatmap_rgb.astype(
            np.float32
        ) * alpha

    )


    blended = np.clip(

        blended,

        0,

        255

    ).astype(
        np.uint8
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )


    Image.fromarray(
        blended
    ).save(
        output_path
    )


    # ========================================================
    # SAVE PURE HEATMAP TOO
    # ========================================================

    pure_heatmap_path = (

        output_path.parent /

        "gradcam_heatmap.png"

    )


    cv2.imwrite(

        str(pure_heatmap_path),

        heatmap

    )


    # ========================================================
    # VERIFY
    # ========================================================

    if not output_path.exists():

        raise RuntimeError(
            "Grad-CAM output was not created."
        )


    print()
    print("✓ Grad-CAM generated")
    print(
        "✓ Overlay:",
        output_path
    )
    print(
        "✓ Heatmap:",
        pure_heatmap_path
    )
    print()
    print("=" * 70)
    print("                 GRAD-CAM COMPLETE")
    print("=" * 70)
    print()


    return str(
        output_path
    )