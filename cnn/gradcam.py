"""
MEDVISION AI - Grad-CAM
=======================

Generates a Grad-CAM overlay for the prediction produced by
models/final_model.keras.

This version is designed for a Keras model containing a nested
DenseNet121 backbone, but also has a fallback for a normal
non-nested CNN.

Usage from app.py:

    from gradcam import generate_gradcam

    gradcam_path = generate_gradcam(
        model=model,
        image_tensor=image_tensor,
        original_image_path=image_path,
        output_path=output_path,
        class_index=class_index
    )

The output is a heatmap OVERLAID ON THE ORIGINAL X-RAY.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# FIND A 4-D FEATURE LAYER
# ============================================================

def _find_last_4d_layer(model):
    """
    Find the last spatial/feature layer.

    Prefer convolutional layers because Grad-CAM should normally
    be generated from a spatial feature representation.
    """

    # First preference: Conv2D / DepthwiseConv2D / SeparableConv2D
    preferred_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    for layer in reversed(model.layers):
        if isinstance(layer, preferred_types):
            try:
                if len(layer.output.shape) == 4:
                    return layer
            except Exception:
                pass

    # Fallback: any 4-D output
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except Exception:
            continue

    return None


def _find_densenet(model):
    """
    Find the nested DenseNet model if the outer model contains one.
    """

    # Common name used in the user's model
    try:
        layer = model.get_layer("densenet121")
        return layer
    except Exception:
        pass

    # Search by name
    for layer in model.layers:
        if "densenet" in layer.name.lower():
            return layer

    # Search by class/model structure
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "dense" in layer.name.lower():
                return layer

    return None


# ============================================================
# NESTED DENSENET GRAD-CAM
# ============================================================

def _gradcam_nested_densenet(
    model,
    image_tensor,
    class_index
):
    """
    Grad-CAM for an outer model containing a nested DenseNet.

    Returns:
        heatmap: numpy array [H, W], normalized 0..1
        target_layer_name: layer used for Grad-CAM
    """

    densenet = _find_densenet(model)

    if densenet is None:
        raise RuntimeError("Nested DenseNet model was not found.")

    print("✓ DenseNet found:", densenet.name)

    last_conv = _find_last_4d_layer(densenet)

    if last_conv is None:
        raise RuntimeError(
            "Could not find a convolutional/spatial layer inside DenseNet."
        )

    print("✓ Grad-CAM target layer:", last_conv.name)
    print("✓ Target shape:", last_conv.output.shape)

    # Model from DenseNet input to:
    # 1. target convolutional feature maps
    # 2. DenseNet final output
    inner_grad_model = tf.keras.Model(
        inputs=densenet.input,
        outputs=[
            last_conv.output,
            densenet.output
        ]
    )

    # Find the DenseNet's position in the outer model
    densenet_index = None

    for i, layer in enumerate(model.layers):
        if layer.name == densenet.name:
            densenet_index = i
            break

    if densenet_index is None:
        raise RuntimeError(
            "DenseNet was found but could not be located in outer model."
        )

    with tf.GradientTape() as tape:

        conv_outputs, dense_output = inner_grad_model(
            image_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        x = dense_output

        # Continue through every layer after DenseNet.
        for layer in model.layers[densenet_index + 1:]:

            # Skip InputLayer if present
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            x = layer(
                x,
                training=False
            )

        predictions = x

        if len(predictions.shape) != 2:
            raise RuntimeError(
                f"Unexpected prediction shape: {predictions.shape}"
            )

        number_of_classes = int(predictions.shape[-1])

        if class_index < 0 or class_index >= number_of_classes:
            raise ValueError(
                f"class_index={class_index} is invalid. "
                f"Model has {number_of_classes} classes."
            )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients are None. "
            "The selected feature layer is not connected to the prediction."
        )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    gradients = gradients[0]

    # Global average pooling of gradients
    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1)
    )

    # Weighted combination of feature maps
    cam = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    # ReLU
    cam = tf.maximum(
        cam,
        0
    )

    cam = cam.numpy()

    # Normalize
    cam_min = float(np.min(cam))
    cam_max = float(np.max(cam))

    print("CAM min:", cam_min)
    print("CAM max:", cam_max)

    if cam_max <= 1e-12:
        raise RuntimeError(
            "Grad-CAM activation is zero. "
            "The model did not produce usable spatial activation "
            "for this class."
        )

    cam = cam / cam_max
    cam = np.clip(cam, 0.0, 1.0)

    return cam, last_conv.name


# ============================================================
# STANDARD MODEL FALLBACK
# ============================================================

def _gradcam_standard_model(
    model,
    image_tensor,
    class_index
):
    """
    Fallback for a standard non-nested Keras CNN.
    """

    last_conv = _find_last_4d_layer(model)

    if last_conv is None:
        raise RuntimeError(
            "No 4-D convolutional/spatial layer was found."
        )

    print("✓ Standard Grad-CAM target:", last_conv.name)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            last_conv.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False
        )

        if len(predictions.shape) != 2:
            raise RuntimeError(
                f"Unexpected prediction shape: {predictions.shape}"
            )

        number_of_classes = int(predictions.shape[-1])

        if class_index < 0 or class_index >= number_of_classes:
            raise ValueError(
                f"class_index={class_index} is invalid. "
                f"Model has {number_of_classes} classes."
            )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError("Gradients are None.")

    conv_outputs = conv_outputs[0]
    gradients = gradients[0]

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1)
    )

    cam = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    cam = tf.maximum(
        cam,
        0
    ).numpy()

    cam_max = float(np.max(cam))

    if cam_max <= 1e-12:
        raise RuntimeError(
            "Grad-CAM activation is zero."
        )

    cam = cam / cam_max
    cam = np.clip(cam, 0.0, 1.0)

    return cam, last_conv.name


# ============================================================
# GENERATE OVERLAY
# ============================================================

def generate_gradcam(
    model,
    image_tensor,
    original_image_path,
    output_path,
    class_index
):
    """
    Generate a Grad-CAM overlay on the original X-ray.

    Parameters
    ----------
    model:
        Loaded Keras model.

    image_tensor:
        Preprocessed image with shape (1, 224, 224, 3).

    original_image_path:
        Path to the original uploaded X-ray.

    output_path:
        Where the Grad-CAM overlay should be saved.

    class_index:
        Index of the disease being explained.

    Returns
    -------
    str
        Absolute path of the generated Grad-CAM image.
    """

    print()
    print("=" * 70)
    print("                  MEDVISION AI")
    print("                   GRAD-CAM")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate image tensor
    # --------------------------------------------------------

    if image_tensor is None:
        raise ValueError("image_tensor is None.")

    if len(image_tensor.shape) != 4:
        raise ValueError(
            f"Expected image tensor with 4 dimensions, "
            f"got {image_tensor.shape}"
        )

    print("Image tensor:", image_tensor.shape)

    # --------------------------------------------------------
    # Validate original image
    # --------------------------------------------------------

    original_path = Path(original_image_path)

    if not original_path.exists():
        raise FileNotFoundError(
            f"Original X-ray not found:\n{original_path}"
        )

    original = cv2.imread(
        str(original_path),
        cv2.IMREAD_COLOR
    )

    if original is None:
        raise RuntimeError(
            f"Could not read original X-ray:\n{original_path}"
        )

    original_height, original_width = original.shape[:2]

    print(
        "Original image:",
        f"{original_width} x {original_height}"
    )

    # --------------------------------------------------------
    # Choose Grad-CAM implementation
    # --------------------------------------------------------

    densenet = _find_densenet(model)

    if densenet is not None:

        print("Model type: nested DenseNet")
        cam, target_layer = _gradcam_nested_densenet(
            model=model,
            image_tensor=image_tensor,
            class_index=class_index
        )

    else:

        print("Model type: standard CNN")
        cam, target_layer = _gradcam_standard_model(
            model=model,
            image_tensor=image_tensor,
            class_index=class_index
        )

    # --------------------------------------------------------
    # Resize CAM to original X-ray size
    # --------------------------------------------------------

    cam = cv2.resize(
        cam,
        (original_width, original_height),
        interpolation=cv2.INTER_CUBIC
    )

    cam = np.clip(
        cam,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Improve visualization
    # --------------------------------------------------------

    # Suppress extremely weak activation.
    # This prevents the entire X-ray from becoming colored.
    threshold = 0.20

    activation = np.maximum(
        cam - threshold,
        0.0
    )

    if np.max(activation) > 0:
        activation = activation / np.max(activation)
    else:
        activation = cam

    # Slightly smooth the activation
    activation = cv2.GaussianBlur(
        activation.astype(np.float32),
        (0, 0),
        sigmaX=5
    )

    activation = np.clip(
        activation,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Create color heatmap
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        activation * 255
    )

    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Original X-ray
    # --------------------------------------------------------

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    original_float = original_rgb.astype(
        np.float32
    )

    heatmap_float = heatmap_rgb.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Alpha based on activation
    # --------------------------------------------------------

    alpha = (
        0.65 * activation
    )

    # Never allow complete replacement of X-ray
    alpha = np.clip(
        alpha,
        0.0,
        0.65
    )

    alpha = alpha[..., np.newaxis]

    # --------------------------------------------------------
    # Blend
    # --------------------------------------------------------

    overlay = (
        original_float * (1.0 - alpha)
        +
        heatmap_float * alpha
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    output_path = Path(
        output_path
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(
        overlay
    ).save(
        str(output_path)
    )

    # --------------------------------------------------------
    # Also save pure heatmap
    # --------------------------------------------------------

    pure_heatmap_path = (
        output_path.parent /
        f"{output_path.stem}_heatmap.png"
    )

    cv2.imwrite(
        str(pure_heatmap_path),
        heatmap_bgr
    )

    # --------------------------------------------------------
    # Save grayscale CAM
    # --------------------------------------------------------

    grayscale_path = (
        output_path.parent /
        f"{output_path.stem}_activation.png"
    )

    cv2.imwrite(
        str(grayscale_path),
        heatmap_uint8
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if not output_path.exists():
        raise RuntimeError(
            "Grad-CAM overlay was not created."
        )

    print()
    print("✓ Grad-CAM COMPLETE")
    print("✓ Target layer:", target_layer)
    print("✓ Overlay:", output_path)
    print("✓ Heatmap:", pure_heatmap_path)
    print("✓ Activation:", grayscale_path)
    print()
    print("=" * 70)

    return str(output_path)


# ============================================================
# OPTIONAL DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        """
Grad-CAM module loaded.

Do not run this file by itself unless you add your model-loading
and image-loading code.

Normally app.py should call:

    generate_gradcam(
        model,
        image_tensor,
        original_image_path,
        output_path,
        class_index
    )
"""
    )