from predictor import model
from preprocess import preprocess_image
import numpy as np

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

image_path = "test_images/test.png"   # Change this to your X-ray image

image = preprocess_image(image_path)

import tensorflow as tf

logits = model.predict(image, verbose=0)[0]

prediction = tf.nn.sigmoid(logits).numpy()

print("\nPredictions:\n")

for disease, score in zip(DISEASES, prediction):
    print(f"{disease:20} : {score:.4f}")