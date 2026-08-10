"""
Project Configuration
Medical X-ray Interpreter
"""

import os

# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RAW_DATASET_DIR = os.path.join(DATASET_DIR, "raw")
PROCESSED_DATASET_DIR = os.path.join(DATASET_DIR, "processed")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")

REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ==========================================
# Image Settings
# ==========================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

CHANNELS = 3

# ==========================================
# Training Settings
# ==========================================

EPOCHS = 30

LEARNING_RATE = 0.0001

RANDOM_SEED = 42