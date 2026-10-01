"""
Configuration parameters for Skin Disease CNN Classification.
Ensures consistency and reproducibility across all models.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrix"
REPORTS_DIR = RESULTS_DIR / "classification_reports"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = BASE_DIR / "logs"
REPORT_ASSETS_DIR = BASE_DIR / "report_assets"

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, CONFUSION_DIR, REPORTS_DIR, METRICS_DIR, LOGS_DIR, REPORT_ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Image & Data Settings
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
CHANNELS = 3
BATCH_SIZE = 32
RANDOM_SEED = 42

# Training Hyperparameters
INITIAL_EPOCHS = 15
INITIAL_LR = 1e-3
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LR = 1e-4

# Class Names (Alphabetical order for consistency across all models)
CLASS_NAMES = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion"
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
