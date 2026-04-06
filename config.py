# === COMPLETE CODE BLOCK — COPY ENTIRE BLOCK ===
"""
config.py — Centralized Configuration
DFU Classification System — MSc Thesis
All scripts import from this file. Never hard-code paths or hyperparameters elsewhere.
"""

import os
import sys

# ══════════════════════════════════════════════════════════════════
# 1. ROOT DIRECTORIES
# ══════════════════════════════════════════════════════════════════

PROJECT_ROOT  = os.path.join("H:\\", "Torrens", "ITW602", "dfu_project")
DATASET_ROOT  = os.path.join("H:\\", "Torrens", "ITW602", "Dataset")

# ══════════════════════════════════════════════════════════════════
# 2. DATASET SOURCE PATHS  (read-only — never written to)
# ══════════════════════════════════════════════════════════════════

# DFUC2021 — primary training dataset (no pre-split; we stratify)
DFUC2021_ULCER  = os.path.join(DATASET_ROOT, "DFUC 2021", "Foot Ulcer")
DFUC2021_NORMAL = os.path.join(DATASET_ROOT, "DFUC 2021", "Normal")

# DFUC2020 — cross-dataset evaluation
DFUC2020_TRAIN_ULCER  = os.path.join(DATASET_ROOT, "DFUC 2020", "Train Set", "Foot Ulcer")
DFUC2020_TRAIN_NORMAL = os.path.join(DATASET_ROOT, "DFUC 2020", "Train Set", "Normal")
DFUC2020_TEST_ULCER   = os.path.join(DATASET_ROOT, "DFUC 2020", "Test Set",  "Foot Ulcer")
DFUC2020_TEST_NORMAL  = os.path.join(DATASET_ROOT, "DFUC 2020", "Test Set",  "Normal")

# KDFU (DFU Patches) — cross-dataset evaluation
KDFU_ULCER  = os.path.join(DATASET_ROOT, "DFU", "Patches", "Abnormal(Ulcer)")
KDFU_NORMAL = os.path.join(DATASET_ROOT, "DFU", "Patches", "Normal(Healthy skin)")

# Convenience dict for iteration
DATASET_SOURCES = {
    "dfuc2021": {
        "ulcer":  DFUC2021_ULCER,
        "normal": DFUC2021_NORMAL,
    },
    "dfuc2020_train": {
        "ulcer":  DFUC2020_TRAIN_ULCER,
        "normal": DFUC2020_TRAIN_NORMAL,
    },
    "dfuc2020_test": {
        "ulcer":  DFUC2020_TEST_ULCER,
        "normal": DFUC2020_TEST_NORMAL,
    },
    "kdfu": {
        "ulcer":  KDFU_ULCER,
        "normal": KDFU_NORMAL,
    },
}

# ══════════════════════════════════════════════════════════════════
# 3. PROJECT OUTPUT PATHS
# ══════════════════════════════════════════════════════════════════

# Data
DATA_RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Processed split paths (written by Step 5)
PROCESSED = {
    "dfuc2021": {
        "X_train": os.path.join(DATA_PROCESSED_DIR, "dfuc2021_X_train.npy"),
        "X_test":  os.path.join(DATA_PROCESSED_DIR, "dfuc2021_X_test.npy"),
        "y_train": os.path.join(DATA_PROCESSED_DIR, "dfuc2021_y_train.npy"),
        "y_test":  os.path.join(DATA_PROCESSED_DIR, "dfuc2021_y_test.npy"),
        "class_weights": os.path.join(DATA_PROCESSED_DIR, "dfuc2021_class_weights.npy"),
    },
    "dfuc2020": {
        "X_test":  os.path.join(DATA_PROCESSED_DIR, "dfuc2020_X_test.npy"),
        "y_test":  os.path.join(DATA_PROCESSED_DIR, "dfuc2020_y_test.npy"),
    },
    "kdfu": {
        "X_test":  os.path.join(DATA_PROCESSED_DIR, "kdfu_X_test.npy"),
        "y_test":  os.path.join(DATA_PROCESSED_DIR, "kdfu_y_test.npy"),
    },
}

# Models
WEIGHTS_DIR      = os.path.join(PROJECT_ROOT, "models", "weights")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")

def weights_path(model_name: str) -> str:
    """Returns path for best .h5 checkpoint, e.g. weights_path('efficientnetb0')"""
    return os.path.join(WEIGHTS_DIR, f"{model_name}_best.h5")

def saved_model_path(model_name: str) -> str:
    """Returns directory for SavedModel format."""
    return os.path.join(SAVED_MODELS_DIR, model_name)

# Results
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR  = os.path.join(RESULTS_DIR, "metrics")
GRADCAM_DIR  = os.path.join(RESULTS_DIR, "gradcam")

# Training logs (one CSV per model)
def training_log_path(model_name: str) -> str:
    return os.path.join(METRICS_DIR, f"{model_name}_training_log.csv")

# ══════════════════════════════════════════════════════════════════
# 4. IMAGE & PREPROCESSING CONSTANTS
# ══════════════════════════════════════════════════════════════════

IMG_SIZE     = (224, 224)          # ALWAYS 224×224 — never change
IMG_SHAPE    = (224, 224, 3)
NUM_CLASSES  = 1                   # Binary: sigmoid output

# ImageNet normalisation (applied after /255.0 rescaling)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Valid image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ══════════════════════════════════════════════════════════════════
# 5. TRAINING HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════

RANDOM_SEED  = 42
BATCH_SIZE   = 32

# ── Baseline models (Step 7) ──────────────────────────────────────
BASELINE_PHASE1_EPOCHS   = 15
BASELINE_PHASE1_LR       = 1e-4
BASELINE_PHASE2_EPOCHS   = 40
BASELINE_PHASE2_LR       = 1e-5
BASELINE_ES_PATIENCE_P1  = 5
BASELINE_ES_PATIENCE_P2  = 7

# ── EfficientNetB0 primary model (Step 8) ─────────────────────────
EFFNET_PHASE1_EPOCHS     = 15
EFFNET_PHASE1_LR         = 1e-3
EFFNET_PHASE2_EPOCHS     = 50
EFFNET_PHASE2_INITIAL_LR = 1e-4
EFFNET_UNFREEZE_LAYERS   = 30      # Unfreeze top N layers in Phase 2
EFFNET_ES_PATIENCE        = 10

# ── Architecture head (same for all models) ───────────────────────
DROPOUT_RATE   = 0.4
DENSE_UNITS    = 256

# ── Decision threshold ────────────────────────────────────────────
CLASSIFICATION_THRESHOLD = 0.5    # sigmoid ≥ 0.5 → Ulcer (class 1)

# ── Target metrics (thesis requirements) ─────────────────────────
TARGET_AUC = 0.95
TARGET_F1  = 0.93
TARGET_ACC = 0.95

# ══════════════════════════════════════════════════════════════════
# 6. MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════

# All supported model names (used in model_factory.py)
BASELINE_MODELS = [
    "densenet121",
    "mobilenetv2",
    "resnet50v2",
    "inceptionv3",
    "vgg16",
]

PRIMARY_MODEL = "efficientnetb0"

ALL_MODELS = BASELINE_MODELS + [PRIMARY_MODEL]

# Grad-CAM target layers per model
GRADCAM_LAYERS = {
    "efficientnetb0": "top_activation",
    "densenet121":    "relu",               # last BN+ReLU block
    "mobilenetv2":    "out_relu",
    "resnet50v2":     "post_relu",
    "inceptionv3":    "mixed10",
    "vgg16":          "block5_conv3",
}

# ══════════════════════════════════════════════════════════════════
# 7. EDA CONSTANTS  (Step 4)
# ══════════════════════════════════════════════════════════════════

PHASH_HASH_SIZE  = 8
PHASH_THRESHOLD  = 10              # Hamming distance ≤ 10 → duplicate
EDA_REMOVED_CSV  = os.path.join(METRICS_DIR, "eda_removed.csv")
RESOLUTION_HIST  = os.path.join(PLOTS_DIR,   "resolution_histogram.png")
SAMPLE_BATCH_IMG = os.path.join(PLOTS_DIR,   "sample_batch.png")

# ══════════════════════════════════════════════════════════════════
# 8. EVALUATION CONSTANTS  (Steps 9–11)
# ══════════════════════════════════════════════════════════════════

BOOTSTRAP_N       = 1000
BOOTSTRAP_CI      = 0.95
MODEL_COMPARE_CSV = os.path.join(METRICS_DIR, "model_comparison.csv")
BOOTSTRAP_CSV     = os.path.join(METRICS_DIR, "bootstrap_ci.csv")
CROSS_DATASET_CSV = os.path.join(METRICS_DIR, "cross_dataset_results.csv")
ROC_CURVES_IMG    = os.path.join(PLOTS_DIR,   "roc_curves_all_models.png")
PLOT_DPI          = 300

# ══════════════════════════════════════════════════════════════════
# 9. SELF-VALIDATION  (run: python config.py)
# ══════════════════════════════════════════════════════════════════

def validate():
    print("=" * 60)
    print("  config.py — Path Validation")
    print("=" * 60)

    all_ok = True

    # Check dataset sources
    print("\n[Dataset sources]")
    for dataset_name, splits in DATASET_SOURCES.items():
        for label, path in splits.items():
            exists = os.path.isdir(path)
            status = "OK" if exists else "MISSING"
            if not exists:
                all_ok = False
            exts = IMAGE_EXTENSIONS
            count = (
                sum(1 for f in os.listdir(path)
                    if os.path.splitext(f)[1].lower() in exts)
                if exists else 0
            )
            print(f"  [{status}] {dataset_name}/{label:<8} → {count:>5} imgs  {path}")

    # Check project output dirs
    print("\n[Output directories]")
    output_dirs = [
        ("weights",      WEIGHTS_DIR),
        ("saved_models", SAVED_MODELS_DIR),
        ("plots",        PLOTS_DIR),
        ("metrics",      METRICS_DIR),
        ("gradcam",      GRADCAM_DIR),
        ("processed",    DATA_PROCESSED_DIR),
    ]
    for name, path in output_dirs:
        exists = os.path.isdir(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{status}] {name:<14} → {path}")

    # Print key hyperparameters
    print("\n[Hyperparameters]")
    print(f"  Image size       : {IMG_SIZE}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Random seed      : {RANDOM_SEED}")
    print(f"  Dropout rate     : {DROPOUT_RATE}")
    print(f"  Dense units      : {DENSE_UNITS}")
    print(f"  Target AUC       : {TARGET_AUC}")
    print(f"  Target F1        : {TARGET_F1}")
    print(f"  Threshold        : {CLASSIFICATION_THRESHOLD}")
    print(f"  EfficientNet LR1 : {EFFNET_PHASE1_LR}")
    print(f"  EfficientNet LR2 : {EFFNET_PHASE2_INITIAL_LR} (CosineDecay)")
    print(f"  Unfreeze top N   : {EFFNET_UNFREEZE_LAYERS} layers")

    print("\n" + "=" * 60)
    if all_ok:
        print("  RESULT: All paths verified — config is ready.")
        print("  Next step → python eda.py  (Step 4)")
    else:
        print("  RESULT: MISSING paths found — check folder structure.")
        print("  Re-run setup_project.py if output dirs are missing.")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
