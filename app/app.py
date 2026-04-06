# === COMPLETE CODE BLOCK — COPY ENTIRE BLOCK ===
"""
Step 13: Streamlit DFU Classifier Web Application
Run with: streamlit run app/app.py  (from dfu_project root)

Features:
  - Upload foot image (JPG/PNG)
  - 224×224 resize + ImageNet normalisation
  - EfficientNetB0 inference
  - Grad-CAM heatmap overlay
  - Confidence meter
  - Clinical disclaimer
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import streamlit as st

# ── Resolve project root regardless of where streamlit is launched from ───────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

# ── Constants ─────────────────────────────────────────────────────
IMAGENET_MEAN = np.array(cfg.IMAGENET_MEAN, dtype=np.float32)
IMAGENET_STD  = np.array(cfg.IMAGENET_STD,  dtype=np.float32)
GRADCAM_LAYER = cfg.GRADCAM_LAYERS["efficientnetb0"]   # 'top_activation'

# Model search order — app/model/ first, then project weights dir
MODEL_SEARCH_PATHS = [
    os.path.join(APP_DIR,      "model", "efficientnetb0_best.keras"),
    os.path.join(APP_DIR,      "model", "efficientnetb0_best.h5"),
    os.path.join(PROJECT_ROOT, "models", "weights", "efficientnetb0_best.keras"),
    os.path.join(PROJECT_ROOT, "models", "weights", "efficientnetb0_best.h5"),
]


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DFU Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load EfficientNetB0 once and cache for all sessions."""
    from model_factory import build_model

    # Find weights file
    weights_file = None
    for path in MODEL_SEARCH_PATHS:
        if os.path.exists(path):
            weights_file = path
            break

    if weights_file is None:
        return None, (
            "Model weights not found. Expected one of:\n"
            + "\n".join(f"  • {p}" for p in MODEL_SEARCH_PATHS)
        )

    try:
        model = build_model("efficientnetb0")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        model.load_weights(weights_file)
        return model, os.path.basename(weights_file)
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    PIL image → normalised float32 array (1, 224, 224, 3).
    Matches exact pipeline used during training.
    """
    img_resized = pil_img.resize((224, 224), Image.BILINEAR)
    img_array   = np.array(img_resized, dtype=np.float32) / 255.0
    img_norm    = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img_norm, axis=0)


def denormalize_for_display(img_norm: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation → uint8 [0, 255] for display."""
    img = img_norm[0] * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════

def make_gradcam_heatmap(img_array: np.ndarray,
                          model: tf.keras.Model,
                          last_conv_layer_name: str = GRADCAM_LAYER,
                          pred_index: int | None = None) -> np.ndarray | None:
    """Compute Grad-CAM heatmap. Returns None if layer not found."""
    layer_names = [l.name for l in model.layers]
    if last_conv_layer_name not in layer_names:
        # Fallback
        last_conv_layer_name = model.layers[-4].name

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output,
            ],
        )
        with tf.GradientTape() as tape:
            last_conv_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads        = tape.gradient(class_channel, last_conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_output = last_conv_output[0]
        heatmap      = last_conv_output @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None


def apply_gradcam_overlay(img_uint8: np.ndarray,
                           heatmap: np.ndarray,
                           alpha: float = 0.4) -> np.ndarray:
    """Overlay jet-colourmap heatmap on image."""
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    jet_map         = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb         = cv2.cvtColor(jet_map, cv2.COLOR_BGR2RGB)
    overlay         = (1 - alpha) * img_uint8.astype(np.float32) + alpha * jet_rgb
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

def render_sidebar(model_file: str | None):
    with st.sidebar:
        st.markdown("## 🔬 DFU Classifier")
        st.markdown("**Diabetic Foot Ulcer Detection**")
        st.markdown("---")

        st.markdown("### About")
        st.markdown(
            "This tool uses **EfficientNetB0** transfer learning "
            "to classify foot images as *Ulcer* or *Normal*. "
            "Trained on DFUC2021, DFUC2020, and KDFU benchmark datasets."
        )

        st.markdown("### Model")
        st.markdown(
            f"- Architecture: EfficientNetB0\n"
            f"- Input: 224×224×3\n"
            f"- Head: GAP → Dense(256) → Dropout(0.4) → Sigmoid\n"
            f"- Target AUC: ≥ 0.95\n"
            f"- Target F1: ≥ 0.93"
        )
        if model_file:
            st.success(f"Weights loaded: `{model_file}`")
        else:
            st.error("Model weights not loaded")

        st.markdown("### Datasets")
        st.markdown(
            "- [DFUC2021](https://dfu-challenge.github.io/) — Primary\n"
            "- [DFUC2020](https://dfu-challenge.github.io/) — Cross-eval\n"
            "- [KDFU (Kaggle)](https://www.kaggle.com/) — Cross-eval"
        )

        st.markdown("### Interpretability")
        st.markdown(
            f"Grad-CAM heatmaps highlight regions the model "
            f"focuses on. Target layer: `{GRADCAM_LAYER}`"
        )

        st.markdown("---")
        st.markdown(
            "*MSc Thesis — Torrens University Australia*\n\n"
            "*ITA602 Advanced IT Work Integrated Learning*"
        )


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════

def main():
    # ── Load model ────────────────────────────────────────────────
    model, model_info = load_model()
    render_sidebar(model_info if model is not None else None)

    # ── Header ────────────────────────────────────────────────────
    st.title("🔬 Diabetic Foot Ulcer Classifier")
    st.markdown(
        "Upload a foot image to receive an AI-assisted classification. "
        "The model returns a binary prediction (**Ulcer** / **Normal**) "
        "along with a Grad-CAM attention map."
    )

    # ── Clinical disclaimer ───────────────────────────────────────
    st.warning(
        "⚠️ **CLINICAL DISCLAIMER:** This tool is a decision-support aid only. "
        "All predictions must be reviewed by a qualified healthcare professional "
        "before any clinical action is taken. "
        "Not validated for clinical use."
    )

    # ── Model error guard ─────────────────────────────────────────
    if model is None:
        st.error(f"**Model failed to load:**\n\n```\n{model_info}\n```")
        st.info(
            "Copy your trained weights file to:\n\n"
            f"`{os.path.join(APP_DIR, 'model', 'efficientnetb0_best.keras')}`\n\n"
            "or run `train_efficientnet.py` (Step 8) first."
        )
        return

    st.markdown("---")

    # ── File upload ───────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a foot image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Max size: 10 MB.",
    )

    if uploaded_file is None:
        st.markdown(
            "### How to use\n"
            "1. Click **Browse files** above\n"
            "2. Select a foot photograph (JPG or PNG)\n"
            "3. Results appear automatically\n\n"
            "The model analyses the image in under 2 seconds."
        )
        return

    # ── Load and validate image ───────────────────────────────────
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image: {e}")
        return

    original_size = pil_img.size
    img_input     = preprocess_image(pil_img)   # (1, 224, 224, 3)

    # ── Inference ─────────────────────────────────────────────────
    with st.spinner("Analysing image..."):
        prediction = float(model(img_input, training=False).numpy()[0][0])

    label      = "ULCER DETECTED"  if prediction >= cfg.CLASSIFICATION_THRESHOLD else "NORMAL FOOT"
    confidence = prediction        if prediction >= cfg.CLASSIFICATION_THRESHOLD else 1.0 - prediction
    is_ulcer   = prediction >= cfg.CLASSIFICATION_THRESHOLD

    # ── Grad-CAM ──────────────────────────────────────────────────
    with st.spinner("Generating Grad-CAM..."):
        heatmap      = make_gradcam_heatmap(img_input, model)
        img_uint8    = denormalize_for_display(img_input)
        gradcam_img  = apply_gradcam_overlay(img_uint8, heatmap) if heatmap is not None else None

    # ── Result header ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Prediction Result")

    if is_ulcer:
        st.error(f"### 🔴 {label}")
    else:
        st.success(f"### 🟢 {label}")

    # Confidence bar
    st.markdown(f"**Confidence: {confidence*100:.1f}%**")
    st.progress(float(confidence))

    col_detail1, col_detail2, col_detail3 = st.columns(3)
    with col_detail1:
        st.metric("Raw probability", f"{prediction:.4f}")
    with col_detail2:
        st.metric("Threshold", f"{cfg.CLASSIFICATION_THRESHOLD:.1f}")
    with col_detail3:
        st.metric("Original size", f"{original_size[0]}×{original_size[1]}")

    # ── Image columns ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Visual Analysis")

    col_orig, col_grad = st.columns(2)

    with col_orig:
        st.markdown("**Original Image** (resized to 224×224)")
        st.image(img_uint8, use_container_width=True)

    with col_grad:
        if gradcam_img is not None:
            st.markdown(f"**Grad-CAM Overlay** (layer: `{GRADCAM_LAYER}`)")
            st.image(gradcam_img, use_container_width=True)
            st.caption(
                "🔴 Red/yellow regions = areas the model weighted most heavily. "
                "For ulcer predictions these should localise the wound area."
            )
        else:
            st.markdown("**Grad-CAM**")
            st.warning("Grad-CAM could not be generated for this image.")

    # ── Interpretation guide ──────────────────────────────────────
    st.markdown("---")
    st.markdown("## Interpretation Guide")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown(
            "**Score ≥ 0.5 → Ulcer**\n\n"
            "The model predicts diabetic foot ulcer tissue. "
            "Clinical review is strongly recommended.\n\n"
            "**Score < 0.5 → Normal**\n\n"
            "The model predicts healthy/non-ulcerated foot tissue."
        )
    with col_g2:
        st.markdown(
            "**Confidence**\n\n"
            "- ≥ 90% — high confidence\n"
            "- 70–90% — moderate confidence\n"
            "- 50–70% — low confidence (borderline case)\n\n"
            "Borderline cases (<70% confidence) should always be "
            "escalated to clinical assessment."
        )

    # ── Footer disclaimer ─────────────────────────────────────────
    st.markdown("---")
    st.warning(
        "⚠️ **CLINICAL DISCLAIMER:** This tool is a decision-support aid only. "
        "All predictions must be reviewed by a qualified healthcare professional "
        "before any clinical action is taken. "
        "Not validated for clinical use."
    )
    st.caption(
        "MSc Thesis — Torrens University Australia | ITA602 | "
        "EfficientNetB0 Transfer Learning | DFUC2021/2020/KDFU"
    )


if __name__ == "__main__":
    main()
