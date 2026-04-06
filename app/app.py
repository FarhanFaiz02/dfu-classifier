# === COMPLETE CODE BLOCK — COPY ENTIRE BLOCK ===
"""
DFU Classifier Web Application — ONNX Runtime version
Run locally:  streamlit run app/app.py  (from dfu_project root)
Cloud:        Streamlit Community Cloud (Python 3.14, onnxruntime)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import streamlit as st

# ── Resolve project root regardless of where streamlit is launched from ──────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

# ── Constants (from config.py — hardcoded here to avoid Windows-path import) ─
IMAGENET_MEAN           = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD            = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSIFICATION_THRESHOLD = 0.5
IMG_SIZE                = (224, 224)

# ── ONNX model search paths (app/model/ first, then local weights dir) ───────
ONNX_SEARCH_PATHS = [
    os.path.join(APP_DIR,      "model", "efficientnetb0.onnx"),
    os.path.join(PROJECT_ROOT, "models", "weights", "efficientnetb0.onnx"),
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
    """Load ONNX session once and cache across all sessions."""
    import onnxruntime as ort

    onnx_path = None
    for path in ONNX_SEARCH_PATHS:
        if os.path.exists(path):
            onnx_path = path
            break

    if onnx_path is None:
        return None, (
            "ONNX model not found. Expected one of:\n"
            + "\n".join(f"  • {p}" for p in ONNX_SEARCH_PATHS)
        )

    try:
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        return session, os.path.basename(onnx_path)
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    PIL image → normalised float32 array (1, 224, 224, 3).
    Identical pipeline to training preprocessing.
    """
    img_resized = pil_img.resize(IMG_SIZE, Image.BILINEAR)
    img_array   = np.array(img_resized, dtype=np.float32) / 255.0
    img_norm    = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img_norm, axis=0)           # (1, 224, 224, 3)


def denormalize_for_display(img_norm: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation → uint8 for display."""
    img = img_norm[0] * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

def render_sidebar(model_file):
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
            "- Architecture: EfficientNetB0\n"
            "- Format: ONNX (optimised for CPU inference)\n"
            "- Input: 224×224×3\n"
            "- Head: GAP → Dense(256) → Dropout(0.4) → Sigmoid\n"
            "- Target AUC: ≥ 0.95\n"
            "- Target F1: ≥ 0.93"
        )
        if model_file:
            st.success(f"Model loaded: `{model_file}`")
        else:
            st.error("Model not loaded")

        st.markdown("### Datasets")
        st.markdown(
            "- [DFUC2021](https://dfu-challenge.github.io/) — Primary training\n"
            "- [DFUC2020](https://dfu-challenge.github.io/) — Cross-dataset eval\n"
            "- [KDFU (Kaggle)](https://www.kaggle.com/) — Cross-dataset eval"
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
    session, model_info = load_model()
    render_sidebar(model_info if session is not None else None)

    # ── Header ────────────────────────────────────────────────────
    st.title("🔬 Diabetic Foot Ulcer Classifier")
    st.markdown(
        "Upload a foot image to receive an AI-assisted classification. "
        "The model returns a binary prediction (**Ulcer** / **Normal**) "
        "with confidence score."
    )

    # ── Clinical disclaimer ───────────────────────────────────────
    st.warning(
        "⚠️ **CLINICAL DISCLAIMER:** This tool is a decision-support aid only. "
        "All predictions must be reviewed by a qualified healthcare professional "
        "before any clinical action is taken. "
        "Not validated for clinical use."
    )

    # ── Model error guard ─────────────────────────────────────────
    if session is None:
        st.error(f"**Model failed to load:**\n\n```\n{model_info}\n```")
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
    img_display   = denormalize_for_display(img_input)

    # ── Inference ─────────────────────────────────────────────────
    with st.spinner("Analysing image..."):
        input_name  = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        raw_output  = session.run([output_name], {input_name: img_input})[0]
        prediction  = float(raw_output.flatten()[0])

    is_ulcer   = prediction >= CLASSIFICATION_THRESHOLD
    label      = "ULCER DETECTED" if is_ulcer else "NORMAL FOOT"
    confidence = prediction if is_ulcer else 1.0 - prediction

    # ── Result header ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Prediction Result")

    if is_ulcer:
        st.error(f"### 🔴 {label}")
    else:
        st.success(f"### 🟢 {label}")

    st.markdown(f"**Confidence: {confidence * 100:.1f}%**")
    st.progress(float(confidence))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Raw probability", f"{prediction:.4f}")
    with col2:
        st.metric("Threshold", f"{CLASSIFICATION_THRESHOLD:.1f}")
    with col3:
        st.metric("Original size", f"{original_size[0]}×{original_size[1]}")

    # ── Image display ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Uploaded Image")
    st.image(img_display, caption="Input image (224×224, ImageNet normalised)", use_container_width=True)

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
            "Borderline cases (<70%) should always be escalated "
            "to clinical assessment."
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
