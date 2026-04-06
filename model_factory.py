# === COMPLETE CODE BLOCK — COPY ENTIRE BLOCK ===
"""
Step 6: Model Factory
Builds any supported model with the standardised DFU classification head.
All models are returned UNCOMPILED — compilation happens in training scripts.

Head: GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.4) → Dense(1, sigmoid)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join("H:\\", "Torrens", "ITW602", "dfu_project"))
import config as cfg

# ── Reproducibility ───────────────────────────────────────────────
np.random.seed(cfg.RANDOM_SEED)
tf.random.set_seed(cfg.RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════
# BASE MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════

# Maps model name → (keras application class, any extra kwargs)
_REGISTRY = {
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, {}),
    "densenet121":    (tf.keras.applications.DenseNet121,    {}),
    "mobilenetv2":    (tf.keras.applications.MobileNetV2,    {}),
    "resnet50v2":     (tf.keras.applications.ResNet50V2,     {}),
    "inceptionv3":    (tf.keras.applications.InceptionV3,    {}),
    "vgg16":          (tf.keras.applications.VGG16,          {}),
}


# ══════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════

def build_model(
    model_name: str,
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    """
    Build and return an UNCOMPILED Keras model.

    Parameters
    ----------
    model_name   : one of 'efficientnetb0', 'densenet121', 'mobilenetv2',
                   'resnet50v2', 'inceptionv3', 'vgg16'
    input_shape  : always (224, 224, 3) per thesis spec
    dropout_rate : always 0.4 per thesis spec

    Returns
    -------
    tf.keras.Model — uncompiled, base.trainable = False (frozen for Phase 1)
    """
    name = model_name.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported: {sorted(_REGISTRY.keys())}"
        )

    print(f"\n[model_factory] Building '{name}' ...")
    print(f"  input_shape  = {input_shape}")
    print(f"  dropout_rate = {dropout_rate}")

    app_class, extra_kwargs = _REGISTRY[name]

    # ── 1. Base (pre-trained backbone, top removed) ───────────────
    base = app_class(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        **extra_kwargs,
    )
    base.trainable = False          # frozen for Phase 1 training

    # ── 2. Classification head ────────────────────────────────────
    #   GlobalAveragePooling2D → Dense(256,relu) → Dropout(0.4) → Dense(1,sigmoid)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = tf.keras.layers.Dense(
        cfg.DENSE_UNITS, activation="relu", name="dense_256"
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    output = tf.keras.layers.Dense(
        1, activation="sigmoid", name="output_sigmoid"
    )(x)

    model = tf.keras.Model(inputs=base.input, outputs=output, name=name)

    # ── 3. Parameter summary ──────────────────────────────────────
    total      = model.count_params()
    trainable  = sum(tf.size(v).numpy() for v in model.trainable_variables)
    frozen     = total - trainable

    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}  (head only — base frozen)")
    print(f"  Frozen params    : {frozen:,}")

    return model


# ══════════════════════════════════════════════════════════════════
# FINE-TUNING HELPER
# ══════════════════════════════════════════════════════════════════

def unfreeze_top_n_layers(model: tf.keras.Model, n: int, verbose: bool = True):
    """
    Unfreeze the top `n` layers of the model's base backbone.
    Used for Phase 2 fine-tuning.
    All BatchNormalization layers remain frozen to preserve ImageNet statistics.
    """
    # Freeze everything first, then selectively unfreeze
    for layer in model.layers:
        layer.trainable = False

    unfrozen = 0
    for layer in model.layers[-n:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            unfrozen += 1

    if verbose:
        trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
        print(f"  [unfreeze] Top {n} layers examined → {unfrozen} unfrozen "
              f"(BN layers kept frozen)")
        print(f"  [unfreeze] Trainable params now: {trainable:,}")

    return model


def unfreeze_top_fraction(model: tf.keras.Model,
                          fraction: float = 0.2,
                          verbose: bool = True):
    """
    Unfreeze the top `fraction` (0–1) of the base backbone's layers.
    Used for baseline Phase 2 fine-tuning (top 20%).
    BatchNormalization layers remain frozen.
    """
    # Identify base layers (everything except the 4-layer head we added)
    base_layers = model.layers[:-4]
    n = max(1, int(len(base_layers) * fraction))
    layers_to_unfreeze = base_layers[-n:]

    for layer in model.layers:
        layer.trainable = False

    unfrozen = 0
    for layer in layers_to_unfreeze:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            unfrozen += 1

    if verbose:
        trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
        print(f"  [unfreeze] Top {fraction*100:.0f}% of base "
              f"({n} layers) → {unfrozen} unfrozen (BN frozen)")
        print(f"  [unfreeze] Trainable params now: {trainable:,}")

    return model


# ══════════════════════════════════════════════════════════════════
# SELF-TEST  (run: python model_factory.py)
# ══════════════════════════════════════════════════════════════════

def _self_test():
    print("=" * 65)
    print("  model_factory.py — Self-Test: Build All 6 Models")
    print("=" * 65)

    dummy = tf.random.normal([1, 224, 224, 3])
    results = []

    for name in cfg.ALL_MODELS:
        try:
            model = build_model(name)

            # Forward pass
            pred = model(dummy, training=False)
            assert pred.shape == (1, 1), f"Bad output shape: {pred.shape}"
            conf = float(pred[0][0])
            assert 0.0 <= conf <= 1.0, f"Sigmoid out of range: {conf}"

            # Verify head layers by name
            layer_names = [l.name for l in model.layers]
            assert "gap"           in layer_names, "Missing GAP layer"
            assert "dense_256"     in layer_names, "Missing Dense(256) layer"
            assert "dropout"       in layer_names, "Missing Dropout layer"
            assert "output_sigmoid" in layer_names, "Missing output layer"

            total = model.count_params()
            results.append((name, "PASS", total, conf))
            print(f"  [PASS] {name:<16} params={total:>10,}  "
                  f"pred={conf:.4f}")

        except Exception as e:
            results.append((name, f"FAIL: {e}", 0, 0))
            print(f"  [FAIL] {name:<16} ERROR: {e}")

    print("\n" + "=" * 65)
    passed = sum(1 for r in results if r[1] == "PASS")
    print(f"  Results: {passed}/{len(results)} models passed")

    if passed == len(results):
        print("  ALL MODELS OK — model_factory is ready.")
        print("  Next step → python train_baselines.py  (Step 7)")
    else:
        print("  FAILURES detected — fix before proceeding to training.")
    print("=" * 65)


if __name__ == "__main__":
    _self_test()
