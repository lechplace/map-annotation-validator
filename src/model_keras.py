"""
model_keras.py
--------------
EfficientNetB0 jako binarny klasyfikator patchy (OK vs NOT-OK) — wersja Keras/keras_hub.

Backend: TensorFlow (domyślny gdy zainstalowany).
Alternatywnie: KERAS_BACKEND=torch uv run python src/train_keras.py
"""

import keras
import keras_hub


def build_model_keras(num_classes: int = 2, img_size: int = 128) -> keras.Model:
    """
    EfficientNetB0 z pretrained wagami ImageNet via keras_hub.
    Głowa: GlobalAveragePooling → Dropout(0.3) → Dense(num_classes, softmax).
    """
    backbone = keras_hub.models.EfficientNetB0Backbone.from_preset(
        "efficientnet_b0_ra4_e3600_r224_in1k",
        load_weights=True,
    )

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = backbone(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="efficientnetb0_tree_road")


def load_model_keras(path: str) -> keras.Model:
    return keras.saving.load_model(path)
