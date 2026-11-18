"""Kiến trúc các mô hình: CNN tự xây, MobileNetV2 (transfer learning), MLP trên điểm mốc."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def augmentation() -> keras.Sequential:
    """Tăng cường dữ liệu nhẹ (chỉ hoạt động khi huấn luyện). Không lật ngang để giữ nguyên dấu tay."""
    return keras.Sequential([
        layers.RandomRotation(0.06),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="augmentation")


def build_cnn(n_classes: int, input_shape=(64, 64, 3)) -> keras.Model:
    """CNN tự xây: 4 khối Conv-BN-ReLU-MaxPool + Dropout. Đầu vào uint8 0-255."""
    inp = keras.Input(input_shape)
    x = augmentation()(inp)
    x = layers.Rescaling(1 / 255.0)(x)
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn_scratch")


def build_mobilenet(n_classes: int, input_shape=(64, 64, 3), trainable_backbone=False,
                    resize_to: int = 96) -> keras.Model:
    """MobileNetV2 pretrained ImageNet. Phóng ảnh 64 -> 96 để hợp kích thước backbone."""
    base = keras.applications.MobileNetV2(input_shape=(resize_to, resize_to, 3),
                                          include_top=False, weights="imagenet")
    base.trainable = trainable_backbone
    inp = keras.Input(input_shape)
    x = augmentation()(inp)
    x = layers.Resizing(resize_to, resize_to)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="mobilenetv2")


def build_mlp(n_classes: int, n_features: int = 63) -> keras.Model:
    """MLP nhỏ trên vector 63 chiều (21 điểm mốc x 3 toạ độ đã chuẩn hóa)."""
    inp = keras.Input((n_features,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="mlp_landmarks")


def compile_model(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def default_callbacks(patience: int = 4):
    return [
        keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True,
                                      monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5),
    ]
