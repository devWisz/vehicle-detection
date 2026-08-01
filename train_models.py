import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

VEHICLE_TYPE_RULES = {
    ("white", "black"): {"id": 0, "name": "Public Vehicle"},
    ("red",   "white"): {"id": 1, "name": "Private Vehicle"},
    ("white", "red"):   {"id": 2, "name": "Government Vehicle"},
    ("green", "white"): {"id": 3, "name": "Tourist Vehicle"},
}

CLASS_IDS = {
    "blur":      {"clear": 0, "blurry": 1},
    "bg_color":  {"white": 0, "red": 1, "green": 2},
    "font_color": {"black": 0, "white": 1, "red": 2},
}


def setup_gpu():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow {tf.__version__} | GPU: "
          f"{len(tf.config.list_physical_devices('GPU')) > 0}")


class SqueezeExcitation(layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        c = input_shape[-1]
        self.gap  = layers.GlobalAveragePooling2D()
        self.fc1  = layers.Dense(max(c // self.ratio, 8), activation="relu")
        self.fc2  = layers.Dense(c, activation="sigmoid")
        self.rsh  = layers.Reshape((1, 1, c))

    def call(self, x):
        se = self.gap(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.rsh(se)
        return x * se

    def get_config(self):
        return {**super().get_config(), "ratio": self.ratio}


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        eps = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        ce = -y_true * tf.math.log(y_pred)
        w  = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(w * ce, axis=-1)
    return loss


def _build_model(backbone, num_classes, input_shape=(224, 224, 3),
                 dropout=0.4, loss=None):
    inp = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomBrightness(0.25)(x)
    x = layers.RandomContrast(0.2)(x)

    backbone.trainable = False
    x = layers.Rescaling(scale=255.0)(x)
    x = backbone(x, training=False)

    x = SqueezeExcitation(ratio=16)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout * 0.6)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=loss or keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model, backbone


def build_blur_classifier(input_shape=(224, 224, 3)):
    bb = tf.keras.applications.EfficientNetB2(
        input_shape=input_shape, include_top=False, weights="imagenet")
    return _build_model(bb, num_classes=2, input_shape=input_shape,
                        dropout=0.4, loss=focal_loss())


def build_bg_color_classifier(input_shape=(224, 224, 3)):
    bb = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights="imagenet")
    return _build_model(bb, num_classes=3, input_shape=input_shape, dropout=0.35)


def build_font_color_classifier(input_shape=(224, 224, 3)):
    bb = tf.keras.applications.EfficientNetB1(
        input_shape=input_shape, include_top=False, weights="imagenet")
    return _build_model(bb, num_classes=3, input_shape=input_shape, dropout=0.4)


def _callbacks(name, phase):
    Path("models").mkdir(exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, min_delta=0.001),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=5, min_lr=1e-8),
        keras.callbacks.ModelCheckpoint(
            f"models/{name}_{phase}.keras",
            monitor="val_accuracy", save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=f"logs/{name}/{phase}"),
        keras.callbacks.TerminateOnNaN(),
    ]


def _make_generators(train_dir, val_dir, test_dir, batch=32, size=(224, 224)):
    aug = ImageDataGenerator(
        rescale=1. / 255, rotation_range=20,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.15, zoom_range=0.2,
        horizontal_flip=True, brightness_range=[0.7, 1.3],
        channel_shift_range=20.0, fill_mode="nearest")
    plain = ImageDataGenerator(rescale=1. / 255)

    kw = dict(target_size=size, batch_size=batch, class_mode="categorical")
    tr = aug.flow_from_directory(str(train_dir), shuffle=True,  **kw)
    va = plain.flow_from_directory(str(val_dir),  shuffle=False, **kw)
    te = (plain.flow_from_directory(str(test_dir), shuffle=False, **kw)
          if test_dir and Path(test_dir).exists() else None)
    return tr, va, te


def train(name, model, backbone, train_dir, val_dir, test_dir=None,
          epochs1=30, epochs2=20, batch=32, finetune_layers=30):
    print(f"\n{'='*55}\n  {name.upper()}\n{'='*55}")

    if not Path(train_dir).exists():
        print(f"  Missing: {train_dir}"); return

    tr, va, te = _make_generators(train_dir, val_dir, test_dir, batch)
    if tr.samples == 0:
        print(f"  No images in {train_dir}"); return

    print(f"  Train: {tr.samples}  Val: {va.samples}  "
          f"Classes: {list(tr.class_indices.keys())}")

    print(f"\n  Phase 1 — frozen backbone ({epochs1} epochs)")
    model.fit(tr, validation_data=va, epochs=epochs1,
              callbacks=_callbacks(name, "phase1"), verbose=1)

    print(f"\n  Phase 2 — fine-tune top {finetune_layers} layers ({epochs2} epochs)")
    backbone.trainable = True
    for layer in backbone.layers[:-finetune_layers]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-5),
        loss=model.loss, metrics=model.metrics_names[1:])
    model.fit(tr, validation_data=va, epochs=epochs2,
              callbacks=_callbacks(name, "phase2"), verbose=1)

    if te and te.samples > 0:
        print(f"\n  Test Evaluation:")
        res = model.evaluate(te, verbose=1)
        metrics = dict(zip(model.metrics_names, res))
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}  "
              f"AUC: {metrics.get('auc', 0):.4f}")

        preds = np.argmax(model.predict(te, verbose=0), axis=1)
        names = list(te.class_indices.keys())
        print(f"\n{classification_report(te.classes, preds, target_names=names)}")

        cm = confusion_matrix(te.classes, preds)
        fig, ax = plt.subplots(figsize=(max(5, len(names) * 2), max(4, len(names) * 1.8)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name} — Confusion Matrix")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        thresh = cm.max() / 2
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"{name}_confusion_matrix.png", dpi=150)
        plt.close()

    model.save(f"{name}_final.keras")
    print(f"  Saved: {name}_final.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    Path("tflite").mkdir(exist_ok=True)
    with open(f"tflite/{name}.tflite", "wb") as f:
        f.write(converter.convert())
    print(f"  Saved: tflite/{name}.tflite")

    metadata = {
        "model_name": name,
        "class_ids": CLASS_IDS.get(name.replace("_classifier", "").replace("_final", ""), {}),
        "vehicle_type_rules": {
            f"{k[0]}+{k[1]}": v for k, v in VEHICLE_TYPE_RULES.items()
        },
        "input_shape": [224, 224, 3],
        "accuracy": float(metrics.get("accuracy", 0)) if te else None,
    }
    with open(f"{name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {name}_metadata.json")


def main():
    setup_gpu()

    blur_model,  blur_bb  = build_blur_classifier()
    bg_model,    bg_bb    = build_bg_color_classifier()
    font_model,  font_bb  = build_font_color_classifier()

    train("blur_classifier",
          blur_model, blur_bb,
          train_dir="dataset/blur_quality/train",
          val_dir="dataset/blur_quality/val",
          test_dir="dataset/blur_quality/test",
          epochs1=30, epochs2=15, finetune_layers=30)

    train("bg_color_classifier",
          bg_model, bg_bb,
          train_dir="dataset/bg_color/train",
          val_dir="dataset/bg_color/val",
          test_dir="dataset/bg_color/test",
          epochs1=30, epochs2=15, finetune_layers=25)

    train("font_color_classifier",
          font_model, font_bb,
          train_dir="dataset/font_color/train",
          val_dir="dataset/font_color/val",
          test_dir="dataset/font_color/test",
          epochs1=30, epochs2=15, finetune_layers=25)

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("  Models: blur/bg_color/font_color _final.keras")
    print("  TFLite: tflite/*.tflite")
    print("=" * 55)


if __name__ == "__main__":
    main()