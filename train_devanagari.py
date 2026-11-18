
import os, random, json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Register custom layer (must match train_models.py) ──────────────────────
@keras.utils.register_keras_serializable(package="vehicle_detection")
class SqueezeExcitation(layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        c = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(max(c // self.ratio, 8), activation="relu")
        self.fc2 = layers.Dense(c, activation="sigmoid")
        self.rsh = layers.Reshape((1, 1, c))

    def call(self, x):
        se = self.gap(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.rsh(se)
        return x * se

    def get_config(self):
        return {**super().get_config(), "ratio": self.ratio}


# ── Nepali / Devanagari character sets ──────────────────────────────────────
DEV_DIGITS   = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]
DEV_DISTRICTS = {
    "ब":  "Bagmati",    "ग":  "Gandaki",    "प":  "Province 5",
    "ज":  "Janakpur",   "का": "Karnali",    "ल":  "Lumbini",
    "म":  "Madhesh",    "न":  "Narayani",   "त":  "Tanahun",
    "द":  "Dang",       "स":  "Surkhet",    "र":  "Rapti",
    "खो": "Khotang",    "ता": "Taplejung",  "धा": "Dhankuta",
}
DEV_LETTERS  = [
    "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श",
    "ष", "स", "ह",
]
LAT_DISTRICTS = ["BA", "KA", "PA", "JA", "GA", "LA", "MA",
                  "NA", "TA", "DA", "SA", "RA", "HA"]
LAT_LETTERS   = list("ABCDEFGHJKLMNPQRSTUVWXYZ")


def _dev_num(n):
    return "".join(DEV_DIGITS[int(d)] for d in str(n))


def _gen_dev_text():
    dist = random.choice(list(DEV_DISTRICTS.keys()))
    n1   = DEV_DIGITS[random.randint(1, 9)]
    let  = random.choice(DEV_LETTERS) + random.choice(DEV_LETTERS)
    n2   = _dev_num(random.randint(1000, 9999))
    return f"{dist}.{n1} {let} {n2}"


def _gen_lat_text():
    dist = random.choice(LAT_DISTRICTS)
    n1   = random.randint(1, 9)
    let  = random.choice(LAT_LETTERS) + random.choice(LAT_LETTERS)
    n2   = random.randint(1000, 9999)
    return f"{dist} {n1} {let} {n2}"


# ── Font resolution ──────────────────────────────────────────────────────────
def _get_font(size=46, devanagari=False):
    if devanagari:
        candidates = [
            # Common Devanagari fonts on macOS/Linux
            "/System/Library/Fonts/Supplemental/Devanagari MT.ttf",
            "/Library/Fonts/Devanagari MT.ttf",
            "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    # Fallback: PIL default (won't render Devanagari but avoids crash)
    return ImageFont.load_default()


# ── Colour definitions ───────────────────────────────────────────────────────
PLATE_BG = {
    "white": (245, 245, 245),
    "red":   (200, 20,  20),
    "green": (0,   120, 50),
}
PLATE_FG = {
    "black": (10,  10,  10),
    "white": (245, 245, 245),
    "red":   (200, 20,  20),
}
VEHICLE_TYPES = [
    {"bg": "white", "font": "black", "id": 0, "name": "Public"},
    {"bg": "red",   "font": "white", "id": 1, "name": "Private"},
    {"bg": "white", "font": "red",   "id": 2, "name": "Government"},
    {"bg": "green", "font": "white", "id": 3, "name": "Tourist"},
]

BLUR_TYPES = ["gaussian", "motion_h", "motion_v", "defocus"]


def _apply_blur(img, btype, strength):
    k = 2 * strength + 1
    if btype == "gaussian":
        return cv2.GaussianBlur(img, (k, k), strength * 0.5)
    elif btype == "motion_h":
        ker = np.zeros((k, k), dtype=np.float32)
        ker[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, ker)
    elif btype == "motion_v":
        ker = np.zeros((k, k), dtype=np.float32)
        ker[:, k // 2] = 1.0 / k
        return cv2.filter2D(img, -1, ker)
    elif btype == "defocus":
        ker = np.zeros((k, k), dtype=np.float32)
        cv2.circle(ker, (k // 2, k // 2), strength, 1, -1)
        s = ker.sum()
        return cv2.filter2D(img, -1, ker / s if s > 0 else ker)
    return img


def render_plate(text, bg_color, font_color,
                 devanagari=True, blur=False, noise=False):
    W, H    = 420, 130
    bg_rgb  = PLATE_BG[bg_color]
    txt_rgb = PLATE_FG[font_color]

    img  = Image.new("RGB", (W, H), bg_rgb)
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([3, 3, W - 3, H - 3], outline=(0, 0, 0), width=5)

    # Corner bolts
    for cx, cy in [(12, 12), (W - 12, 12), (12, H - 12), (W - 12, H - 12)]:
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4],
                     fill=(150, 150, 150), outline=(60, 60, 60))

    # Province header
    province = random.choice(["BAGMATI", "GANDAKI", "LUMBINI",
                               "बागमती", "गण्डकी", "लुम्बिनी",
                               "कर्णाली", "मधेश", "कोशी"])
    hdr_font = _get_font(15, devanagari=any(ord(c) > 0x900 for c in province))
    try:
        bb = draw.textbbox((0, 0), province, font=hdr_font)
        tw = bb[2] - bb[0]
    except Exception:
        tw = len(province) * 9
    draw.text(((W - tw) // 2, 6), province, fill=txt_rgb, font=hdr_font)

    # Main plate text
    main_font = _get_font(46, devanagari=devanagari)
    try:
        bb = draw.textbbox((0, 0), text, font=main_font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        tw, th = len(text) * 26, 46
    tx = (W - tw) // 2
    ty = H // 2 - th // 2 + 10
    draw.text((tx + 2, ty + 2), text, fill=(0, 0, 0, 60), font=main_font)
    draw.text((tx, ty), text, fill=txt_rgb, font=main_font)

    out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Colour jitter
    if random.random() < 0.35:
        out = cv2.convertScaleAbs(out,
                                   alpha=random.uniform(0.75, 1.30),
                                   beta=random.randint(-20, 20))

    # Noise
    if noise:
        n = int(out.size * random.uniform(0.005, 0.04))
        coords = [np.random.randint(0, d - 1, n) for d in out.shape[:2]]
        out[coords[0], coords[1]] = 255 if random.random() < 0.5 else 0

    # Blur
    if blur:
        bt       = random.choice(BLUR_TYPES)
        strength = random.choices([2, 3, 4, 5, 6, 7],
                                  weights=[1, 2, 3, 3, 2, 1])[0]
        out = _apply_blur(out, bt, strength)

    # Perspective warp
    if random.random() < 0.4:
        h, w = out.shape[:2]
        off   = random.randint(3, 18)
        src   = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst   = np.float32([
            [random.randint(0, off), random.randint(0, off)],
            [w - random.randint(0, off), random.randint(0, off)],
            [random.randint(0, off), h - random.randint(0, off)],
            [w - random.randint(0, off), h - random.randint(0, off)],
        ])
        M   = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, M, (w, h))

    return out


def _save(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# ── Dataset generation ───────────────────────────────────────────────────────
def generate_devanagari_dataset(base="dataset_dev", n_per_class=1500):
    """
    Generate a Devanagari-only dataset for bg_color and font_color classifiers.
    Also generates a script_type dataset (Latin vs Devanagari).
    """
    n_tr = int(n_per_class * 0.70)
    n_va = int(n_per_class * 0.15)
    n_te = n_per_class - n_tr - n_va
    splits = [("train", n_tr), ("val", n_va), ("test", n_te)]

    print(f"\nGenerating Devanagari dataset → {base}/")
    print(f"  {n_per_class} images/class  (train={n_tr} val={n_va} test={n_te})")

    # ── BG colour (Devanagari plates only) ──────────────────────────────────
    print("\n[1/3] Background Color (Devanagari)")
    bg_map = {"white": "black", "red": "white", "green": "white"}
    for split, n in splits:
        for bg in ["white", "red", "green"]:
            out = Path(base) / "bg_color" / split / bg
            for i in range(n):
                text = _gen_dev_text()
                img  = render_plate(text, bg, bg_map[bg],
                                    devanagari=True,
                                    blur=random.random() < 0.25,
                                    noise=random.random() < 0.25)
                _save(img, out / f"dev_{bg}_{i:05d}.jpg")
        print(f"  {split} done")

    # ── Font colour (Devanagari plates only) ─────────────────────────────────
    print("\n[2/3] Font Color (Devanagari)")
    font_bg = {"black": "white", "white": "red", "red": "white"}
    for split, n in splits:
        for fc in ["black", "white", "red"]:
            out = Path(base) / "font_color" / split / fc
            for i in range(n):
                text = _gen_dev_text()
                # Alternate: 50% red bg for white font (Private+Tourist)
                bg = font_bg[fc]
                if fc == "white" and random.random() < 0.5:
                    bg = "green"
                img  = render_plate(text, bg, fc,
                                    devanagari=True,
                                    blur=random.random() < 0.25,
                                    noise=random.random() < 0.25)
                _save(img, out / f"dev_{fc}_{i:05d}.jpg")
        print(f"  {split} done")

    # ── Script type: Latin=0 / Devanagari=1 ─────────────────────────────────
    print("\n[3/3] Script Type Classifier")
    for split, n in splits:
        # Latin plates
        out_lat = Path(base) / "script_type" / split / "latin"
        for i in range(n):
            vt  = random.choice(VEHICLE_TYPES)
            img = render_plate(_gen_lat_text(), vt["bg"], vt["font"],
                               devanagari=False,
                               blur=random.random() < 0.20,
                               noise=random.random() < 0.20)
            _save(img, out_lat / f"lat_{i:05d}.jpg")
        # Devanagari plates
        out_dev = Path(base) / "script_type" / split / "devanagari"
        for i in range(n):
            vt  = random.choice(VEHICLE_TYPES)
            img = render_plate(_gen_dev_text(), vt["bg"], vt["font"],
                               devanagari=True,
                               blur=random.random() < 0.25,
                               noise=random.random() < 0.25)
            _save(img, out_dev / f"dev_{i:05d}.jpg")
        print(f"  {split} done")

    print("\nDataset Summary:")
    for d in ["bg_color", "font_color", "script_type"]:
        dp = Path(base) / d
        if not dp.exists(): continue
        print(f"  {d}/")
        for sp in ["train", "val", "test"]:
            sdir = dp / sp
            if not sdir.exists(): continue
            total   = sum(len(list(c.glob("*.jpg"))) for c in sdir.iterdir()
                          if c.is_dir())
            classes = sorted(c.name for c in sdir.iterdir() if c.is_dir())
            print(f"    {sp:<6}: {total:>5} imgs  ({', '.join(classes)})")


# ── Model utils ──────────────────────────────────────────────────────────────
def _callbacks(name, phase):
    Path("models").mkdir(exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                      restore_best_weights=True, min_delta=0.002),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                                          patience=4, min_lr=1e-8),
        keras.callbacks.ModelCheckpoint(f"models/{name}_{phase}.keras",
                                        monitor="val_accuracy",
                                        save_best_only=True),
        keras.callbacks.TerminateOnNaN(),
    ]


def _gens(train_dir, val_dir, test_dir, batch=24, size=(224, 224)):
    aug = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=18,
        width_shift_range=0.18,
        height_shift_range=0.18,
        shear_range=0.12,
        zoom_range=0.18,
        horizontal_flip=True,
        brightness_range=[0.65, 1.40],
        channel_shift_range=25.0,
        fill_mode="nearest",
    )
    plain = ImageDataGenerator(rescale=1. / 255)
    kw = dict(target_size=size, batch_size=batch, class_mode="categorical")
    tr = aug.flow_from_directory(str(train_dir),  shuffle=True,  **kw)
    va = plain.flow_from_directory(str(val_dir),   shuffle=False, **kw)
    te = (plain.flow_from_directory(str(test_dir), shuffle=False, **kw)
          if test_dir and Path(test_dir).exists() else None)
    return tr, va, te


def _build_small_cnn(num_classes, input_shape=(224, 224, 3)):
    """Lightweight MobileNetV2-based head for script_type classifier."""
    bb  = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet",
        alpha=0.75)
    bb.trainable = False
    inp = layers.Input(shape=input_shape)
    x   = layers.Rescaling(1. / 255)(inp) if False else inp  # already rescaled
    x   = bb(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model, bb


def finetune_existing(model_path, train_dir, val_dir, test_dir,
                      name, epochs_ft=20, batch=24, unfreeze=20):
    """Load existing model and fine-tune top layers on Devanagari data."""
    print(f"\n{'='*55}\n  FINE-TUNE: {name.upper()}\n{'='*55}")
    if not Path(model_path).exists():
        print(f"  Model not found: {model_path}  — skipping"); return

    model = keras.models.load_model(model_path, compile=False)
    tr, va, te = _gens(train_dir, val_dir, test_dir, batch)
    if tr.samples == 0:
        print(f"  No images in {train_dir}"); return

    print(f"  Train: {tr.samples}  Val: {va.samples}  "
          f"Classes: {list(tr.class_indices.keys())}")

    # Unfreeze last `unfreeze` layers of the backbone
    for layer in model.layers:
        layer.trainable = True
    for layer in model.layers[:-unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3e-5, weight_decay=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )

    model.fit(tr, validation_data=va, epochs=epochs_ft,
              callbacks=_callbacks(name, "dev_finetune"), verbose=1)

    # Evaluate
    if te and te.samples > 0:
        print(f"\n  Test Evaluation:")
        res  = model.evaluate(te, verbose=0)
        mets = dict(zip(model.metrics_names, res))
        print(f"  Accuracy: {mets.get('accuracy', 0):.4f}  "
              f"AUC: {mets.get('auc', 0):.4f}")

        preds = np.argmax(model.predict(te, verbose=0), axis=1)
        names = list(te.class_indices.keys())
        print(classification_report(te.classes, preds, target_names=names))

    # Save fine-tuned model
    out_path = model_path.replace("_final.keras", "_dev_final.keras")
    model.save(out_path)
    print(f"  Saved: {out_path}")

    # Also save over the main path so test_image.py picks it up automatically
    model.save(model_path)
    print(f"  Updated: {model_path}")


def train_script_classifier(train_dir, val_dir, test_dir,
                            epochs1=30, epochs2=15, batch=24):
    """Train Latin vs Devanagari script type classifier."""
    print(f"\n{'='*55}\n  SCRIPT TYPE CLASSIFIER\n{'='*55}")
    tr, va, te = _gens(train_dir, val_dir, test_dir, batch)
    if tr.samples == 0:
        print(f"  No images in {train_dir}"); return

    model, bb = _build_small_cnn(num_classes=2)
    print(f"  Train: {tr.samples}  Val: {va.samples}  "
          f"Classes: {list(tr.class_indices.keys())}")

    print(f"\n  Phase 1 — frozen backbone ({epochs1} epochs)")
    model.fit(tr, validation_data=va, epochs=epochs1,
              callbacks=_callbacks("script_type", "phase1"), verbose=1)

    print(f"\n  Phase 2 — fine-tune top 20 layers ({epochs2} epochs)")
    bb.trainable = True
    for layer in bb.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.fit(tr, validation_data=va, epochs=epochs2,
              callbacks=_callbacks("script_type", "phase2"), verbose=1)

    if te and te.samples > 0:
        preds = np.argmax(model.predict(te, verbose=0), axis=1)
        names = list(te.class_indices.keys())
        print(classification_report(te.classes, preds, target_names=names))

    out_path = "script_type_classifier_final.keras"
    model.save(out_path)
    print(f"  Saved: {out_path}")

    # Metadata
    meta = {"model_name": "script_type_classifier",
            "class_ids": dict(te.class_indices) if te else {},
            "description": "0=devanagari 1=latin (sorted alphabetically)"}
    with open("script_type_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: script_type_metadata.json")


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow {tf.__version__} | "
          f"GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

    BASE = "dataset_dev"

    # Step 1: Generate Devanagari dataset
    generate_devanagari_dataset(base=BASE, n_per_class=1500)

    # Step 2: Fine-tune bg_color model on Devanagari data (20 extra epochs)
    finetune_existing(
        model_path="bg_color_classifier_final.keras",
        train_dir=f"{BASE}/bg_color/train",
        val_dir=f"{BASE}/bg_color/val",
        test_dir=f"{BASE}/bg_color/test",
        name="bg_color_devanagari",
        epochs_ft=20,
        unfreeze=25,
    )

    # Step 3: Fine-tune font_color model on Devanagari data (20 extra epochs)
    finetune_existing(
        model_path="font_color_classifier_final.keras",
        train_dir=f"{BASE}/font_color/train",
        val_dir=f"{BASE}/font_color/val",
        test_dir=f"{BASE}/font_color/test",
        name="font_color_devanagari",
        epochs_ft=20,
        unfreeze=25,
    )

    # Step 4: Train script-type classifier (Latin vs Devanagari)
    train_script_classifier(
        train_dir=f"{BASE}/script_type/train",
        val_dir=f"{BASE}/script_type/val",
        test_dir=f"{BASE}/script_type/test",
        epochs1=30,
        epochs2=15,
    )

    print("\n" + "=" * 55)
    print("  DEVANAGARI TRAINING COMPLETE")
    print("  Updated: bg_color_classifier_final.keras")
    print("  Updated: font_color_classifier_final.keras")
    print("  New:     script_type_classifier_final.keras")
    print("=" * 55)


if __name__ == "__main__":
    main()
