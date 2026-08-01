import cv2
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    layers = None

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Must be registered here (same as train_models.py) so .keras files load correctly
if TF_AVAILABLE:
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

BASE_DIR   = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "images" / "image.png"

BLUR_IDS     = {0: "clear",  1: "blurry"}
BG_COLOR_IDS = {0: "white",  1: "red",   2: "green"}
FONT_IDS     = {0: "black",  1: "white", 2: "red"}

VEHICLE_RULES = {
    ("white", "black"): {"id": 0, "name": "Public Vehicle"},
    ("red",   "white"): {"id": 1, "name": "Private Vehicle"},
    ("white", "red"):   {"id": 2, "name": "Government Vehicle"},
    ("green", "white"): {"id": 3, "name": "Tourist Vehicle"},
}

MODEL_PATHS = {
    "blur":        BASE_DIR / "blur_classifier_final.keras",
    "bg":          BASE_DIR / "bg_color_classifier_final.keras",
    "font":        BASE_DIR / "font_color_classifier_final.keras",
    "script_type": BASE_DIR / "script_type_classifier_final.keras",
}

SCRIPT_IDS = {0: "devanagari", 1: "latin"}  # alphabetical order

# Common EasyOCR misreadings on Devanagari plates → corrections
DEV_CORRECTIONS = {
    "%": "प",  "NE": "ने", "8": "ट",  "$": "स",  "!": "ि",
    "@": "ा",  "#": "ी",  "&": "ु",  "*": "ू",  "(": "े",
    ")": "ै",  "0": "०",  "1": "१",  "2": "२",  "3": "३",
    "4": "४",  "5": "५",  "6": "६",  "7": "७",  "9": "९",
}

# Common EasyOCR misreadings on English/Latin plates → corrections
LATIN_CORRECTIONS = {
    "HEP": "NEP",
    "BAGIAT": "BAGMATI",
    "BAGIATI": "BAGMATI",
}


# =============================================================================
# IMAGE ENHANCEMENT (used when blur is detected)
# =============================================================================

def enhance_image(img):
    """
    Multi-stage enhancement pipeline for blurry licence plate images.
    Steps:
      1. Bilateral filter   — removes noise, preserves edges
      2. CLAHE              — fixes uneven lighting and low contrast
      3. Unsharp mask       — boosts edge sharpness
      4. Mild colour boost  — restores washed-out colours lost due to blur
    Returns the enhanced image (same dtype/shape as input).
    """
    # Step 1: Bilateral denoise (preserve plate borders and text edges)
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: CLAHE on L channel in LAB colour space
    lab   = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Step 3: Unsharp mask  (sharpen = img + alpha*(img - blur))
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    sharp   = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)

    # Step 4: Mild saturation boost to recover colour vividness
    hsv      = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    boosted  = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return boosted


def locate_plate(img):
    """
    Finds the licence plate region within a full vehicle image.

    Strategy:
      1. Look for bright (high-value) rectangular regions — plates are
         typically lighter than the surrounding car body.
      2. Filter by aspect ratio (2:1 – 7:1), minimum area, and position
         (plates rarely appear in the top 20% of the image).
      3. Return the best candidate crop.  Falls back to centre-bottom
         strip if nothing suitable is found.

    Returns (plate_crop, bounding_box_dict)
    """
    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Only search in the lower 80% (plates are never at the very top)
    roi_y0 = int(h * 0.20)
    roi    = gray[roi_y0:, :]

    # Threshold for bright regions (plates are lighter than car bodies)
    _, bright = cv2.threshold(roi, 160, 255, cv2.THRESH_BINARY)

    # Close gaps between characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 8))
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area   = (h * w) * 0.01   # at least 1% of image
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0:
            continue
        aspect = cw / ch
        area   = cw * ch
        if 1.2 <= aspect <= 7.5 and area >= min_area:
            # Absolute y in original image
            abs_y = y + roi_y0
            candidates.append((area, x, abs_y, cw, ch))

    if candidates:
        candidates.sort(reverse=True)        # biggest first
        _, x, y, cw, ch = candidates[0]
        pad = max(8, int(min(cw, ch) * 0.05))
        x1  = max(0,  x - pad)
        y1  = max(0,  y - pad)
        x2  = min(w,  x + cw + pad)
        y2  = min(h,  y + ch + pad)
        return img[y1:y2, x1:x2], {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                     "auto_detected": True}

    # Fallback: centre-bottom strip
    y1 = int(h * 0.50)
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)
    return img[y1:, x1:x2], {"x1": x1, "y1": y1, "x2": x2, "y2": h,
                               "auto_detected": False}


def detect_blur(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    status = "clear" if score > 100 else "blurry"
    raw    = float(1 / (1 + np.exp(-(score - 100) / 80)))
    conf   = min(raw, 0.99) if status == "clear" else min(1 - raw + 0.01, 0.99)
    return {
        "class_id": 0 if status == "clear" else 1,
        "status": status,
        "confidence": round(conf, 4),
        "laplacian_score": round(score, 2),
    }


# =============================================================================
# COLOUR DETECTION
# =============================================================================

def detect_bg_color(img):
    h, w  = img.shape[:2]

    def _score(region):
        hsv   = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        total = max(region.shape[0] * region.shape[1], 1)
        red   = cv2.bitwise_or(
            cv2.inRange(hsv, (0,   100, 70), (10,  255, 255)),
            cv2.inRange(hsv, (165, 100, 70), (180, 255, 255)),
        )
        green = cv2.inRange(hsv, (38, 80, 40), (88, 255, 210))
        white = cv2.bitwise_and(
            cv2.inRange(hsv, (0, 0, 160), (180, 50, 255)),
            cv2.bitwise_not(red),
        )
        return {
            "red":   int(red.sum()   / 255) / total,
            "green": int(green.sum() / 255) / total,
            "white": int(white.sum() / 255) / total,
        }

    # Whole image scores
    scores_full = _score(img)

    # Right 60% only — skips the blue Nepal emblem on split-design plates
    right_crop  = img[:, int(w * 0.40):]
    scores_right = _score(right_crop)

    # Use right-crop scores if they are more decisive (max score is higher)
    best_full  = max(scores_full.values())
    best_right = max(scores_right.values())
    scores     = scores_right if best_right > best_full * 1.15 else scores_full

    best  = max(scores, key=scores.get)
    denom = sum(scores.values()) or 1
    return {
        "class_id": {"white": 0, "red": 1, "green": 2}[best],
        "color": best,
        "confidence": min(round(scores[best] / denom, 4), 0.99),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


def detect_font_color(img, bg_color="white"):
    h    = img.shape[0]
    crop = img[int(h * 0.15):int(h * 0.85), :]

    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total = max(crop.shape[0] * crop.shape[1], 1)

    bg_masks = {
        "white": cv2.inRange(hsv, (0, 0, 160), (180, 50, 255)),
        "red":   cv2.bitwise_or(
                     cv2.inRange(hsv, (0,   100, 70), (10,  255, 255)),
                     cv2.inRange(hsv, (165, 100, 70), (180, 255, 255))),
        "green": cv2.inRange(hsv, (38, 80, 40), (88, 255, 210)),
    }
    not_bg = cv2.bitwise_not(bg_masks.get(bg_color, bg_masks["white"]))

    red_font   = cv2.bitwise_and(
        cv2.bitwise_or(
            cv2.inRange(hsv, (0,   100, 70), (10,  255, 255)),
            cv2.inRange(hsv, (165, 100, 70), (180, 255, 255)),
        ), not_bg)
    white_font = cv2.bitwise_and(
        cv2.inRange(hsv, (0, 0, 190), (180, 40, 255)), not_bg)
    black_font = cv2.bitwise_and(
        cv2.inRange(hsv, (0, 0, 0), (180, 80, 80)), not_bg)

    scores = {
        "red":   int(red_font.sum()   / 255) / total,
        "white": int(white_font.sum() / 255) / total,
        "black": int(black_font.sum() / 255) / total,
    }

    if scores["red"] >= 0.03:
        best = "red"
    elif scores["white"] > scores["black"]:
        best = "white"
    else:
        best = "black"

    denom = sum(scores.values()) or 1
    return {
        "class_id": {"black": 0, "white": 1, "red": 2}[best],
        "color": best,
        "confidence": min(round(scores[best] / denom, 4), 0.99),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


def determine_vehicle_type(bg_color, font_color):
    return VEHICLE_RULES.get((bg_color, font_color), {"id": -1, "name": "Unknown"})


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models():
    models = {}
    if not TF_AVAILABLE:
        return models
    for name, path in MODEL_PATHS.items():
        if path.exists():
            try:
                models[name] = keras.models.load_model(str(path), compile=False)
                print(f"  + {name} model loaded")
            except Exception as e:
                print(f"  ! {name}: {e}")
        else:
            print(f"  - {name}: not found, using rule-based fallback")
    return models


def preprocess(img, size=(224, 224)):
    rgb = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb.astype("float32") / 255.0, axis=0)


def predict_cnn(model, img, id_map):
    variants = [
        img,
        cv2.convertScaleAbs(img, alpha=1.15, beta=10),
        cv2.convertScaleAbs(img, alpha=0.85, beta=-10),
    ]
    probs = np.mean([model.predict(preprocess(v), verbose=0)[0]
                     for v in variants], axis=0)
    cls = int(np.argmax(probs))
    return {
        "class_id": cls,
        "label": id_map[cls],
        "confidence": min(round(float(np.max(probs)), 4), 0.99),
        "all_probabilities": {id_map[i]: round(float(p), 4)
                              for i, p in enumerate(probs)},
    }


# Singleton OCR reader — initialised once, reused across calls
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        # ne=Nepali, hi=Hindi share the same Devanagari script
        # en=Latin characters used alongside (e.g. B AA4777)
        _ocr_reader = easyocr.Reader(["ne", "hi", "en"],
                                      gpu=False, verbose=False)
    return _ocr_reader


def _preprocess_for_ocr(gray):
    """Return multiple preprocessed versions to maximise OCR accuracy."""
    # Upscale to at least 400px tall for better character recognition
    h, w = gray.shape
    if h < 100:
        scale = 100 / h
        gray  = cv2.resize(gray, (int(w * scale), 100),
                           interpolation=cv2.INTER_CUBIC)

    # Sharpen before thresholding
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1)
    sharp   = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # Pass 1: Otsu on sharpened
    _, p1 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pass 2: Adaptive gaussian
    p2 = cv2.adaptiveThreshold(sharp, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 8)

    # Pass 3: CLAHE to fix uneven lighting, then Otsu
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    _, p3 = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pass 4: Inverted (dark background → white background)
    _, p4_raw = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    p4 = cv2.bitwise_not(p4_raw)

    return [p1, p2, p3, p4]


def _run_ocr(reader, img):
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants = _preprocess_for_ocr(gray)

    results = [reader.readtext(v) for v in variants]

    # Pick variant with highest cumulative confidence
    best = max(results, key=lambda rs: sum(p for _, _, p in rs if p > 0.20))
    valid = [(t.strip(), float(p)) for _, t, p in best
             if p > 0.22 and t.strip()]
    if not valid:
        return {"text": "", "confidence": 0.0, "words": []}
    return {
        "text": " ".join(t for t, _ in valid),
        "confidence": round(float(np.mean([p for _, p in valid])), 4),
        "words": [{"word": t, "confidence": round(p, 4)} for t, p in valid],
    }


def _correct_devanagari(text):
    """
    Fix common EasyOCR misreadings on Devanagari plates.
    Runs on text that likely contains Devanagari characters.
    """
    # Only apply if text has any non-ASCII (likely Devanagari)
    if not any(ord(c) > 127 for c in text):
        return text
    corrected = text
    for wrong, right in DEV_CORRECTIONS.items():
        corrected = corrected.replace(wrong, right)
    return corrected


def _correct_latin(text):
    """
    Fix common EasyOCR misreadings on English/Latin plates.
    """
    corrected = text
    for wrong, right in LATIN_CORRECTIONS.items():
        corrected = corrected.replace(wrong, right)
    return corrected


def extract_text(img, enhanced_img=None, script="auto"):
    if not OCR_AVAILABLE:
        return {"text": "", "confidence": 0.0, "words": []}
    try:
        reader = _get_ocr_reader()

        result_raw = _run_ocr(reader, img)

        if enhanced_img is not None:
            result_enh = _run_ocr(reader, enhanced_img)
            if result_enh["confidence"] > result_raw["confidence"]:
                result_enh["source"] = "enhanced"
                best = result_enh
            else:
                result_raw["source"] = "raw"
                best = result_raw
        else:
            best = result_raw

        # Apply Devanagari correction if script detected or likely Devanagari
        if script == "devanagari" or any(ord(c) > 0x900 for c in best.get("text", "")):
            best["text"]    = _correct_devanagari(best["text"])
            best["script"]  = "devanagari"
        else:
            best["text"]    = _correct_latin(best["text"])
            best["script"]  = "latin"

        return best

    except Exception as e:
        return {"text": "", "confidence": 0.0, "error": str(e)}



# =============================================================================
# ANNOTATED OUTPUT
# =============================================================================

def draw_result(img, result, enhanced_img=None):
    h, w = img.shape[:2]
    panel = 380
    canvas = np.full((max(h, 500), w + panel, 3), 18, dtype=np.uint8)
    canvas[:h, :w] = img

    vt       = result["vehicle_type"]
    is_blurry = result["blur"]["status"] == "blurry"
    VT_COLORS = {0: (60, 200, 60), 1: (50, 100, 220),
                 2: (0, 190, 200), 3: (50, 200, 150)}
    vt_color  = VT_COLORS.get(vt["id"], (180, 180, 180))
    border_c  = (80, 80, 220) if is_blurry else vt_color
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), border_c, 3)

    # Blur indicator bar
    if is_blurry:
        cv2.rectangle(canvas, (0, h - 30), (w, h), (50, 50, 200), -1)
        cv2.putText(canvas, "BLURRY — enhanced for analysis",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 100), 1, cv2.LINE_AA)

    def txt(s, y, color=(210, 210, 210), scale=0.52, bold=False):
        cv2.putText(canvas, str(s), (w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    2 if bold else 1, cv2.LINE_AA)

    y = 30
    txt("PLATE ANALYSIS", y, (0, 210, 255), 0.70, True); y += 32
    txt(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), y, (110, 110, 110)); y += 30

    txt("Vehicle Type", y, (150, 150, 150), 0.48); y += 20
    txt(vt["name"], y, vt_color, 0.65, True); y += 28

    txt("Plate Background", y, (150, 150, 150), 0.48); y += 20
    bg = result["bg_color"]
    txt(f'{bg["color"].upper()}  [ID {bg["class_id"]}]', y, (255, 200, 50)); y += 20
    txt(f'Confidence: {bg["confidence"]:.1%}', y); y += 28

    txt("Font Color", y, (150, 150, 150), 0.48); y += 20
    fc = result["font_color"]
    txt(f'{fc["color"].upper()}  [ID {fc["class_id"]}]', y, (255, 200, 50)); y += 20
    txt(f'Confidence: {fc["confidence"]:.1%}', y); y += 28

    txt("Blur Quality", y, (150, 150, 150), 0.48); y += 20
    bl  = result["blur"]
    bc  = (60, 200, 60) if bl["status"] == "clear" else (80, 80, 220)
    txt(f'{bl["status"].upper()}  [ID {bl["class_id"]}]', y, bc); y += 20
    txt(f'Lap: {bl["laplacian_score"]}   Conf: {bl["confidence"]:.1%}', y); y += 28

    txt("OCR", y, (150, 150, 150), 0.48); y += 20
    ocr      = result["ocr"]
    plate_txt = (ocr["text"] or "N/A")[:22]
    ocr_src  = f' [{ocr.get("source", "")}]' if ocr.get("source") else ""
    txt(plate_txt + ocr_src, y, (60, 220, 80), 0.60, True); y += 20
    txt(f'Confidence: {ocr["confidence"]:.1%}', y); y += 28

    cnn = result["cnn_models_used"]
    txt("CNN Models", y, (150, 150, 150), 0.48); y += 20
    txt(", ".join(cnn) if cnn else "rule-based fallback",
        y, (60, 200, 60) if cnn else (160, 110, 50))

    return canvas


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(image_path=IMAGE_PATH):
    t0 = time.time()
    print(f"\n{'='*58}")
    print(f"  PLATE ANALYSIS: {Path(image_path).name}")
    print(f"{'='*58}")

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ERROR: Cannot read {image_path}"); sys.exit(1)
    print(f"  Size: {img.shape[1]}x{img.shape[0]} px")

    # ── Auto plate localization ──────────────────────────────────────────────
    # Always try to locate the plate. If auto-detected, use the crop.
    # Otherwise, only crop if the image is large (since smaller images might be pre-cropped plates).
    plate_box = None
    h_orig, w_orig = img.shape[:2]
    plate_crop, plate_box = locate_plate(img)
    if plate_box.get("auto_detected"):
        b = plate_box
        print(f"  Plate auto-detected at ({b['x1']},{b['y1']})–({b['x2']},{b['y2']})")
        img = plate_crop
        print(f"  Plate crop size: {img.shape[1]}x{img.shape[0]} px\n")
    elif w_orig > 800 or h_orig > 400:
        print("  No plate auto-detected — using fallback centre-bottom strip")
        img = plate_crop
        print(f"  Plate crop size: {img.shape[1]}x{img.shape[0]} px\n")
    else:
        print("  No plate auto-detected — using original image\n")

    # Step 1: Detect blur on the plate crop
    print("[1] Blur Quality")
    blur = detect_blur(img)
    print(f"    {blur['status'].upper()}  score={blur['laplacian_score']}  "
          f"conf={blur['confidence']:.1%}")

    # Step 2: If blurry, enhance → use for colour + OCR analysis
    is_blurry    = blur["status"] == "blurry"
    analysis_img = img
    enhanced_img = None

    if is_blurry:
        print("    → Applying image enhancement (bilateral + CLAHE + unsharp mask + colour boost)")
        enhanced_img = enhance_image(img)
        analysis_img = enhanced_img
        # Re-measure blur on enhanced image (for reporting)
        enh_blur = detect_blur(enhanced_img)
        print(f"    → Enhanced score={enh_blur['laplacian_score']}  "
              f"(was {blur['laplacian_score']})")

    # Step 3: Colour analysis on (possibly enhanced) image
    print("[2] Plate Background Color")
    bg = detect_bg_color(analysis_img)
    print(f"    {bg['color'].upper()}  conf={bg['confidence']:.1%}")
    print(f"    scores: {bg['scores']}")

    print("[3] Font Color")
    font = detect_font_color(analysis_img, bg_color=bg["color"])
    print(f"    {font['color'].upper()}  conf={font['confidence']:.1%}")
    print(f"    scores: {font['scores']}")

    print("[4] Vehicle Type")
    vtype = determine_vehicle_type(bg["color"], font["color"])

    # If still Unknown after enhancement, try a secondary pass with a widened
    # colour threshold (lower saturation requirement to catch washed-out plates)
    if vtype["id"] == -1 and is_blurry:
        print("    → Unknown after enhancement — trying relaxed colour pass")
        bg2   = _detect_bg_relaxed(analysis_img)
        font2 = _detect_font_relaxed(analysis_img, bg2["color"])
        vtype2 = determine_vehicle_type(bg2["color"], font2["color"])
        if vtype2["id"] != -1:
            bg, font, vtype = bg2, font2, vtype2
            print(f"    → Relaxed pass: bg={bg2['color']} font={font2['color']}")

    print(f"    ID={vtype['id']}  {vtype['name']}")

    print("[5] OCR")
    # Detect script type first so OCR is corrected properly
    detected_script = "auto"
    if "script_type" in (load_models()):
        _st = predict_cnn(load_models()["script_type"], analysis_img, SCRIPT_IDS)
        detected_script = _st["label"]
        print(f"    Script: {detected_script} ({_st['confidence']:.1%})")

    ocr = extract_text(img, enhanced_img=enhanced_img, script=detected_script)
    print(f"    '{ocr['text']}'  conf={ocr['confidence']:.1%}"
          f"  [{ocr.get('source', 'raw')}] script={ocr.get('script', '?')}")

    print("[6] CNN Inference")
    models   = load_models()
    cnn_used = []

    cnn_input = analysis_img

    if "blur" in models:
        r = predict_cnn(models["blur"], img, BLUR_IDS)
        if r["confidence"] >= 0.70:
            blur.update({"class_id": r["class_id"], "status": r["label"],
                         "confidence": r["confidence"],
                         "all_probabilities": r["all_probabilities"]})
            print(f"    CNN blur       -> {r['label']} ({r['confidence']:.1%})")
        else:
            print(f"    CNN blur (low confidence: {r['confidence']:.1%}) -> ignored, kept rule-based ({blur['status']})")
        cnn_used.append("blur")

    if "bg" in models:
        r = predict_cnn(models["bg"], cnn_input, BG_COLOR_IDS)
        if r["confidence"] >= 0.70:
            bg.update({"class_id": r["class_id"], "color": r["label"],
                       "confidence": r["confidence"],
                       "all_probabilities": r["all_probabilities"]})
            vtype = determine_vehicle_type(bg["color"], font["color"])
            print(f"    CNN bg_color   -> {r['label']} ({r['confidence']:.1%})")
        else:
            print(f"    CNN bg_color (low confidence: {r['confidence']:.1%}) -> ignored, kept rule-based ({bg['color']})")
        cnn_used.append("bg_color")

    if "font" in models:
        r = predict_cnn(models["font"], cnn_input, FONT_IDS)
        if r["confidence"] >= 0.70:
            font.update({"class_id": r["class_id"], "color": r["label"],
                         "confidence": r["confidence"],
                         "all_probabilities": r["all_probabilities"]})
            vtype = determine_vehicle_type(bg["color"], font["color"])
            print(f"    CNN font_color -> {r['label']} ({r['confidence']:.1%})")
        else:
            print(f"    CNN font_color (low confidence: {r['confidence']:.1%}) -> ignored, kept rule-based ({font['color']})")
        cnn_used.append("font_color")

    if "script_type" in models:
        r = predict_cnn(models["script_type"], analysis_img, SCRIPT_IDS)
        cnn_used.append("script_type")
        print(f"    CNN script     -> {r['label']} ({r['confidence']:.1%})")

    ms = round((time.time() - t0) * 1000, 1)

    result = {
        "timestamp":      datetime.now().isoformat(),
        "image":          str(image_path),
        "processing_ms":  ms,
        "enhanced":       is_blurry,
        "vehicle_type":   vtype,
        "blur":           blur,
        "bg_color":       bg,
        "font_color":     font,
        "ocr":            ocr,
        "cnn_models_used": cnn_used,
        "is_readable":    blur["laplacian_score"] > 60 or is_blurry,
    }

    print(f"\n{'='*58}\n  RESULT\n{'='*58}")
    print(f"  Vehicle Type : {vtype['name']} (ID {vtype['id']})")
    print(f"  BG Color     : {bg['color']} (conf {bg['confidence']:.1%})")
    print(f"  Font Color   : {font['color']} (conf {font['confidence']:.1%})")
    print(f"  Blur         : {blur['status']} (score {blur['laplacian_score']}, "
          f"conf {blur['confidence']:.1%})")
    print(f"  Plate Number : '{ocr['text']}' (conf {ocr['confidence']:.1%})")
    print(f"  Enhanced     : {is_blurry}")
    print(f"  Time         : {ms} ms")

    out_path = Path(image_path).parent / "test_result.png"
    cv2.imwrite(str(out_path), draw_result(img, result, enhanced_img))

    # Save automatically in a JSON file
    out_json = Path(image_path).parent / "test_result.json"
    try:
        import json
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  JSON Results : {out_json}")
    except Exception as e:
        print(f"  ERROR: Cannot save JSON to {out_json}: {e}")

    if enhanced_img is not None:
        enh_path = Path(image_path).parent / "test_result_enhanced.png"
        cv2.imwrite(str(enh_path), enhanced_img)
        print(f"  Enhanced img : {enh_path}")

    print(f"  Annotated    : {out_path}\n")
    return result


# =============================================================================
# RELAXED COLOUR FALLBACKS (for heavily blurry / washed-out images)
# =============================================================================

def _detect_bg_relaxed(img):
    """Lower saturation threshold — catches faded colours in blurry images."""
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = max(img.shape[0] * img.shape[1], 1)
    red   = cv2.bitwise_or(
        cv2.inRange(hsv, (0,   50, 50), (12,  255, 255)),
        cv2.inRange(hsv, (163, 50, 50), (180, 255, 255)),
    )
    green = cv2.inRange(hsv, (34, 40, 30), (92, 255, 220))
    white = cv2.bitwise_and(
        cv2.inRange(hsv, (0, 0, 130), (180, 70, 255)),
        cv2.bitwise_not(red),
    )
    scores = {
        "red":   int(red.sum()   / 255) / total,
        "green": int(green.sum() / 255) / total,
        "white": int(white.sum() / 255) / total,
    }
    best  = max(scores, key=scores.get)
    denom = sum(scores.values()) or 1
    return {
        "class_id": {"white": 0, "red": 1, "green": 2}[best],
        "color": best,
        "confidence": min(round(scores[best] / denom, 4), 0.99),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


def _detect_font_relaxed(img, bg_color="white"):
    """Lower thresholds for blurry/washed images."""
    h    = img.shape[0]
    crop = img[int(h * 0.15):int(h * 0.85), :]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total = max(crop.shape[0] * crop.shape[1], 1)

    bg_masks = {
        "white": cv2.inRange(hsv, (0, 0, 130), (180, 70, 255)),
        "red":   cv2.bitwise_or(
                     cv2.inRange(hsv, (0,   50, 50), (12,  255, 255)),
                     cv2.inRange(hsv, (163, 50, 50), (180, 255, 255))),
        "green": cv2.inRange(hsv, (34, 40, 30), (92, 255, 220)),
    }
    not_bg = cv2.bitwise_not(bg_masks.get(bg_color, bg_masks["white"]))

    red_font   = cv2.bitwise_and(
        cv2.bitwise_or(
            cv2.inRange(hsv, (0,   50, 50), (12,  255, 255)),
            cv2.inRange(hsv, (163, 50, 50), (180, 255, 255)),
        ), not_bg)
    white_font = cv2.bitwise_and(
        cv2.inRange(hsv, (0, 0, 160), (180, 60, 255)), not_bg)
    black_font = cv2.bitwise_and(
        cv2.inRange(hsv, (0, 0, 0), (180, 100, 100)), not_bg)

    scores = {
        "red":   int(red_font.sum()   / 255) / total,
        "white": int(white_font.sum() / 255) / total,
        "black": int(black_font.sum() / 255) / total,
    }
    if scores["red"] >= 0.02:
        best = "red"
    elif scores["white"] > scores["black"]:
        best = "white"
    else:
        best = "black"

    denom = sum(scores.values()) or 1
    return {
        "class_id": {"black": 0, "white": 1, "red": 2}[best],
        "color": best,
        "confidence": min(round(scores[best] / denom, 4), 0.99),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else IMAGE_PATH
    run(path)
