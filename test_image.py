"""
Standalone Test Script — images/image.png
=========================================
Tests a single license plate image WITHOUT requiring YOLO.
Works even before CNN models are trained.

Pipeline:
  1. Load images/image.png
  2. Laplacian blur detection
  3. HSV rule-based color classification (fallback)
  4. EasyOCR text extraction
  5. CNN inference (if models exist)
  6. Print full JSON result
  7. Save annotated image → images/test_result.png
"""

import cv2
import numpy as np
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# ── Optional CNN ──────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("  [WARNING] TensorFlow not found — CNN inference will be skipped")

# ── Optional EasyOCR ──────────────────────────────────────────────────────────
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("  [WARNING] easyocr not found — OCR will be skipped")

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR  = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "images" / "image.png"
OUTPUT_IMG = BASE_DIR / "images" / "test_result.png"

MODEL_PATHS = {
    'blur':  BASE_DIR / "blur_classifier_final.h5",
    'color': BASE_DIR / "color_classifier_final.h5",
    'font':  BASE_DIR / "font_classifier_final.h5",
}

CLASS_MAPS = {
    'blur':  {0: 'clear',  1: 'blurry'},
    'color': {0: 'yellow', 1: 'white', 2: 'green'},
    'font':  {0: 'black',  1: 'white', 2: 'red',   3: 'green'},
}

VEHICLE_TYPE = {
    'yellow': 'Private Vehicle',
    'white':  'Commercial / Government',
    'green':  'Government Vehicle',
}

CNN_INPUT_SIZE = (224, 224)

# =============================================================================
# RULE-BASED CLASSIFIERS (fallback / cross-check)
# =============================================================================

def classify_blur_laplacian(image: np.ndarray,
                             threshold: float = 80.0) -> dict:
    """Blur detection via Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    status = 'clear' if score > threshold else 'blurry'
    # Confidence: sigmoid mapping around threshold
    conf = 1.0 / (1.0 + np.exp(-(score - threshold) / 30.0))
    if status == 'blurry':
        conf = 1.0 - conf
    return {
        'status': status,
        'confidence': round(float(conf), 4),
        'laplacian_score': round(float(score), 2),
    }


def classify_color_hsv(image: np.ndarray) -> dict:
    """
    Rule-based plate color classification using HSV histograms.
    Handles yellow, white, and green Nepali plates.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Masks for each color range
    masks = {
        # Yellow: Hue 15–40
        'yellow': cv2.inRange(hsv, (15, 60, 60), (40, 255, 255)),
        # White: low saturation, high value
        'white':  cv2.inRange(hsv, (0, 0, 160), (180, 50, 255)),
        # Green: Hue 40–90
        'green':  cv2.inRange(hsv, (40, 60, 40), (90, 255, 200)),
    }

    total_pixels = image.shape[0] * image.shape[1]
    scores = {c: int(m.sum() / 255) / max(total_pixels, 1)
              for c, m in masks.items()}

    best_color = max(scores, key=scores.get)
    best_score = scores[best_color]

    # Normalise scores to confidence
    total = sum(scores.values()) or 1.0
    confidence = best_score / total if total > 0 else 0.0

    return {
        'color': best_color,
        'confidence': round(float(confidence), 4),
        'scores': {k: round(v, 4) for k, v in scores.items()},
        'vehicle_type': VEHICLE_TYPE.get(best_color, 'Unknown'),
    }


def classify_font_hsv(image: np.ndarray) -> dict:
    """
    Rule-based font color detection.
    Focuses on text-sized dark/bright pixels.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to find text pixels (darker or brighter than background)
    _, dark_mask  = cv2.threshold(gray, 80,  255, cv2.THRESH_BINARY_INV)
    _, light_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    masks = {
        'black': dark_mask,
        'white': light_mask,
        'red':   cv2.inRange(hsv, (0, 80, 80), (15, 255, 255)),
        'green': cv2.inRange(hsv, (40, 80, 40), (90, 255, 200)),
    }

    total = image.shape[0] * image.shape[1]
    scores = {c: int(m.sum() / 255) / max(total, 1) for c, m in masks.items()}
    best_font = max(scores, key=scores.get)
    best_score = scores[best_font]
    denom = sum(scores.values()) or 1.0

    return {
        'color': best_font,
        'confidence': round(float(best_score / denom), 4),
        'scores': {k: round(v, 4) for k, v in scores.items()},
    }


# =============================================================================
# CNN INFERENCE
# =============================================================================

def load_cnn_models() -> dict:
    """Load available trained CNN models."""
    models = {}
    if not TF_AVAILABLE:
        return models

    for name, path in MODEL_PATHS.items():
        # Also try .keras extension
        keras_path = Path(str(path).replace('.h5', '.keras'))
        load_path = keras_path if keras_path.exists() else path

        if load_path.exists():
            try:
                models[name] = keras.models.load_model(str(load_path),
                                                        compile=False)
                print(f"  ✓ Loaded {name} model from {load_path.name}")
            except Exception as e:
                print(f"  ✗ Failed to load {name} model: {e}")
        else:
            print(f"  ⚠  {name} model not found ({load_path.name}) — "
                  f"using rule-based fallback")

    return models


def preprocess_for_cnn(image: np.ndarray,
                        size: tuple = CNN_INPUT_SIZE) -> np.ndarray:
    """Resize, normalise [0,1], add batch dimension."""
    resized = cv2.resize(image, size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)


def predict_cnn(model, image: np.ndarray, class_map: dict) -> dict:
    """Run CNN inference with test-time augmentation (TTA × 4)."""
    h, w = image.shape[:2]

    # TTA: original + hflip + brightness up + brightness down
    variants = [
        image,
        cv2.flip(image, 1),
        cv2.convertScaleAbs(image, alpha=1.15, beta=10),
        cv2.convertScaleAbs(image, alpha=0.85, beta=-10),
    ]

    preds = []
    for v in variants:
        inp = preprocess_for_cnn(v)
        p   = model.predict(inp, verbose=0)[0]
        preds.append(p)

    avg = np.mean(preds, axis=0)
    cls = int(np.argmax(avg))
    return {
        'label':      class_map[cls],
        'confidence': round(float(np.max(avg)), 4),
        'all_probs':  {class_map[i]: round(float(avg[i]), 4)
                       for i in range(len(avg))},
    }


# =============================================================================
# OCR
# =============================================================================

def extract_text_ocr(image: np.ndarray) -> dict:
    """Extract plate text using EasyOCR."""
    if not OCR_AVAILABLE:
        return {'text': '', 'confidence': 0.0, 'raw': []}

    try:
        reader = easyocr.Reader(['ne', 'en'], gpu=False, verbose=False)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance for OCR
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        results = reader.readtext(thresh)

        valid = [(text.strip(), float(prob))
                 for _, text, prob in results
                 if prob > 0.2 and text.strip()]

        if not valid:
            return {'text': '', 'confidence': 0.0, 'raw': []}

        # Sort top-to-bottom, left-to-right
        full_text = ' '.join(t for t, _ in valid)
        avg_conf  = float(np.mean([p for _, p in valid]))

        return {
            'text':       full_text.strip(),
            'confidence': round(avg_conf, 4),
            'raw':        [{'word': t, 'conf': round(p, 4)} for t, p in valid],
        }
    except Exception as e:
        return {'text': '', 'confidence': 0.0, 'error': str(e)}


# =============================================================================
# ANNOTATED OUTPUT
# =============================================================================

COLORS = {
    'green':  (0, 200, 60),
    'red':    (0, 50, 220),
    'yellow': (0, 200, 200),
    'blue':   (200, 100, 0),
    'white':  (255, 255, 255),
    'dark':   (20, 20, 20),
}


def draw_annotated(image: np.ndarray, result: dict) -> np.ndarray:
    """Draw a rich annotation panel on the image."""
    h, w = image.shape[:2]
    panel_w = 340
    canvas = np.zeros((max(h, 420), w + panel_w, 3), dtype=np.uint8)
    canvas[:h, :w] = image

    # Panel background
    canvas[:, w:] = (30, 30, 30)

    def put(text, y, color=(220, 220, 220), scale=0.55, thickness=1):
        cv2.putText(canvas, text, (w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                    cv2.LINE_AA)

    y = 30
    put("PLATE ANALYSIS", y, COLORS['yellow'], 0.7, 2);  y += 35
    put(f"Time: {datetime.now().strftime('%H:%M:%S')}", y); y += 28

    # OCR
    put("─── OCR ───", y, (150, 150, 150)); y += 25
    ocr = result.get('ocr', {})
    put(f"Text : {ocr.get('text', 'N/A')}", y, COLORS['green']); y += 22
    put(f"Conf : {ocr.get('confidence', 0):.1%}", y); y += 30

    # Blur
    put("─── Blur Quality ───", y, (150, 150, 150)); y += 25
    blur = result.get('blur', {})
    b_color = COLORS['green'] if blur.get('status') == 'clear' else COLORS['red']
    put(f"Status: {blur.get('status', 'N/A').upper()}", y, b_color); y += 22
    put(f"Lap Score: {blur.get('laplacian_score', 0):.1f}", y); y += 22
    put(f"Conf: {blur.get('confidence', 0):.1%}", y); y += 30

    # Color
    put("─── Plate Color ───", y, (150, 150, 150)); y += 25
    col = result.get('plate_color', {})
    put(f"Color : {col.get('color', 'N/A').upper()}", y, COLORS['yellow']); y += 22
    put(f"Type  : {col.get('vehicle_type', 'N/A')}", y); y += 22
    put(f"Conf  : {col.get('confidence', 0):.1%}", y); y += 30

    # Font
    put("─── Font Color ───", y, (150, 150, 150)); y += 25
    font = result.get('font_color', {})
    put(f"Color : {font.get('color', 'N/A').upper()}", y, COLORS['yellow']); y += 22
    put(f"Conf  : {font.get('confidence', 0):.1%}", y); y += 30

    # CNN badge
    cnn_used = result.get('cnn_models_used', [])
    put("─── CNN Models ───", y, (150, 150, 150)); y += 25
    if cnn_used:
        put(f"Active: {', '.join(cnn_used)}", y, COLORS['green']); y += 22
    else:
        put("Using rule-based only", y, (180, 120, 80)); y += 22

    # Border on original image
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), COLORS['green'], 3)

    return canvas


# =============================================================================
# MAIN
# =============================================================================

def run_test(image_path: Path = IMAGE_PATH) -> dict:
    """Full test pipeline on a single plate image."""
    t_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"  PLATE TEST: {image_path.name}")
    print(f"{'=' * 60}\n")

    # 1. Load image
    if not image_path.exists():
        print(f"  ✗ Image not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ✗ Cannot read image: {image_path}")
        sys.exit(1)

    print(f"  Image size: {img.shape[1]}×{img.shape[0]} px")

    # 2. Blur detection (rule-based)
    print("\n  [1] Blur Detection (Laplacian)")
    blur_result = classify_blur_laplacian(img)
    print(f"      Status : {blur_result['status'].upper()}")
    print(f"      Score  : {blur_result['laplacian_score']}")
    print(f"      Conf   : {blur_result['confidence']:.1%}")

    # 3. Color classification (rule-based)
    print("\n  [2] Plate Color (HSV)")
    color_result = classify_color_hsv(img)
    print(f"      Color  : {color_result['color'].upper()}")
    print(f"      Type   : {color_result['vehicle_type']}")
    print(f"      Conf   : {color_result['confidence']:.1%}")

    # 4. Font color (rule-based)
    print("\n  [3] Font Color (HSV)")
    font_result = classify_font_hsv(img)
    print(f"      Color  : {font_result['color'].upper()}")
    print(f"      Conf   : {font_result['confidence']:.1%}")

    # 5. OCR
    print("\n  [4] OCR (EasyOCR)")
    ocr_result = extract_text_ocr(img)
    if ocr_result['text']:
        print(f"      Text   : {ocr_result['text']}")
        print(f"      Conf   : {ocr_result['confidence']:.1%}")
    else:
        print("      No text detected (OCR not available or low confidence)")

    # 6. CNN inference (if models are available)
    print("\n  [5] CNN Inference")
    cnn_models = load_cnn_models()
    cnn_used = []

    if 'blur' in cnn_models:
        cnn_blur = predict_cnn(cnn_models['blur'], img, CLASS_MAPS['blur'])
        print(f"      CNN Blur  : {cnn_blur['label'].upper()} "
              f"({cnn_blur['confidence']:.1%})")
        blur_result['cnn'] = cnn_blur
        cnn_used.append('blur')

    if 'color' in cnn_models:
        cnn_color = predict_cnn(cnn_models['color'], img, CLASS_MAPS['color'])
        print(f"      CNN Color : {cnn_color['label'].upper()} "
              f"({cnn_color['confidence']:.1%})")
        color_result['cnn'] = cnn_color
        cnn_used.append('color')

    if 'font' in cnn_models:
        cnn_font = predict_cnn(cnn_models['font'], img, CLASS_MAPS['font'])
        print(f"      CNN Font  : {cnn_font['label'].upper()} "
              f"({cnn_font['confidence']:.1%})")
        font_result['cnn'] = cnn_font
        cnn_used.append('font')

    total_ms = (time.time() - t_start) * 1000

    # 7. Assemble result
    result = {
        'image':            str(image_path),
        'datetime':         datetime.now().isoformat(),
        'processing_ms':    round(total_ms, 1),
        'blur':             blur_result,
        'plate_color':      color_result,
        'font_color':       font_result,
        'ocr':              ocr_result,
        'cnn_models_used':  cnn_used,
        'summary': {
            'plate_number':  ocr_result['text'],
            'plate_color':   color_result['color'],
            'vehicle_type':  color_result['vehicle_type'],
            'font_color':    font_result['color'],
            'quality':       blur_result['status'],
            'is_readable':  (blur_result['status'] == 'clear'
                              and blur_result['laplacian_score'] > 80),
        }
    }

    # 8. Print JSON summary
    print(f"\n{'=' * 60}")
    print("  RESULT (JSON)")
    print(f"{'=' * 60}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 9. Annotate and save image
    annotated = draw_annotated(img, result)
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMG), annotated)
    print(f"\n  ✓ Annotated image saved: {OUTPUT_IMG}")
    print(f"  ✓ Total time: {total_ms:.1f} ms\n")

    return result


if __name__ == "__main__":
    # Allow overriding image path via CLI: python test_image.py path/to/image.png
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else IMAGE_PATH
    run_test(img_path)
