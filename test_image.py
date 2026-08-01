import cv2
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

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
    "blur": BASE_DIR / "blur_classifier_final.keras",
    "bg":   BASE_DIR / "bg_color_classifier_final.keras",
    "font": BASE_DIR / "font_color_classifier_final.keras",
}


def detect_blur(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    status = "clear" if score > 100 else "blurry"
    raw = float(1 / (1 + np.exp(-(score - 100) / 80)))
    conf = min(raw, 0.99) if status == "clear" else min(1 - raw + 0.01, 0.99)
    return {
        "class_id": 0 if status == "clear" else 1,
        "status": status,
        "confidence": round(conf, 4),
        "laplacian_score": round(score, 2),
    }


def detect_bg_color(img):
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = max(img.shape[0] * img.shape[1], 1)

    # Red plate: strong saturation in red hue band (handles hue wrap-around 0/180)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0,   100, 70), (10,  255, 255)),
        cv2.inRange(hsv, (165, 100, 70), (180, 255, 255)),
    )
    # Green plate: green hue, medium-high saturation
    green = cv2.inRange(hsv, (38, 80, 40), (88, 255, 210))
    # White plate: very low saturation, high value
    white = cv2.inRange(hsv, (0, 0, 160), (180, 50, 255))
    # Penalise white pixels that overlap with the red zone (avoid edge artifacts)
    white = cv2.bitwise_and(white, cv2.bitwise_not(red))

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


def detect_font_color(img, bg_color="white"):
    h = img.shape[0]
    # Analyse only the central 70% of height where text lives.
    # This avoids corner emblems (flags, seals) triggering false colour hits.
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

    red_font = cv2.bitwise_and(
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

    # Red font needs >= 3% of the text-zone pixels to confirm.
    # Government plates have red text across the full width — easily ≥3%.
    # Corner emblems (flags) are much smaller and won't exceed this.
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
        cv2.flip(img, 1),
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


def extract_text(img):
    if not OCR_AVAILABLE:
        return {"text": "", "confidence": 0.0, "words": []}
    try:
        reader = easyocr.Reader(["ne", "en"], gpu=False, verbose=False)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = reader.readtext(thresh)
        valid = [(t.strip(), float(p)) for _, t, p in results
                 if p > 0.25 and t.strip()]
        if not valid:
            return {"text": "", "confidence": 0.0, "words": []}
        return {
            "text": " ".join(t for t, _ in valid),
            "confidence": round(float(np.mean([p for _, p in valid])), 4),
            "words": [{"word": t, "confidence": round(p, 4)} for t, p in valid],
        }
    except Exception as e:
        return {"text": "", "confidence": 0.0, "error": str(e)}


def draw_result(img, result):
    h, w = img.shape[:2]
    panel = 370
    canvas = np.full((max(h, 480), w + panel, 3), 18, dtype=np.uint8)
    canvas[:h, :w] = img

    vt = result["vehicle_type"]
    VT_COLORS = {
        0: (60,  200, 60),   # Public   — green
        1: (50,  100, 220),  # Private  — blue
        2: (0,   190, 200),  # Govt     — cyan
        3: (50,  200, 150),  # Tourist  — teal
    }
    vt_color = VT_COLORS.get(vt["id"], (180, 180, 180))
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), vt_color, 3)

    def txt(s, y, color=(210, 210, 210), scale=0.52, bold=False):
        cv2.putText(canvas, str(s), (w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    2 if bold else 1, cv2.LINE_AA)

    y = 30
    txt("PLATE ANALYSIS", y, (0, 210, 255), 0.7, True);  y += 32
    txt(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), y, (110, 110, 110)); y += 30

    txt("Vehicle Type", y, (150, 150, 150), 0.48); y += 20
    txt(vt["name"], y, vt_color, 0.65, True);      y += 28

    txt("Plate Background", y, (150, 150, 150), 0.48); y += 20
    bg = result["bg_color"]
    txt(f'{bg["color"].upper()}  [ID {bg["class_id"]}]', y, (255, 200, 50)); y += 20
    txt(f'Confidence: {bg["confidence"]:.1%}', y); y += 28

    txt("Font Color", y, (150, 150, 150), 0.48); y += 20
    fc = result["font_color"]
    txt(f'{fc["color"].upper()}  [ID {fc["class_id"]}]', y, (255, 200, 50)); y += 20
    txt(f'Confidence: {fc["confidence"]:.1%}', y); y += 28

    txt("Blur Quality", y, (150, 150, 150), 0.48); y += 20
    bl = result["blur"]
    bc = (60, 200, 60) if bl["status"] == "clear" else (50, 50, 220)
    txt(f'{bl["status"].upper()}  [ID {bl["class_id"]}]', y, bc); y += 20
    txt(f'Laplacian: {bl["laplacian_score"]}  Conf: {bl["confidence"]:.1%}', y); y += 28

    txt("OCR", y, (150, 150, 150), 0.48); y += 20
    ocr = result["ocr"]
    plate_txt = ocr["text"] or "N/A"
    # Truncate long text to fit panel
    if len(plate_txt) > 22:
        plate_txt = plate_txt[:22] + "..."
    txt(plate_txt, y, (60, 220, 80), 0.6, True); y += 20
    txt(f'Confidence: {ocr["confidence"]:.1%}', y); y += 28

    cnn = result["cnn_models_used"]
    txt("CNN Models", y, (150, 150, 150), 0.48); y += 20
    txt(", ".join(cnn) if cnn else "rule-based fallback",
        y, (60, 200, 60) if cnn else (160, 110, 50))

    return canvas


def run(image_path=IMAGE_PATH):
    t0 = time.time()
    print(f"\n{'='*55}")
    print(f"  PLATE ANALYSIS: {Path(image_path).name}")
    print(f"{'='*55}")

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ERROR: Cannot read {image_path}"); sys.exit(1)
    print(f"  Size: {img.shape[1]}x{img.shape[0]} px\n")

    print("[1] Blur Quality")
    blur = detect_blur(img)
    print(f"    {blur['status'].upper()}  score={blur['laplacian_score']}  conf={blur['confidence']:.1%}")

    print("[2] Plate Background Color")
    bg = detect_bg_color(img)
    print(f"    {bg['color'].upper()}  conf={bg['confidence']:.1%}")
    print(f"    scores: {bg['scores']}")

    print("[3] Font Color")
    font = detect_font_color(img, bg_color=bg["color"])
    print(f"    {font['color'].upper()}  conf={font['confidence']:.1%}")
    print(f"    scores: {font['scores']}")

    print("[4] Vehicle Type")
    vtype = determine_vehicle_type(bg["color"], font["color"])
    print(f"    ID={vtype['id']}  {vtype['name']}")

    print("[5] OCR")
    ocr = extract_text(img)
    print(f"    '{ocr['text']}'  conf={ocr['confidence']:.1%}")

    print("[6] CNN Inference")
    models = load_models()
    cnn_used = []

    if "blur" in models:
        r = predict_cnn(models["blur"], img, BLUR_IDS)
        blur.update({"class_id": r["class_id"], "status": r["label"],
                     "confidence": r["confidence"],
                     "all_probabilities": r["all_probabilities"]})
        cnn_used.append("blur")
        print(f"    CNN blur  -> {r['label']} ({r['confidence']:.1%})")

    if "bg" in models:
        r = predict_cnn(models["bg"], img, BG_COLOR_IDS)
        bg.update({"class_id": r["class_id"], "color": r["label"],
                   "confidence": r["confidence"],
                   "all_probabilities": r["all_probabilities"]})
        vtype = determine_vehicle_type(bg["color"], font["color"])
        cnn_used.append("bg_color")
        print(f"    CNN bg    -> {r['label']} ({r['confidence']:.1%})")

    if "font" in models:
        r = predict_cnn(models["font"], img, FONT_IDS)
        font.update({"class_id": r["class_id"], "color": r["label"],
                     "confidence": r["confidence"],
                     "all_probabilities": r["all_probabilities"]})
        vtype = determine_vehicle_type(bg["color"], font["color"])
        cnn_used.append("font_color")
        print(f"    CNN font  -> {r['label']} ({r['confidence']:.1%})")

    ms = round((time.time() - t0) * 1000, 1)

    result = {
        "timestamp": datetime.now().isoformat(),
        "image": str(image_path),
        "processing_ms": ms,
        "vehicle_type": vtype,
        "blur": blur,
        "bg_color": bg,
        "font_color": font,
        "ocr": ocr,
        "cnn_models_used": cnn_used,
        "is_readable": blur["status"] == "clear" and blur["laplacian_score"] > 100,
    }

    print(f"\n{'='*55}\n  RESULT\n{'='*55}")
    for k, v in result.items():
        if k not in ("blur", "bg_color", "font_color", "ocr"):
            print(f"  {k}: {v}")
    print(f"  blur:       {blur['status']} (conf {blur['confidence']:.1%})")
    print(f"  bg_color:   {bg['color']} (conf {bg['confidence']:.1%})")
    print(f"  font_color: {font['color']} (conf {font['confidence']:.1%})")
    print(f"  ocr:        '{ocr['text']}' (conf {ocr['confidence']:.1%})")

    out_path = Path(image_path).parent / "test_result.png"
    cv2.imwrite(str(out_path), draw_result(img, result))
    print(f"\n  Annotated image: {out_path}")
    print(f"  Total time: {ms} ms\n")
    return result


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else IMAGE_PATH
    run(path)
