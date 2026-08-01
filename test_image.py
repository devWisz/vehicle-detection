import cv2
import json
import sys
import time
import uuid
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

BASE_DIR     = Path(__file__).resolve().parent
IMAGE_PATH   = BASE_DIR / "images" / "image.png"
RESULTS_FILE = BASE_DIR / "detection_results.json"

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
    conf = float(1 / (1 + np.exp(-(score - 100) / 40)))
    if status == "blurry":
        conf = 1 - conf
    return {
        "class_id": 0 if status == "clear" else 1,
        "status": status,
        "confidence": round(conf, 4),
        "laplacian_score": round(score, 2),
    }


def detect_bg_color_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = img.shape[0] * img.shape[1]
    masks = {
        "white": cv2.inRange(hsv, (0,  0,  170), (180, 45, 255)),
        "red":   cv2.bitwise_or(
                     cv2.inRange(hsv, (0,  80,  60), (10,  255, 255)),
                     cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))),
        "green": cv2.inRange(hsv, (35, 60,  30), (90,  255, 200)),
    }
    scores = {c: int(m.sum() / 255) / max(total, 1) for c, m in masks.items()}
    best   = max(scores, key=scores.get)
    denom  = sum(scores.values()) or 1
    return {
        "class_id": {"white": 0, "red": 1, "green": 2}[best],
        "color": best,
        "confidence": round(scores[best] / denom, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


def detect_font_color_hsv(img):
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark  = cv2.threshold(gray, 80,  255, cv2.THRESH_BINARY_INV)
    _, light = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    total = img.shape[0] * img.shape[1]
    masks = {
        "black": dark,
        "white": light,
        "red":   cv2.bitwise_or(
                     cv2.inRange(hsv, (0,  80, 60), (10,  255, 255)),
                     cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))),
    }
    scores = {c: int(m.sum() / 255) / max(total, 1) for c, m in masks.items()}
    best   = max(scores, key=scores.get)
    denom  = sum(scores.values()) or 1
    return {
        "class_id": {"black": 0, "white": 1, "red": 2}[best],
        "color": best,
        "confidence": round(scores[best] / denom, 4),
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
                print(f"  ✓ {name} model loaded")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        else:
            print(f"  ⚠ {name} model not found — using rule-based fallback")
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
        "confidence": round(float(np.max(probs)), 4),
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
                 if p > 0.2 and t.strip()]
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
    panel = 360
    canvas = np.full((max(h, 460), w + panel, 3), 25, dtype=np.uint8)
    canvas[:h, :w] = img
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 200, 60), 3)

    vt = result["vehicle_type"]
    VT_COLORS = {0: (0, 200, 60), 1: (0, 80, 200),
                 2: (0, 180, 200), 3: (0, 200, 150)}
    vt_color = VT_COLORS.get(vt["id"], (200, 200, 200))

    def txt(s, y, color=(210, 210, 210), scale=0.52, bold=False):
        cv2.putText(canvas, s, (w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    2 if bold else 1, cv2.LINE_AA)

    y = 28
    txt("PLATE ANALYSIS", y, (0, 210, 255), 0.68, True); y += 30
    txt(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), y, (140, 140, 140)); y += 28

    txt("-- Vehicle Type --", y, (140, 140, 140)); y += 22
    txt(vt["name"], y, vt_color, 0.62, True); y += 26

    txt("-- Plate Background --", y, (140, 140, 140)); y += 22
    bg = result["bg_color"]
    txt(f"Color: {bg['color'].upper()}  [ID:{bg['class_id']}]", y, (255, 210, 60)); y += 20
    txt(f"Conf : {bg['confidence']:.1%}", y); y += 26

    txt("-- Font Color --", y, (140, 140, 140)); y += 22
    fc = result["font_color"]
    txt(f"Color: {fc['color'].upper()}  [ID:{fc['class_id']}]", y, (255, 210, 60)); y += 20
    txt(f"Conf : {fc['confidence']:.1%}", y); y += 26

    txt("-- Blur Quality --", y, (140, 140, 140)); y += 22
    bl = result["blur"]
    bc = (0, 200, 60) if bl["status"] == "clear" else (0, 50, 220)
    txt(f"Status: {bl['status'].upper()}  [ID:{bl['class_id']}]", y, bc); y += 20
    txt(f"Lap Score: {bl['laplacian_score']:.1f}", y); y += 20
    txt(f"Conf     : {bl['confidence']:.1%}", y); y += 26

    txt("-- OCR --", y, (140, 140, 140)); y += 22
    ocr = result["ocr"]
    txt(f"{ocr['text'] or 'N/A'}", y, (0, 220, 80), 0.58, True); y += 20
    txt(f"Conf: {ocr['confidence']:.1%}", y); y += 26

    cnn = result["cnn_models_used"]
    txt("-- CNN Models --", y, (140, 140, 140)); y += 22
    txt(", ".join(cnn) if cnn else "rule-based only",
        y, (0, 200, 60) if cnn else (180, 120, 60))

    return canvas


def load_results_db():
    if RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0:
        with open(RESULTS_FILE) as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def save_results_db(db):
    with open(RESULTS_FILE, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def run(image_path=IMAGE_PATH):
    t0 = time.time()
    print(f"\n{'='*55}")
    print(f"  PLATE ANALYSIS: {Path(image_path).name}")
    print(f"{'='*55}")

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ERROR: Cannot read {image_path}")
        sys.exit(1)
    print(f"  Size: {img.shape[1]}x{img.shape[0]} px\n")

    print("[1] Blur Detection")
    blur = detect_blur(img)
    print(f"    {blur['status'].upper()}  score={blur['laplacian_score']}  "
          f"conf={blur['confidence']:.1%}")

    print("[2] Background Color (HSV)")
    bg = detect_bg_color_hsv(img)
    print(f"    {bg['color'].upper()}  conf={bg['confidence']:.1%}")

    print("[3] Font Color (HSV)")
    font = detect_font_color_hsv(img)
    print(f"    {font['color'].upper()}  conf={font['confidence']:.1%}")

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
                   "all_probabilities": r.get("all_probabilities")})
        vtype = determine_vehicle_type(bg["color"], font["color"])
        cnn_used.append("bg_color")
        print(f"    CNN bg    -> {r['label']} ({r['confidence']:.1%})")

    if "font" in models:
        r = predict_cnn(models["font"], img, FONT_IDS)
        font.update({"class_id": r["class_id"], "color": r["label"],
                     "confidence": r["confidence"],
                     "all_probabilities": r.get("all_probabilities")})
        vtype = determine_vehicle_type(bg["color"], font["color"])
        cnn_used.append("font_color")
        print(f"    CNN font  -> {r['label']} ({r['confidence']:.1%})")

    ms = round((time.time() - t0) * 1000, 1)

    result = {
        "detection_id": str(uuid.uuid4()),
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

    db = load_results_db()
    db.append(result)
    save_results_db(db)
    print(f"\n  Saved to {RESULTS_FILE}  (total records: {len(db)})")

    print(f"\n{'='*55}\n  RESULT\n{'='*55}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = Path(image_path).parent / "test_result.png"
    cv2.imwrite(str(out_path), draw_result(img, result))
    print(f"\n  Annotated: {out_path}")
    print(f"  Total time: {ms} ms\n")
    return result


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else IMAGE_PATH
    run(path)
