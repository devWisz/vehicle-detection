import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent

BLUR_IDS     = {0: "clear",  1: "blurry"}
BG_COLOR_IDS = {0: "white",  1: "red",   2: "green"}
FONT_IDS     = {0: "black",  1: "white", 2: "red"}

VEHICLE_RULES = {
    ("white", "black"): {"id": 0, "name": "Public Vehicle"},
    ("red",   "white"): {"id": 1, "name": "Private Vehicle"},
    ("white", "red"):   {"id": 2, "name": "Government Vehicle"},
    ("green", "white"): {"id": 3, "name": "Tourist Vehicle"},
}


@dataclass
class Detection:
    timestamp: str
    image: str
    bounding_box: Dict
    plate_number: str
    ocr_confidence: float
    vehicle_type: Dict
    blur: Dict
    bg_color: Dict
    font_color: Dict
    yolo_confidence: float
    combined_confidence: float
    is_reliable: bool
    is_readable: bool
    processing_ms: float
    cnn_models_used: List[str] = field(default_factory=list)


def _preprocess(img, size=(224, 224)):
    rgb = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb.astype("float32") / 255.0, axis=0)


def _predict_tta(model, img, id_map):
    variants = [
        img,
        cv2.flip(img, 1),
        cv2.convertScaleAbs(img, alpha=1.15, beta=10),
        cv2.convertScaleAbs(img, alpha=0.85, beta=-10),
    ]
    probs = np.mean([model.predict(_preprocess(v), verbose=0)[0]
                     for v in variants], axis=0)
    cls = int(np.argmax(probs))
    return {"class_id": cls, "label": id_map[cls],
            "confidence": min(round(float(np.max(probs)), 4), 0.99)}


def _hsv_bg_color(img):
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = max(img.shape[0] * img.shape[1], 1)

    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0,   100, 70), (10,  255, 255)),
        cv2.inRange(hsv, (165, 100, 70), (180, 255, 255)),
    )
    green = cv2.inRange(hsv, (38, 80, 40), (88, 255, 210))
    white = cv2.bitwise_and(
        cv2.inRange(hsv, (0, 0, 160), (180, 50, 255)),
        cv2.bitwise_not(red),
    )
    scores = {
        "red":   int(red.sum()   / 255) / total,
        "green": int(green.sum() / 255) / total,
        "white": int(white.sum() / 255) / total,
    }
    best  = max(scores, key=scores.get)
    denom = sum(scores.values()) or 1
    return {"class_id": {"white": 0, "red": 1, "green": 2}[best],
            "color": best,
            "confidence": min(round(scores[best] / denom, 4), 0.99)}


def _hsv_font_color(img, bg_color="white"):
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
    if scores["red"] >= 0.03:
        best = "red"
    elif scores["white"] > scores["black"]:
        best = "white"
    else:
        best = "black"

    denom = sum(scores.values()) or 1
    return {"class_id": {"black": 0, "white": 1, "red": 2}[best],
            "color": best,
            "confidence": min(round(scores[best] / denom, 4), 0.99)}


def _detect_blur(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    status = "clear" if score > 100 else "blurry"
    raw = float(1 / (1 + np.exp(-(score - 100) / 80)))
    conf = min(raw, 0.99) if status == "clear" else min(1 - raw + 0.01, 0.99)
    return {"class_id": 0 if status == "clear" else 1,
            "status": status,
            "confidence": round(conf, 4),
            "laplacian_score": round(score, 2)}


def _determine_vehicle_type(bg_color, font_color):
    return VEHICLE_RULES.get((bg_color, font_color), {"id": -1, "name": "Unknown"})


class Pipeline:
    YOLO_MODEL  = BASE_DIR / "license_plate.pt"
    BLUR_MODEL  = BASE_DIR / "blur_classifier_final.keras"
    BG_MODEL    = BASE_DIR / "bg_color_classifier_final.keras"
    FONT_MODEL  = BASE_DIR / "font_color_classifier_final.keras"
    YOLO_CONF   = 0.3
    MIN_CONF    = 0.70

    def __init__(self):
        print("Initializing pipeline...")
        self.yolo = self.blur_model = self.bg_model = self.font_model = self.ocr = None
        self._load_yolo()
        self._load_cnns()
        self._load_ocr()
        print("Ready.\n")

    def _load_yolo(self):
        if not YOLO_AVAILABLE:
            print("  - YOLO not installed"); return
        if not self.YOLO_MODEL.exists():
            print(f"  - YOLO model not found: {self.YOLO_MODEL.name}"); return
        try:
            self.yolo = YOLO(str(self.YOLO_MODEL))
            print("  + YOLO loaded")
        except Exception as e:
            print(f"  ! YOLO: {e}")

    def _load_cnns(self):
        if not TF_AVAILABLE:
            print("  - TensorFlow not available"); return
        for attr, path, label in [
            ("blur_model", self.BLUR_MODEL, "blur"),
            ("bg_model",   self.BG_MODEL,   "bg_color"),
            ("font_model", self.FONT_MODEL, "font_color"),
        ]:
            if path.exists():
                try:
                    setattr(self, attr,
                            keras.models.load_model(str(path), compile=False))
                    print(f"  + {label} model loaded")
                except Exception as e:
                    print(f"  ! {label}: {e}")
            else:
                print(f"  - {label}: not found, using rule-based fallback")

    def _load_ocr(self):
        if not OCR_AVAILABLE:
            print("  - easyocr not installed"); return
        try:
            self.ocr = easyocr.Reader(["ne", "en"], gpu=False, verbose=False)
            print("  + OCR ready")
        except Exception as e:
            print(f"  ! OCR: {e}")

    def _classify_blur(self, crop):
        if self.blur_model:
            r = _predict_tta(self.blur_model, crop, BLUR_IDS)
            base = _detect_blur(crop)
            return {**base, "class_id": r["class_id"],
                    "status": r["label"], "confidence": r["confidence"]}
        return _detect_blur(crop)

    def _classify_bg(self, crop):
        if self.bg_model:
            return _predict_tta(self.bg_model, crop, BG_COLOR_IDS)
        return _hsv_bg_color(crop)

    def _classify_font(self, crop, bg_color="white"):
        if self.font_model:
            return _predict_tta(self.font_model, crop, FONT_IDS)
        return _hsv_font_color(crop, bg_color)

    def _extract_text(self, crop):
        if not self.ocr:
            return {"text": "", "confidence": 0.0}
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            results = self.ocr.readtext(thresh)
            valid = [(t.strip(), float(p)) for _, t, p in results
                     if p > 0.25 and t.strip()]
            if not valid:
                return {"text": "", "confidence": 0.0}
            return {"text": " ".join(t for t, _ in valid),
                    "confidence": round(float(np.mean([p for _, p in valid])), 4)}
        except Exception:
            return {"text": "", "confidence": 0.0}

    def _process_crop(self, crop, yolo_conf, img_path):
        t0 = time.time()
        blur  = self._classify_blur(crop)
        bg    = self._classify_bg(crop)
        font  = self._classify_font(crop, bg_color=bg.get("color", "white"))
        vtype = _determine_vehicle_type(bg.get("color", ""),
                                        font.get("color", ""))
        ocr   = self._extract_text(crop)

        cnn_used = []
        if self.blur_model: cnn_used.append("blur")
        if self.bg_model:   cnn_used.append("bg_color")
        if self.font_model: cnn_used.append("font_color")

        combined = (
            blur.get("confidence", 0) * 0.20 +
            bg.get("confidence",   0) * 0.30 +
            font.get("confidence", 0) * 0.20 +
            yolo_conf                 * 0.30
        )

        return Detection(
            timestamp=datetime.now().isoformat(),
            image=str(img_path),
            bounding_box={},
            plate_number=ocr["text"],
            ocr_confidence=ocr["confidence"],
            vehicle_type=vtype,
            blur=blur,
            bg_color=bg,
            font_color=font,
            yolo_confidence=yolo_conf,
            combined_confidence=round(combined, 4),
            is_reliable=combined > self.MIN_CONF,
            is_readable=(blur.get("status") == "clear"
                         and blur.get("laplacian_score", 0) > 100),
            processing_ms=round((time.time() - t0) * 1000, 1),
            cnn_models_used=cnn_used,
        )

    def process(self, image_path: str) -> Optional[List[Detection]]:
        t0   = time.time()
        path = Path(image_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.exists():
            print(f"Image not found: {path}"); return None

        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Cannot read: {path}"); return None

        print(f"Processing: {path.name}  ({frame.shape[1]}x{frame.shape[0]})")

        if self.yolo is None:
            print("  YOLO unavailable — treating full image as plate")
            det = self._process_crop(frame, yolo_conf=0.5, img_path=path)
            det.bounding_box = {"x1": 0, "y1": 0,
                                 "x2": frame.shape[1], "y2": frame.shape[0]}
            detections = [det]
        else:
            t_yolo   = time.time()
            results  = self.yolo(frame, conf=self.YOLO_CONF)[0]
            print(f"  YOLO: {len(results.boxes)} plate(s)  "
                  f"({(time.time()-t_yolo)*1000:.0f}ms)")
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                det = self._process_crop(crop, float(box.conf[0]), path)
                det.bounding_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                detections.append(det)

        if not detections:
            print("  No plates detected"); return None

        print(f"  Done: {(time.time()-t0)*1000:.0f}ms\n")
        return detections

    def print_results(self, detections: List[Detection]):
        for i, d in enumerate(detections, 1):
            print(f"  [{i}] Plate: {d.plate_number or 'N/A'}")
            print(f"      Type  : {d.vehicle_type['name']} (ID {d.vehicle_type['id']})")
            print(f"      BG    : {d.bg_color.get('color')} | "
                  f"Font: {d.font_color.get('color')}")
            print(f"      Blur  : {d.blur.get('status')} "
                  f"(lap={d.blur.get('laplacian_score')})")
            print(f"      Conf  : {d.combined_confidence:.1%} "
                  f"{'[RELIABLE]' if d.is_reliable else '[LOW]'}")
            print()


if __name__ == "__main__":
    pipeline   = Pipeline()
    detections = pipeline.process("images/image.png")
    if detections:
        pipeline.print_results(detections)
    else:
        print("No detections.")
