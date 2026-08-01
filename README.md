# Nepali License Plate Detection System

Detects and classifies Nepali vehicle license plates using CNN models + OCR.

## Vehicle Type Classification

| ID | BG Color | Font Color | Vehicle Type       |
|----|----------|------------|--------------------|
| 0  | White    | Black      | Public Vehicle     |
| 1  | Red      | White      | Private Vehicle    |
| 2  | White    | Red        | Government Vehicle |
| 3  | Green    | White      | Tourist Vehicle    |

## Models

| Model | Backbone | Classes | Task |
|---|---|---|---|
| `blur_classifier` | EfficientNetB2 | clear / blurry | Image quality |
| `bg_color_classifier` | EfficientNetB0 | white / red / green | Plate background |
| `font_color_classifier` | EfficientNetB1 | black / white / red | Font color |

Vehicle type is determined by combining bg_color + font_color outputs via the rule table above.

---

## Step-by-Step Guide

### Step 0 — Set up the environment

```bash
cd /path/to/vehicle-detection
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
pip install tensorflow opencv-python pillow scikit-learn matplotlib tqdm easyocr ultralytics
```

---

### Step 1 — Generate the synthetic dataset

```bash
python data_preparation.py
```

**What it does:**
- Generates 560 images per class across 3 classifiers
- Creates `dataset/blur_quality/`, `dataset/bg_color/`, `dataset/font_color/`
- Each has `train/` (392), `val/` (84), and `test/` (84) splits

**Expected output:**
```
Generating dataset → dataset/
  560 images/class  (train=392 val=84 test=84)

[1/3] Blur Quality
  train done
  val done
  test done
...
Dataset Summary:
  blur_quality/
    train :  1184 images  (blurry, clear)
    val   :   168 images  (blurry, clear)
    test  :   168 images  (blurry, clear)
  ...
```

> **Optional:** Add real Nepali plate images to the appropriate class folders before training for better accuracy.

---

### Step 2 — Test on the sample image (before training)

```bash
python test_image.py
```

This works **without any trained models** using rule-based HSV detection + EasyOCR.

You can also test any image:
```bash
python test_image.py path/to/your/plate.png
```

**Expected output:**
```
[1] Blur Detection   → CLEAR  score=637.2  conf=100.0%
[2] Background Color → WHITE  conf=72.3%
[3] Font Color       → BLACK  conf=77.9%
[4] Vehicle Type     → ID=0  Public Vehicle
[5] OCR              → 'BA 1 AB 1350'  conf=51.0%
[6] CNN Inference    → (rule-based, no models yet)
```

Results are saved to `detection_results.json` (appended each run with unique IDs).
Annotated image is saved to `images/test_result.png`.

---

### Step 3 — Train the models

```bash
python train_models.py
```

**What it does:**
- Phase 1: Trains classifier head with frozen backbone (30 epochs)
- Phase 2: Fine-tunes top backbone layers (15 epochs)
- Saves `.keras` models, TFLite files, metadata JSON

**Expected files after training:**
```
blur_classifier_final.keras
bg_color_classifier_final.keras
font_color_classifier_final.keras
blur_classifier_metadata.json
bg_color_classifier_metadata.json
font_color_classifier_metadata.json
tflite/blur_classifier.tflite
tflite/bg_color_classifier.tflite
tflite/font_color_classifier.tflite
```

> Training on CPU takes ~2–4 hours. On GPU it takes ~30–60 minutes.

---

### Step 4 — Test again with trained CNN models

```bash
python test_image.py
```

Now CNN models will be used with Test-Time Augmentation (4 variants averaged) for higher accuracy.

---

### Step 5 — Run the full pipeline (with YOLO plate detection)

```bash
python main_enhanced.py
```

Requires `license_plate.pt` (YOLO model). Without it, the script falls back to treating the full image as the plate.

To use a custom image:
```python
pipeline = Pipeline()
detections = pipeline.process("path/to/vehicle_image.jpg")
pipeline.save(detections)
```

---

## JSON Output Format

Every detection run appends to `detection_results.json`:

```json
[
  {
    "detection_id": "3f2a1b4c-...",
    "timestamp": "2026-08-01T10:45:00",
    "image": "/path/to/image.png",
    "processing_ms": 2172.4,
    "vehicle_type": { "id": 0, "name": "Public Vehicle" },
    "blur": {
      "class_id": 0, "status": "clear",
      "confidence": 1.0, "laplacian_score": 637.22
    },
    "bg_color": { "class_id": 0, "color": "white", "confidence": 0.85 },
    "font_color": { "class_id": 0, "color": "black", "confidence": 0.91 },
    "ocr": { "text": "BA 1 AB 1350", "confidence": 0.51 },
    "cnn_models_used": ["blur", "bg_color", "font_color"],
    "is_readable": true
  }
]
```

---

## File Overview

| File | Purpose |
|---|---|
| `data_preparation.py` | Generate synthetic dataset |
| `train_models.py` | Train 3 CNN classifiers |
| `test_image.py` | Test a single plate image |
| `main_enhanced.py` | Full YOLO + CNN + OCR pipeline |
| `dataset/` | Generated training data (gitignored) |
| `detection_results.json` | Cumulative JSON results database |
| `images/image.png` | Sample Nepali plate image |
| `images/test_result.png` | Annotated detection output |
