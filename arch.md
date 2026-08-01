# Architecture — Vehicle Plate Detection System

## Overview

This system reads a vehicle licence plate image and outputs:
- **Vehicle type** (Public / Private / Government / Tourist)
- **Plate number** (via OCR)
- **Image quality** (clear / blurry)
- **Confidence scores** for every classification

It works in two modes:
1. **Rule-based only** — pure HSV colour analysis, no GPU, no trained models needed.
2. **Full CNN + YOLO** — trained neural networks for higher accuracy, YOLO for plate localisation.

---

## Vehicle Type Rules

The vehicle type is determined by the **combination of plate background colour and font colour**:

| Background | Font  | Vehicle Type       | Class ID |
|------------|-------|--------------------|----------|
| White      | Black | Public Vehicle     | 0        |
| Red        | White | Private Vehicle    | 1        |
| White      | Red   | Government Vehicle | 2        |
| Green      | White | Tourist Vehicle    | 3        |

These rules are defined in `VEHICLE_RULES` in both `test_image.py` and `main_enhanced.py`.

---

## System Components

```
images/image.png
       │
       ▼
┌──────────────────┐    (optional)   ┌─────────────────┐
│   YOLO Detector  │ ──────────────▶ │  Plate Crop(s)  │
│  license_plate.pt│                 └────────┬────────┘
└──────────────────┘                          │
       │ (if not available)                   │
       └──── full image used as crop ─────────┘
                                              │
              ┌───────────────────────────────┤
              │                               │
              ▼                               ▼
     ┌────────────────┐             ┌──────────────────┐
     │  Blur Detector │             │  BG Colour Model │
     │ EfficientNetB2 │             │  EfficientNetB0  │
     │ (or Laplacian) │             │  (or HSV masks)  │
     └───────┬────────┘             └────────┬─────────┘
             │                               │
             ▼                               ▼
       clear / blurry                white / red / green
                                              │
                                     ┌────────┴──────────┐
                                     │  Font Colour Model│
                                     │  EfficientNetB1   │
                                     │  (or HSV masks)   │
                                     └────────┬──────────┘
                                              │
                                     black / white / red
                                              │
                                     ┌────────┴──────────┐
                                     │   VEHICLE_RULES   │
                                     │  (lookup table)   │
                                     └────────┬──────────┘
                                              │
                                     Public / Private /
                                     Government / Tourist
                                              │
                                     ┌────────┴──────────┐
                                     │    EasyOCR        │
                                     │  (plate number)   │
                                     └───────────────────┘
```

---

## File-by-File Explanation

### `data_preparation.py` — Dataset Generator

Generates synthetic licence plate images for training three separate classifiers.

**What it creates:**

| Dataset folder  | Classes              | Purpose                    |
|-----------------|----------------------|----------------------------|
| `blur_quality/` | `clear`, `blurry`    | Train blur detector        |
| `bg_color/`     | `white`, `red`, `green` | Train plate BG classifier |
| `font_color/`   | `black`, `white`, `red` | Train font classifier     |

**How it generates plates:**
- Draws a filled rectangle in the target colour (BG)
- Renders characters in the target font colour using OpenCV `putText`
- For `blurry`: applies Gaussian blur (`kernel_size=15`)
- Splits automatically into `train/` (70%), `val/` (15%), `test/` (15%)

**Output structure:**
```
dataset/
  blur_quality/  train/blurry/  train/clear/  val/  test/
  bg_color/      train/white/   train/red/    train/green/  val/  test/
  font_color/    train/black/   train/white/  train/red/    val/  test/
```

---

### `train_models.py` — CNN Trainer

Trains three EfficientNet models. Each one is independent and does a single classification task.

#### Model Architecture (shared by all three)

```
Input (224×224×3, normalised 0–1)
  │
  ├── RandomFlip / RandomRotation / RandomZoom / RandomBrightness / RandomContrast
  │       (data augmentation, only active during training)
  │
  ├── Rescaling(255.0)
  │       (restores [0,255] range that EfficientNet expects)
  │
  ├── EfficientNetBx (pretrained ImageNet backbone, frozen first)
  │
  ├── SqueezeExcitation(ratio=16)
  │       (channel attention: learns which feature maps matter most)
  │
  ├── GlobalAveragePooling2D
  │
  ├── Dense(512, relu) → BatchNorm → Dropout(p)
  ├── Dense(256, relu) → BatchNorm → Dropout(p*0.6)
  ├── Dense(128, relu)
  │
  └── Dense(num_classes, softmax)
```

#### SqueezeExcitation Attention Block

```python
# Squeeze: compress spatial info into channel vector
x_squeeze = GlobalAveragePooling2D()(x)
# Excitation: learn channel importance weights
x_excite  = Dense(C//ratio, relu)(x_squeeze)
x_excite  = Dense(C, sigmoid)(x_excite)
# Scale: multiply original features by learned weights
output    = x * x_excite
```

This helps the model focus on the most informative colour/texture channels.

#### Three Models

| Model              | Backbone       | Classes            | Dropout |
|--------------------|----------------|--------------------|---------|
| Blur Classifier    | EfficientNetB2 | clear, blurry      | 0.40    |
| BG Color Classifier| EfficientNetB0 | white, red, green  | 0.35    |
| Font Color Classifier| EfficientNetB1| black, white, red  | 0.40    |

#### Training Strategy (Two-Phase Fine-tuning)

**Phase 1 — Feature Extraction** (backbone frozen):
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Loss: Focal Loss (γ=2.0, α=0.25) for blur; CategoricalCrossentropy for colour
- Epochs: 40, with early stopping (patience=10)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)

**Phase 2 — Fine-tuning** (last 30 layers of backbone unfrozen):
- Optimizer: AdamW (lr=1e-4)
- Epochs: 30, with early stopping (patience=8)

**Why Focal Loss for blur?**
Blurry plates are less common than clear ones → class imbalance. Focal Loss down-weights easy examples and focuses training on hard cases (moderately blurry images).

#### Saved Outputs (per model)

| File                             | Format  | Use                    |
|----------------------------------|---------|------------------------|
| `blur_classifier_final.keras`    | Keras   | Python inference       |
| `blur_classifier_final.tflite`   | TFLite  | Mobile/Edge deployment |
| `blur_classifier_metadata.json`  | JSON    | Class map, threshold   |
| `models/blur_confusion.png`      | Image   | Training evaluation    |

---

### `test_image.py` — Single Image Tester

The standalone test script. Does not need YOLO. Processes one image at a time.

#### Processing Pipeline

```
Step 1  detect_blur(img)
          Laplacian variance → if var > 100: clear, else: blurry
          Confidence = sigmoid((score − 100) / 80), capped at 0.99

Step 2  detect_bg_color(img)
          HSV colour masks for white / red / green
          White: (H=any, S<50, V>160)
          Red:   (H=0–10 or H=165–180, S>100, V>70) [hue wrap-around]
          Green: (H=38–88, S>80, V=40–210)
          White pixels that overlap red are excluded (border artefact fix)

Step 3  detect_font_color(img, bg_color)
          Crops central 70% of height to avoid corner emblems (flags, seals)
          Excludes background colour pixels before searching for font
          Red font: H=red, S>100, V>70 — needs ≥3% of text-zone pixels
          White font: S<40, V>190
          Black font: V<80, S<80
          Priority: red → white → black

Step 4  determine_vehicle_type(bg_color, font_color)
          Direct lookup in VEHICLE_RULES dictionary

Step 5  EasyOCR
          Converts to grayscale → Otsu threshold → OCR
          Languages: Nepali (ne) + English (en)
          Words with confidence < 0.25 are discarded

Step 6  CNN Models (if trained models exist)
          Runs Test-Time Augmentation (TTA × 4 variants):
            original, horizontal flip, +15% brightness, -15% brightness
          Averages the 4 softmax outputs → final prediction
          CNN result overrides HSV result
```

#### Why Centre-Crop for Font Detection?

Nepali plates often have the Nepal flag in the top-left corner. The flag is red. Without cropping, those red flag pixels can exceed the threshold and cause the font to be mis-classified as "red", making a Public plate (white+black) look like a Government plate (white+red).

By analysing only rows 15%–85% of the image height, we focus on the actual character area and exclude corner decorations.

---

### `main_enhanced.py` — Full Pipeline (Production)

The production script using YOLO + CNN + OCR together.

#### `Pipeline` Class

```
__init__()
  ├── _load_yolo()       load license_plate.pt
  ├── _load_cnns()       load 3 .keras models
  └── _load_ocr()        initialise EasyOCR

process(image_path)
  ├── YOLO detection → list of bounding boxes
  │     (or full image if YOLO unavailable)
  └── For each crop → _process_crop()

_process_crop(crop, yolo_conf)
  ├── _classify_blur()   → CNN or Laplacian
  ├── _classify_bg()     → CNN or HSV
  ├── _classify_font()   → CNN or HSV (uses bg_color to exclude bg pixels)
  ├── _determine_vehicle_type()
  ├── _extract_text()    → EasyOCR
  └── combined_confidence = blur×0.20 + bg×0.30 + font×0.20 + yolo×0.30
```

#### Combined Confidence Score

The final `combined_confidence` weighs each sub-classifier:

| Component     | Weight | Reason                                     |
|---------------|--------|--------------------------------------------|
| Blur          | 20%    | Tells us if the plate is readable          |
| BG Color      | 30%    | Primary classification signal              |
| Font Color    | 20%    | Secondary classification signal            |
| YOLO          | 30%    | Detection confidence = plate is real plate |

A detection is marked `is_reliable = True` if `combined_confidence > 0.70`.

---

## HSV Colour Ranges (Reference)

OpenCV HSV range: H=0–180, S=0–255, V=0–255

| Colour | Hue Range     | Saturation | Value  | Notes                          |
|--------|---------------|------------|--------|--------------------------------|
| Red    | 0–10, 165–180 | > 100      | > 70   | Hue wraps at 0/180 in OpenCV   |
| White  | 0–180 (any)   | < 50       | > 160  | Low sat, high brightness       |
| Black  | 0–180 (any)   | < 80       | < 80   | Low sat, low brightness        |
| Green  | 38–88         | > 80       | 40–210 | Mid-value to avoid dark green  |

---

## Inference Accuracy Notes

### Rule-Based (HSV only, no training needed)
- Works well on clean, well-lit images
- Can fail on: shadows, yellow-tinted plates, extreme glare

### CNN + Rule-Based Hybrid
- CNN gives probability scores across all classes
- TTA (Test-Time Augmentation) averages 4 variants to reduce prediction noise
- HSV is used as fallback when models are not trained yet

### Known Limitations
- Plates with dirt, wear, or partial occlusion may be misclassified
- Yellow plates (older Nepali system) are not a current target class
- OCR accuracy depends on plate condition and font size

---

## Step-by-Step Run Guide

### Prerequisites
```bash
cd vehicle-detection
python3 -m venv venv
source venv/bin/activate
pip install tensorflow opencv-python pillow scikit-learn \
            matplotlib tqdm easyocr ultralytics
```

### Step 1 — Generate Dataset (required for training)
```bash
python data_preparation.py
```
Creates ~5,600 synthetic plate images in `dataset/`.

### Step 2 — Quick Test (no training required)
```bash
python test_image.py
# Or test a specific image:
python test_image.py path/to/plate.jpg
```
Uses rule-based HSV detection. Output includes annotated image at `images/test_result.png`.

### Step 3 — Train CNN Models
```bash
python train_models.py
```
Takes 2–4 hours on CPU. Saves `.keras` and `.tflite` models to the project root.

### Step 4 — Test With CNN Models
```bash
python test_image.py
```
If `.keras` files exist, they load automatically and override HSV.

### Step 5 — Full Pipeline (with YOLO)
Place your YOLOv8 licence plate weights at `license_plate.pt`, then:
```bash
python main_enhanced.py
```

---

## Directory Structure

```
vehicle-detection/
├── images/
│   ├── image.png          # Test image
│   └── test_result.png    # Annotated output (auto-generated)
├── dataset/               # Generated training data (gitignored)
│   ├── blur_quality/
│   ├── bg_color/
│   └── font_color/
├── models/                # Training plots (gitignored)
├── data_preparation.py    # Dataset generator
├── train_models.py        # CNN trainer (3 models)
├── test_image.py          # Single-image tester
├── main_enhanced.py       # Full YOLO+CNN+OCR pipeline
├── README.md              # Quick start guide
├── arch.md                # This file — technical architecture
└── .gitignore
```
