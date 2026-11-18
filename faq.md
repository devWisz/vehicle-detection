# Frequently Asked Questions & Technical Architecture (FAQ.md)

This document provides a comprehensive, detailed overview of the technical architecture, operational workflows, model training strategies, and frequently asked questions for the **Vehicle Plate Detection and Analysis System**.

---

## 1. System Architecture & Workflow

The system takes a raw vehicle or license plate image and outputs the **blur status**, **background color**, **font color**, **vehicle type**, **detected script**, and the **recognized plate number**, saving the results in both visual (PNG) and structured (JSON) formats.

The end-to-end processing pipeline runs as follows:

```
                  ┌──────────────────────────────┐
                  │      Input Image (.png/.jpg) │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │     Plate Detection Stage    │
                  │   - YOLO Bounding Box OR     │
                  │   - CV2 Contour Localization │
                  └──────────────┬───────────────┘
                                 │ (Crop Plate Area)
                                 ▼
                  ┌──────────────────────────────┐
                  │    Image Quality Assessment  │
                  │  - Laplacian score (clear)   │
                  │  - CNN Quality Classifier    │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌─────────────────────────────────┐             ┌─────────────────────────────────┐
│     Color Classification        │             │        OCR Text Extraction      │
│ - Background & Font Classifier  │             │ - Multi-pass Image Prep         │
│ - Rule-based HSV Color Masking  │             │ - EasyOCR (ne, hi, en) Reader   │
└────────────────┬────────────────┘             └────────────────┬────────────────┘
                 │ (white / red / green / black)                 │ (raw text)
                 ▼                                               ▼
┌─────────────────────────────────┐             ┌─────────────────────────────────┐
│      Vehicle Type Lookup        │             │   Post-Processing Corrections   │
│ - Match combination in rules:   │             │ - Devanagari character mapping  │
│   e.g., White + Black = Public   │             │ - Latin corrections (HEP->NEP)  │
└────────────────┬────────────────┘             └────────────────┬────────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────┐
                  │    Output Generation Stage   │
                  │  - Annotated Image Result   │
                  │  - Pretty-printed JSON File  │
                  └──────────────────────────────┘
```

---

## 2. Component Explanations

### A. License Plate Localization
1. **Primary Route (YOLOv8)**:
   - Uses a pre-trained **YOLOv8 Object Detector** (`license_plate.pt`) to locate plate bounding boxes.
   - If multiple plates are present, the pipeline processes each crop independently.
2. **Fallback Route (OpenCV Contours)**:
   - If YOLO is unavailable (e.g. model weights are missing), a shape-based **CV2 Plate Detector** (`locate_plate`) is executed.
   - It converts the image to grayscale, thresholds for bright rectangular regions, applies morphological closing to merge letters, and filters contours using size and aspect ratio bounds.
   - **Double-line Support**: The aspect ratio limits are set between `1.2 <= aspect <= 7.5` to ensure taller double-line motorcycle plates (like `image.png`) are successfully auto-detected and cropped.

### B. Color Classification (Background & Font)
- The vehicle classification is derived directly from the plate's background and font colors:
  - **White BG + Black Font** $\rightarrow$ Public Vehicle (ID 0)
  - **Red BG + White Font** $\rightarrow$ Private Vehicle (ID 1)
  - **White BG + Red Font** $\rightarrow$ Government Vehicle (ID 2)
  - **Green BG + White Font** $\rightarrow$ Tourist Vehicle (ID 3)
- Classification uses a **Hybrid Inference Engine**:
  1. It computes **Rule-based HSV Color Masks** (using predefined hue, saturation, and value bounds) to score the colors.
  2. It executes **CNN classifiers** on the cropped plate.
  3. **Confidence Routing**: If the CNN model yields low confidence (e.g., $< 70\%$), the pipeline discards the CNN predictions and defaults to the highly reliable rule-based HSV output. This prevents unconfident or collapsed neural networks from overriding correct classifications.

### C. Text Recognition (OCR) & Post-Processing
- **Pre-processing pipeline**: Grayscale images are processed into 4 different binary formats (including CLAHE-equalized, Gaussian-sharpened, adaptive thresholded, and inverted versions) to maximize legibility.
- **EasyOCR Reader**: Reads the variants and selects the text output with the highest cumulative confidence.
- **Text Corrections**:
  - *Devanagari*: Nepali characters are mapped to standard digits/letters to correct common OCR transcription glitches.
  - *Latin*: English plates apply a substitution step (e.g., converting `HEP` to `NEP`, `BAGIAT` to `BAGMATI`) to clean up common blocky font misreadings.

---

## 3. Frequently Asked Questions (FAQ)

### Q1: Is the YOLO model pre-trained?
**Yes.** The YOLO model (`license_plate.pt`) is a pre-trained deep learning detector optimized to output bounding coordinates `[x_min, y_min, x_max, y_max]` specifically for vehicle license plates. It does not classify the colors or read text; it acts strictly as the localization front-end.

### Q2: How are the CNN models trained?
The CNN classifiers are trained using a **two-phase transfer learning** approach in Keras:
1. **Backbones**: Pre-trained convolutional backbones from ImageNet (`EfficientNetB0` for BG, `EfficientNetB1` for Font, and `EfficientNetB2` for Blur).
2. **Phase 1 (Feature Extraction)**: The backbone is frozen, and only the dense classifier head is trained with a high learning rate (`1e-3`) using AdamW.
3. **Phase 2 (Fine-Tuning)**: The top layers of the backbone (last ~30 layers) are unfrozen, and the model is trained with a low learning rate (`5e-5`) to specialize on plate textures.
4. **Synthetic Data**: Training data is generated synthetically (`data_preparation.py` and `train_devanagari.py`) by rendering license plate formats onto various color palettes and applying noise, blur, scaling, and warp distortions.

### Q3: What is the Squeeze-and-Excitation (SE) Block?
The SE block is an attention mechanism integrated at the top of the convolutional backbones. It:
1. Squeezes spatial features into a channel descriptor vector using Global Average Pooling.
2. Excites features by learning channel-wise relationships via dense layers (Sigmoid output).
3. Scales the feature maps by multiplying original activations by these learned channel importances.
This helps the models focus on the most informative channels representing color and texture boundary structures.

### Q4: Why does the system fall back to rule-based classification?
Neural networks trained on clean synthetic data can struggle with real-world artifacts (like shadows, dust, uneven lighting, or vehicle bodies). Because the rule-based OpenCV/HSV color masking uses strict physical definitions of color bounds, it is highly accurate on clear images. The hybrid fallback ensures the system gets the "best of both worlds"—leveraging deep learning features when confident, but reverting to physical rules when unconfident.

### Q5: How do I run the project?
- **Rule-based & OCR Test**:
  ```bash
  python test_image.py
  ```
- **Full Pipeline (YOLO + CNN + OCR)**:
  ```bash
  python main_enhanced.py
  ```
- **Dataset Generation**:
  ```bash
  python data_preparation.py
  ```
- **Model Training**:
  ```bash
  python train_models.py
  ```

### Q6: Where are the outputs saved?
For any target image (e.g., `images/image.png`), the system automatically saves:
1. An annotated image output showing the crop and panel statistics at `images/test_result.png`.
2. A structured JSON dump of all analysis variables (confidence scores, colors, raw/corrected OCR text) at `images/test_result.json`.
