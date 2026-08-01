"""
ENHANCED VEHICLE & LICENSE PLATE DETECTION SYSTEM
Integrates:
  - YOLO for license plate detection
  - CNN Blur Classifier (EfficientNetB0)
  - CNN Color Classifier (MobileNetV2)
  - CNN Font Color Classifier (ResNet50)
  - EasyOCR for text extraction
  
Author: BATO-AI System
Version: 2.0
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import easyocr
import tensorflow as tf
from tensorflow import keras
from dataclasses import dataclass, asdict
from datetime import datetime

# Import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for the detection system"""
    BASE_DIR = Path(__file__).resolve().parent
    
    # Model paths
    YOLO_MODEL = BASE_DIR / "license_plate.pt"
    BLUR_MODEL = BASE_DIR / "blur_classifier_final.h5"
    COLOR_MODEL = BASE_DIR / "color_classifier_final.h5"
    FONT_MODEL = BASE_DIR / "font_classifier_final.h5"
    
    # Detection thresholds
    YOLO_CONF = 0.3
    BLUR_THRESHOLD = 100  # Laplacian variance threshold
    MIN_CONFIDENCE = 0.75  # Combined confidence threshold
    
    # CNN input size
    CNN_INPUT_SIZE = (224, 224)
    
    # OCR settings
    OCR_LANGUAGES = ['ne', 'en']  # Nepali, English
    OCR_CONFIDENCE = 0.3
    
    # Class mappings
    BLUR_CLASSES = {0: 'clear', 1: 'blurry'}
    COLOR_CLASSES = {0: 'yellow', 1: 'white', 2: 'green'}
    FONT_CLASSES = {0: 'black', 1: 'white', 2: 'red', 3: 'green'}
    
    # Vehicle type mappings
    VEHICLE_TYPE_MAP = {
        'yellow': 'Private Vehicle (Two-wheeler/Four-wheeler)',
        'white': 'Commercial Vehicle (Taxi/Bus)',
        'green': 'Government Vehicle'
    }

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QualityMetrics:
    """License plate quality metrics"""
    blur_status: str
    blur_confidence: float
    laplacian_score: float
    is_readable: bool
    processing_time_ms: float

@dataclass
class ColorAnalysis:
    """Plate color analysis results"""
    plate_color: str
    confidence: float
    vehicle_type: str
    is_private: bool
    is_government: bool

@dataclass
class FontAnalysis:
    """Font color analysis results"""
    font_color: str
    confidence: float
    indicates_private: bool
    legal_validity: str  # Based on color standards

@dataclass
class DetectionResult:
    """Complete detection result for a single plate"""
    detection_id: int
    timestamp: int
    datetime_str: str
    
    # Bounding box
    bounding_box: Dict[str, int]
    
    # OCR
    plate_number: str
    ocr_confidence: float
    
    # Quality
    quality: QualityMetrics
    
    # Color
    plate_color: ColorAnalysis
    
    # Font
    font_color: FontAnalysis
    
    # Confidence scores
    yolo_confidence: float
    combined_confidence: float
    is_reliable: bool
    
    # Metadata
    processing_time_ms: float

# ============================================================================
# PREPROCESSING MODULE
# ============================================================================

class LicensePlatePreprocessor:
    """Advanced preprocessing for license plate images"""
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """
        CLAHE - Contrast Limited Adaptive Histogram Equalization
        Improves text visibility
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """
        Non-local means denoising
        Removes noise while preserving edges (text)
        """
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=10,
            hForColorComponents=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    @staticmethod
    def calculate_blur_score(image: np.ndarray) -> float:
        """
        Laplacian variance method for blur detection
        Higher variance = sharper/clearer image
        Typically: > 100 = clear, < 50 = blurry
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)
    
    @staticmethod
    def preprocess_for_cnn(image: np.ndarray, 
                          target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Complete preprocessing pipeline for CNN inference
        """
        # Resize
        resized = cv2.resize(image, target_size)
        
        # Enhance contrast
        enhanced = LicensePlatePreprocessor.enhance_contrast(resized)
        
        # Denoise
        denoised = LicensePlatePreprocessor.denoise(enhanced)
        
        # Normalize to [0, 1]
        normalized = denoised.astype('float32') / 255.0
        
        return normalized
    
    @staticmethod
    def extract_text_region_mask(image: np.ndarray) -> np.ndarray:
        """
        Extract binary mask of text regions
        Useful for font color analysis
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Thresholding
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return morph

# ============================================================================
# CNN INFERENCE MODULE
# ============================================================================

class CNNInference:
    """Handle all CNN model inferences"""
    
    def __init__(self, config: Config):
        self.config = config
        self.blur_model = None
        self.color_model = None
        self.font_model = None
        self._load_models()
    
    def _load_models(self):
        """Load all CNN models (with .keras fallback and graceful degradation)"""
        print("Loading CNN Models...")

        def _try_load(h5_path):
            """Try .keras first (new format), then .h5 (legacy)"""
            keras_path = Path(str(h5_path).replace('.h5', '.keras'))
            for p in [keras_path, h5_path]:
                if p.exists():
                    try:
                        m = keras.models.load_model(str(p), compile=False)
                        return m, p.name
                    except Exception as e:
                        print(f"    Error loading {p.name}: {e}")
            return None, None

        self.blur_model,  name = _try_load(self.config.BLUR_MODEL)
        if self.blur_model:  print(f"  ✓ Blur Classifier loaded ({name})")
        else:                print(f"  ⚠ Blur model not found — using Laplacian fallback")

        self.color_model, name = _try_load(self.config.COLOR_MODEL)
        if self.color_model: print(f"  ✓ Color Classifier loaded ({name})")
        else:                print(f"  ⚠ Color model not found — using HSV fallback")

        self.font_model,  name = _try_load(self.config.FONT_MODEL)
        if self.font_model:  print(f"  ✓ Font Classifier loaded ({name})")
        else:                print(f"  ⚠ Font model not found — using HSV fallback")
    
    def _hsv_color_fallback(self, image: np.ndarray) -> Dict:
        """Rule-based HSV plate color when CNN model is unavailable"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            'yellow': cv2.inRange(hsv, (15, 60, 60),  (40, 255, 255)),
            'white':  cv2.inRange(hsv, (0,  0,  160), (180, 50, 255)),
            'green':  cv2.inRange(hsv, (40, 60, 40),  (90, 255, 200)),
        }
        total = image.shape[0] * image.shape[1]
        scores = {c: int(m.sum() / 255) / max(total, 1) for c, m in masks.items()}
        best = max(scores, key=scores.get)
        denom = sum(scores.values()) or 1.0
        return {'color': best, 'confidence': round(scores[best] / denom, 4),
                'vehicle_type': self.config.VEHICLE_TYPE_MAP.get(best, 'Unknown')}

    def _hsv_font_fallback(self, image: np.ndarray) -> Dict:
        """Rule-based HSV font color when CNN model is unavailable"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, dark  = cv2.threshold(gray, 80,  255, cv2.THRESH_BINARY_INV)
        _, light = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        masks = {
            'black': dark, 'white': light,
            'red':   cv2.inRange(hsv, (0,  80, 80), (15, 255, 255)),
            'green': cv2.inRange(hsv, (40, 80, 40), (90, 255, 200)),
        }
        total = image.shape[0] * image.shape[1]
        scores = {c: int(m.sum() / 255) / max(total, 1) for c, m in masks.items()}
        best = max(scores, key=scores.get)
        denom = sum(scores.values()) or 1.0
        return {'color': best, 'confidence': round(scores[best] / denom, 4),
                'legal_validity': 'Rule-based estimate'}

    def classify_blur_quality(self, cropped_plate: np.ndarray) -> Dict:
        """Classify if plate is clear or blurry (CNN with Laplacian fallback)"""
        # Always compute Laplacian score
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        lap_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if self.blur_model is None:
            status = 'clear' if lap_score > 80 else 'blurry'
            conf = 1.0 / (1.0 + np.exp(-(lap_score - 80) / 30.0))
            return {'quality': status, 'confidence': round(conf, 4),
                    'laplacian_score': round(lap_score, 2)}
        
        try:
            resized = cv2.resize(cropped_plate, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            preprocessed = np.expand_dims(rgb.astype('float32') / 255.0, axis=0)
            prediction = self.blur_model.predict(preprocessed, verbose=0)
            blur_class = np.argmax(prediction[0])
            blur_confidence = float(np.max(prediction[0]))
            return {
                'quality': self.config.BLUR_CLASSES[blur_class],
                'confidence': blur_confidence,
                'laplacian_score': round(lap_score, 2)
            }
        except Exception as e:
            print(f"Error in blur classification: {e}")
            return {'quality': 'unknown', 'confidence': 0.0, 'laplacian_score': 0.0}
    
    def classify_plate_color(self, cropped_plate: np.ndarray) -> Dict:
        """Classify plate background color (CNN with HSV fallback)"""
        if self.color_model is None:
            return self._hsv_color_fallback(cropped_plate)
        
        try:
            resized = cv2.resize(cropped_plate, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            preprocessed = np.expand_dims(rgb.astype('float32') / 255.0, axis=0)
            
            prediction = self.color_model.predict(
                preprocessed,
                verbose=0
            )
            
            color_class = np.argmax(prediction[0])
            color_confidence = float(np.max(prediction[0]))
            color_name = self.config.COLOR_CLASSES[color_class]
            
            return {
                'color': color_name,
                'confidence': color_confidence,
                'vehicle_type': self.config.VEHICLE_TYPE_MAP.get(
                    color_name, 'Unknown'
                )
            }
        except Exception as e:
            print(f"Error in color classification: {e}")
            return {'color': 'unknown', 'confidence': 0.0, 'vehicle_type': 'unknown'}
    
    def classify_font_color(self, cropped_plate: np.ndarray) -> Dict:
        """Classify font/text color (CNN with HSV fallback)"""
        if self.font_model is None:
            return self._hsv_font_fallback(cropped_plate)
        
        try:
            resized = cv2.resize(cropped_plate, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            preprocessed = np.expand_dims(rgb.astype('float32') / 255.0, axis=0)
            
            prediction = self.font_model.predict(
                preprocessed,
                verbose=0
            )
            
            font_class = np.argmax(prediction[0])
            font_confidence = float(np.max(prediction[0]))
            font_name = self.config.FONT_CLASSES[font_class]
            
            # Legal validity based on font color
            legal_mapping = {
                'black': 'Legal - Private Vehicle',
                'white': 'Legal - Government',
                'red': 'Special - Emergency Services',
                'green': 'Legal - Eco-friendly'
            }
            
            return {
                'color': font_name,
                'confidence': font_confidence,
                'legal_validity': legal_mapping.get(font_name, 'Unknown')
            }
        except Exception as e:
            print(f"Error in font classification: {e}")
            return {'color': 'unknown', 'confidence': 0.0, 'legal': 'unknown'}

# ============================================================================
# OCR MODULE
# ============================================================================

class OCRProcessor:
    """Handle OCR processing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.ocr_reader = easyocr.Reader(config.OCR_LANGUAGES)
    
    def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Extract text from license plate using EasyOCR
        Returns: (extracted_text, average_confidence)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            results = self.ocr_reader.readtext(gray)
            
            if not results:
                return "", 0.0
            
            # Filter by confidence threshold
            valid_results = [
                (text, prob) for _, text, prob in results 
                if prob > self.config.OCR_CONFIDENCE
            ]
            
            if not valid_results:
                return "", 0.0
            
            extracted_text = " ".join([text for text, _ in valid_results])
            avg_confidence = np.mean([prob for _, prob in valid_results])
            
            return extracted_text.strip(), float(avg_confidence)
        
        except Exception as e:
            print(f"Error in OCR: {e}")
            return "", 0.0

# ============================================================================
# MAIN DETECTION PIPELINE
# ============================================================================

class VehicleDetectionPipeline:
    """Complete detection pipeline"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Load YOLO model
        if YOLO is None:
            raise RuntimeError("YOLO model not available")
        
        print("Initializing Vehicle Detection Pipeline...")
        self.detector = YOLO(str(self.config.YOLO_MODEL))
        
        # Initialize CNN inference
        self.cnn_inference = CNNInference(self.config)
        
        # Initialize OCR
        self.ocr_processor = OCRProcessor(self.config)
        
        print("✓ Pipeline initialized successfully\n")
    
    def process_vehicle_image(self, image_path: str) -> Optional[List[DetectionResult]]:
        """
        Main processing function for vehicle images
        Returns list of DetectionResult objects
        """
        pipeline_start = time.time()
        
        # Load image
        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = self.config.BASE_DIR / img_path
        
        if not img_path.exists():
            print(f"Error: Image not found at {img_path}")
            return None
        
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Error: Could not load image at {img_path}")
            return None
        
        print(f"Processing image: {img_path.name}")
        print(f"Image size: {frame.shape}")
        
        # YOLO detection
        yolo_start = time.time()
        yolo_results = self.detector(frame, conf=self.config.YOLO_CONF)[0]
        yolo_time = (time.time() - yolo_start) * 1000
        
        detected_vehicles = []
        
        for idx, box in enumerate(yolo_results.boxes):
            detection_start = time.time()
            
            # Extract bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_plate = frame[y1:y2, x1:x2]
            
            if cropped_plate.size == 0:
                continue
            
            # ===== STAGE 1: Blur Quality =====
            blur_info = self.cnn_inference.classify_blur_quality(cropped_plate)
            
            # ===== STAGE 2: Plate Color =====
            color_info = self.cnn_inference.classify_plate_color(cropped_plate)
            
            # ===== STAGE 3: Font Color =====
            font_info = self.cnn_inference.classify_font_color(cropped_plate)
            
            # ===== STAGE 4: Text Extraction =====
            plate_text, ocr_conf = self.ocr_processor.extract_text(cropped_plate)
            
            # ===== STAGE 5: Confidence Scoring =====
            combined_confidence = (
                blur_info['confidence'] * 0.2 +
                color_info['confidence'] * 0.25 +
                font_info['confidence'] * 0.25 +
                float(box.conf[0]) * 0.3
            )
            
            # ===== STAGE 6: Build Result =====
            quality_metrics = QualityMetrics(
                blur_status=blur_info['quality'],
                blur_confidence=blur_info['confidence'],
                laplacian_score=blur_info['laplacian_score'],
                is_readable=blur_info['quality'] == 'clear' and blur_info['laplacian_score'] > self.config.BLUR_THRESHOLD,
                processing_time_ms=(time.time() - detection_start) * 1000
            )
            
            color_analysis = ColorAnalysis(
                plate_color=color_info['color'],
                confidence=color_info['confidence'],
                vehicle_type=color_info['vehicle_type'],
                is_private='Private' in color_info['vehicle_type'],
                is_government='Government' in color_info['vehicle_type']
            )
            
            font_analysis = FontAnalysis(
                font_color=font_info['color'],
                confidence=font_info['confidence'],
                indicates_private=font_info['color'] == 'black',
                legal_validity=font_info['legal_validity']
            )
            
            result = DetectionResult(
                detection_id=idx + 1,
                timestamp=int(time.time()),
                datetime_str=datetime.now().isoformat(),
                bounding_box={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                plate_number=plate_text,
                ocr_confidence=ocr_conf,
                quality=quality_metrics,
                plate_color=color_analysis,
                font_color=font_analysis,
                yolo_confidence=float(box.conf[0]),
                combined_confidence=combined_confidence,
                is_reliable=combined_confidence > self.config.MIN_CONFIDENCE,
                processing_time_ms=(time.time() - detection_start) * 1000
            )
            
            detected_vehicles.append(result)
        
        total_time = (time.time() - pipeline_start) * 1000
        
        print(f"\nDetection Summary:")
        print(f"  Plates detected: {len(detected_vehicles)}")
        print(f"  YOLO inference: {yolo_time:.2f}ms")
        print(f"  Total processing: {total_time:.2f}ms")
        
        return detected_vehicles if detected_vehicles else None
    
    def results_to_json(self, results: List[DetectionResult], 
                       indent: bool = True) -> str:
        """Convert results to JSON"""
        data = [asdict(result) for result in results]
        indent_level = 2 if indent else None
        return json.dumps(data, indent=indent_level, ensure_ascii=False)
    
    def save_results(self, results: List[DetectionResult], 
                    output_path: str):
        """Save results to JSON file"""
        json_data = self.results_to_json(results)
        
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        print(f"Results saved to: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Initialize pipeline
        pipeline = VehicleDetectionPipeline()
        
        # Process image
        test_image = "images/image.png"
        results = pipeline.process_vehicle_image(test_image)
        
        if results:
            print("\n" + "="*60)
            print("DETECTION RESULTS")
            print("="*60)
            
            # Print to console
            print(pipeline.results_to_json(results))
            
            # Save to file
            pipeline.save_results(results, "detection_results.json")
            
            # Print summary
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            for result in results:
                print(f"\nDetection #{result.detection_id}:")
                print(f"  Plate Number: {result.plate_number}")
                print(f"  Color: {result.plate_color.plate_color} ({result.plate_color.vehicle_type})")
                print(f"  Font Color: {result.font_color.font_color}")
                print(f"  Quality: {result.quality.blur_status}")
                print(f"  Readable: {'Yes' if result.quality.is_readable else 'No'}")
                print(f"  Confidence: {result.combined_confidence:.2%}")
                print(f"  Reliable: {'Yes' if result.is_reliable else 'No'}")
        else:
            print("No license plates detected")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()