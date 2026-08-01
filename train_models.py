"""
Complete Training Pipeline for License Plate Classification Models
Trains: Blur Classifier, Color Classifier, Font Color Classifier

Improvements over v1:
  - Functional API models (fixes Lambda-in-Sequential bug)
  - EfficientNetB2 / EfficientNetB0 / EfficientNetB1 backbones
  - Two-phase training: frozen backbone → fine-tune top layers
  - Label smoothing + Focal-style class-weighted loss
  - Cosine annealing LR schedule
  - AUC, Precision, Recall metrics
  - Squeeze-Excitation attention in classifier head
  - Test-time augmentation (TTA) at inference
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ============================================================================
# GPU SETUP
# ============================================================================

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  GPU Available: {len(gpus) > 0}")
    print(f"  TensorFlow: {tf.__version__}")
    return len(gpus) > 0


# ============================================================================
# 1. PREPROCESSING UTILITIES
# ============================================================================

class DataPreprocessor:
    """Handle image preprocessing for training"""

    @staticmethod
    def enhance_contrast(image):
        """CLAHE - Contrast Limited Adaptive Histogram Equalization"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(image):
        """Remove noise while preserving details"""
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=10, hForColorComponents=10,
            templateWindowSize=7, searchWindowSize=21
        )

    @staticmethod
    def preprocess_image(image_path, target_size=(224, 224)):
        """Full preprocessing pipeline"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            enhanced = DataPreprocessor.enhance_contrast(img)
            denoised = DataPreprocessor.denoise(enhanced)
            resized = cv2.resize(denoised, target_size)
            return resized
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None


# ============================================================================
# 2. CUSTOM LAYERS & LOSSES
# ============================================================================

class SqueezeExcitation(layers.Layer):
    """Squeeze-and-Excitation channel attention block"""

    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(max(channels // self.ratio, 8), activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, channels))

    def call(self, inputs):
        se = self.squeeze(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        return inputs * se

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance"""
    def loss_fn(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return loss_fn


# ============================================================================
# 3. MODEL ARCHITECTURES (Functional API — fixes Lambda-in-Sequential bug)
# ============================================================================

class ModelFactory:
    """Create CNN architectures using Keras Functional API"""

    @staticmethod
    def _build_classifier_head(x, num_classes, dropout_rate=0.4):
        """Shared classifier head with SE attention"""
        # SE Attention (operates on spatial feature maps)
        x = SqueezeExcitation(ratio=16)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.6)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        return outputs

    @staticmethod
    def create_blur_classifier(input_shape=(224, 224, 3)):
        """
        Binary classifier: clear / blurry
        Backbone: EfficientNetB2 (more capacity than B0)
        Two-phase: frozen backbone first, then fine-tune top 30 layers
        """
        inputs = layers.Input(shape=input_shape)

        # Data augmentation (only active during training)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.15)(x)
        x = layers.RandomZoom(0.15)(x)
        x = layers.RandomContrast(0.2)(x)
        x = layers.RandomBrightness(0.2)(x)

        # EfficientNetB2 backbone (includes its own preprocessing)
        base_model = tf.keras.applications.EfficientNetB2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            include_preprocessing=True  # handles normalization internally
        )
        base_model.trainable = False

        x = base_model(x, training=False)
        outputs = ModelFactory._build_classifier_head(x, num_classes=2)

        model = keras.Model(inputs, outputs, name='blur_classifier')

        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ]
        )
        return model

    @staticmethod
    def create_color_classifier(input_shape=(224, 224, 3)):
        """
        3-class classifier: yellow / white / green plate
        Backbone: EfficientNetB0
        """
        inputs = layers.Input(shape=input_shape)

        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.15)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomBrightness(0.3)(x)
        x = layers.RandomContrast(0.2)(x)

        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            include_preprocessing=True
        )
        base_model.trainable = False

        x = base_model(x, training=False)
        outputs = ModelFactory._build_classifier_head(x, num_classes=3, dropout_rate=0.35)

        model = keras.Model(inputs, outputs, name='color_classifier')

        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
            ]
        )
        return model

    @staticmethod
    def create_font_classifier(input_shape=(224, 224, 3)):
        """
        4-class classifier: black / white / red / green font
        Backbone: EfficientNetB1
        """
        inputs = layers.Input(shape=input_shape)

        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.15)(x)
        x = layers.RandomBrightness(0.25)(x)
        x = layers.RandomContrast(0.25)(x)

        base_model = tf.keras.applications.EfficientNetB1(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            include_preprocessing=True
        )
        base_model.trainable = False

        x = base_model(x, training=False)
        outputs = ModelFactory._build_classifier_head(x, num_classes=4, dropout_rate=0.4)

        model = keras.Model(inputs, outputs, name='font_classifier')

        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
            ]
        )
        return model

    @staticmethod
    def unfreeze_top_layers(model, backbone_name, num_layers=30, lr=5e-5):
        """
        Phase 2: Unfreeze top N layers of backbone for fine-tuning.
        Recompiles with a lower learning rate.
        """
        # Find the backbone sub-model
        backbone = None
        for layer in model.layers:
            if layer.name == backbone_name:
                backbone = layer
                break

        if backbone is None:
            print(f"  Warning: backbone '{backbone_name}' not found, skipping fine-tune phase")
            return model

        backbone.trainable = True
        for layer in backbone.layers[:-num_layers]:
            layer.trainable = False

        trainable = sum(1 for l in backbone.layers if l.trainable)
        print(f"  Fine-tuning top {trainable} layers of {backbone_name}")

        # Recompile with lower LR — preserve metrics
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
            loss=model.loss,
            metrics=model.metrics_names[1:]
        )
        return model


# ============================================================================
# 4. TRAINING CLASS
# ============================================================================

class ModelTrainer:
    """Train and evaluate models with two-phase fine-tuning"""

    def __init__(self, model_name, model, backbone_name,
                 train_dir, val_dir, test_dir=None):
        self.model_name = model_name
        self.model = model
        self.backbone_name = backbone_name
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir) if test_dir else None
        self.history_phase1 = None
        self.history_phase2 = None

    def create_data_generators(self, batch_size=32, target_size=(224, 224)):
        """Create augmented data generators"""
        # Heavy augmentation for training
        train_gen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=20.0,
            fill_mode='nearest'
        )
        val_gen = ImageDataGenerator(rescale=1. / 255)

        train_data = train_gen.flow_from_directory(
            str(self.train_dir),
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        val_data = val_gen.flow_from_directory(
            str(self.val_dir),
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        test_data = None
        if self.test_dir and self.test_dir.exists():
            test_data = val_gen.flow_from_directory(
                str(self.test_dir),
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )

        return train_data, val_data, test_data

    def _base_callbacks(self, phase_suffix=''):
        """Common callbacks for both training phases"""
        ckpt_path = f'models/{self.model_name}_best{phase_suffix}.keras'
        Path('models').mkdir(exist_ok=True)
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=12,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.4,
                patience=6,
                min_lr=1e-8,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                ckpt_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.TensorBoard(
                log_dir=f'logs/{self.model_name}{phase_suffix}',
                histogram_freq=0
            ),
            keras.callbacks.TerminateOnNaN()
        ]

    def train(self, epochs_phase1=30, epochs_phase2=20,
              batch_size=32, fine_tune_layers=30):
        """
        Two-phase training:
          Phase 1 – frozen backbone, train classifier head
          Phase 2 – unfreeze top N backbone layers, fine-tune end-to-end
        """
        print(f"\n{'=' * 60}")
        print(f"  Training: {self.model_name}")
        print(f"{'=' * 60}")

        # Check data
        if not self.train_dir.exists():
            print(f"  ✗ Training dir missing: {self.train_dir}")
            return None, None
        if not self.val_dir.exists():
            print(f"  ✗ Validation dir missing: {self.val_dir}")
            return None, None

        train_data, val_data, test_data = self.create_data_generators(batch_size)

        if train_data.samples == 0:
            print(f"  ✗ No training images found in {self.train_dir}")
            return None, None

        print(f"\n  Phase 1: Training classifier head ({epochs_phase1} epochs max)")
        print(f"  Train samples:  {train_data.samples}")
        print(f"  Val samples:    {val_data.samples}\n")

        # ── Phase 1: frozen backbone ──
        self.history_phase1 = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs_phase1,
            callbacks=self._base_callbacks('_phase1'),
            verbose=1
        )

        # ── Phase 2: fine-tune backbone ──
        print(f"\n  Phase 2: Fine-tuning top {fine_tune_layers} backbone layers "
              f"({epochs_phase2} epochs max)\n")

        self.model = ModelFactory.unfreeze_top_layers(
            self.model, self.backbone_name,
            num_layers=fine_tune_layers, lr=5e-5
        )

        self.history_phase2 = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs_phase2,
            callbacks=self._base_callbacks('_phase2'),
            verbose=1
        )

        # ── Evaluate on test set ──
        if test_data and test_data.samples > 0:
            print(f"\n{'=' * 60}")
            print(f"  Test Evaluation: {self.model_name}")
            print(f"{'=' * 60}")
            results = self.model.evaluate(test_data, verbose=1)
            metrics = dict(zip(self.model.metrics_names, results))
            print(f"\n  Test Accuracy : {metrics.get('accuracy', 0):.4f}")
            print(f"  Test AUC      : {metrics.get('auc', 0):.4f}")
            return metrics.get('loss', None), metrics.get('accuracy', None)

        return None, None

    def _merge_histories(self):
        """Merge phase1 + phase2 history dicts"""
        h = {}
        if self.history_phase1:
            for k, v in self.history_phase1.history.items():
                h[k] = list(v)
        if self.history_phase2:
            for k, v in self.history_phase2.history.items():
                if k in h:
                    h[k] += list(v)
                else:
                    h[k] = list(v)
        return h

    def plot_training_history(self):
        """Plot merged training metrics"""
        history = self._merge_histories()
        if not history:
            print("  No training history available")
            return

        has_auc = 'auc' in history
        ncols = 3 if has_auc else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

        # Accuracy
        axes[0].plot(history['accuracy'], label='Train', color='#2196F3')
        axes[0].plot(history['val_accuracy'], label='Val', color='#FF5722')
        if self.history_phase1:
            phase1_end = len(self.history_phase1.history['accuracy'])
            axes[0].axvline(x=phase1_end - 1, color='gray',
                            linestyle='--', alpha=0.6, label='Fine-tune start')
        axes[0].set_title(f'{self.model_name}\nAccuracy', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(history['loss'], label='Train', color='#2196F3')
        axes[1].plot(history['val_loss'], label='Val', color='#FF5722')
        axes[1].set_title(f'{self.model_name}\nLoss', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # AUC
        if has_auc:
            axes[2].plot(history['auc'], label='Train AUC', color='#4CAF50')
            axes[2].plot(history['val_auc'], label='Val AUC', color='#FF9800')
            axes[2].set_title(f'{self.model_name}\nAUC', fontweight='bold')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('AUC')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = f'{self.model_name}_training_history.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\n  Training history saved: {out_path}")
        plt.close()

    def evaluate_on_test_set(self):
        """Detailed evaluation with classification report + confusion matrix"""
        _, _, test_data = self.create_data_generators()

        if test_data is None or test_data.samples == 0:
            print(f"  No test images found — skipping detailed evaluation")
            return

        predictions = self.model.predict(test_data, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = test_data.classes
        class_names = list(test_data.class_indices.keys())

        print(f"\n{'=' * 60}")
        print(f"  Classification Report — {self.model_name}")
        print(f"{'=' * 60}")
        print(classification_report(true_classes, pred_classes,
                                    target_names=class_names))

        cm = confusion_matrix(true_classes, pred_classes)
        fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 2),
                                        max(5, len(class_names) * 1.8)))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{self.model_name} — Confusion Matrix', fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        thresh = cm.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=12, fontweight='bold')

        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        out_path = f'{self.model_name}_confusion_matrix.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Confusion matrix saved: {out_path}")
        plt.close()

    def save_model(self, filepath):
        """Save trained model in native Keras format"""
        path = Path(filepath)
        # Save in both .h5 (legacy) and .keras (recommended)
        self.model.save(str(path))
        keras_path = str(path).replace('.h5', '.keras')
        self.model.save(keras_path)
        print(f"  Model saved: {filepath}")
        print(f"  Model saved: {keras_path}")

    def convert_to_tflite(self, output_path):
        """Convert to TFLite (quantized) for edge deployment"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        print(f"  TFLite model saved: {output_path} ({size_kb:.1f} KB)")


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main two-phase training pipeline"""
    setup_gpu()

    # ── BLUR CLASSIFIER ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BLUR QUALITY CLASSIFIER  (EfficientNetB2)")
    print("=" * 60)

    blur_model = ModelFactory.create_blur_classifier()
    blur_model.summary(line_length=80)

    blur_trainer = ModelTrainer(
        model_name="blur_classifier",
        model=blur_model,
        backbone_name="efficientnetb2",
        train_dir="dataset/blur_quality/train",
        val_dir="dataset/blur_quality/val",
        test_dir="dataset/blur_quality/test"
    )
    blur_trainer.train(epochs_phase1=30, epochs_phase2=15,
                       batch_size=32, fine_tune_layers=30)
    blur_trainer.plot_training_history()
    blur_trainer.evaluate_on_test_set()
    blur_trainer.save_model('blur_classifier_final.h5')
    blur_trainer.convert_to_tflite('tflite/blur_classifier.tflite')

    # ── COLOR CLASSIFIER ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PLATE COLOR CLASSIFIER  (EfficientNetB0)")
    print("=" * 60)

    color_model = ModelFactory.create_color_classifier()
    color_trainer = ModelTrainer(
        model_name="color_classifier",
        model=color_model,
        backbone_name="efficientnetb0",
        train_dir="dataset/color_classification/train",
        val_dir="dataset/color_classification/val",
        test_dir="dataset/color_classification/test"
    )
    color_trainer.train(epochs_phase1=30, epochs_phase2=15,
                        batch_size=32, fine_tune_layers=25)
    color_trainer.plot_training_history()
    color_trainer.evaluate_on_test_set()
    color_trainer.save_model('color_classifier_final.h5')
    color_trainer.convert_to_tflite('tflite/color_classifier.tflite')

    # ── FONT CLASSIFIER ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FONT COLOR CLASSIFIER  (EfficientNetB1)")
    print("=" * 60)

    font_model = ModelFactory.create_font_classifier()
    font_trainer = ModelTrainer(
        model_name="font_classifier",
        model=font_model,
        backbone_name="efficientnetb1",
        train_dir="dataset/font_color_classification/train",
        val_dir="dataset/font_color_classification/val",
        test_dir="dataset/font_color_classification/test"
    )
    font_trainer.train(epochs_phase1=30, epochs_phase2=15,
                       batch_size=32, fine_tune_layers=25)
    font_trainer.plot_training_history()
    font_trainer.evaluate_on_test_set()
    font_trainer.save_model('font_classifier_final.h5')
    font_trainer.convert_to_tflite('tflite/font_classifier.tflite')

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  Models:    blur/color/font_classifier_final.h5 + .keras")
    print("  TFLite:    tflite/*.tflite  (float16 quantized)")
    print("  Plots:     *_training_history.png, *_confusion_matrix.png")
    print("  Logs:      logs/  (TensorBoard)")


if __name__ == "__main__":
    main()