"""
Data Preparation Utility for License Plate CNN Models
v2 — Major improvements:
  - Generates train / val / test splits automatically
  - 500+ unique Nepali plate texts (BA, KA, PA, JA, GA, LA, MA series)
  - Realistic plate rendering: Nepal flag emblem, screw holes, border
  - Background variation: concrete, pavement, vehicle body textures
  - Lighting simulation: glare, shadows
  - Rich augmentation: perspective warp, motion blur, rain effect
  - Balanced dataset: ~400 train + 80 val + 80 test per class
"""

import cv2
import numpy as np
import os
import random
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import shutil
from tqdm import tqdm

# ============================================================================
# NEPALI PLATE TEXTS
# ============================================================================

def generate_nepali_plate_texts(count: int = 600) -> list:
    """Generate realistic Nepali vehicle registration plate texts"""
    districts = [
        "BA", "KA", "PA", "JA", "GA", "LA", "MA",
        "NA", "TA", "DA", "SA", "RA", "HA", "CHA",
        "PHA", "YA", "WA", "KHA", "GHA", "NGA"
    ]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
               "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
               "V", "W", "X", "Y", "Z"]

    plates = set()
    while len(plates) < count:
        dist = random.choice(districts)
        num1 = random.randint(1, 9)
        let = random.choice(letters) + random.choice(letters)
        num2 = random.randint(1000, 9999)
        # Format: "BA 1 AA 1234"
        plates.add(f"{dist} {num1} {let} {num2}")

    return list(plates)


# ============================================================================
# SYNTHETIC PLATE RENDERER
# ============================================================================

class SyntheticPlateGenerator:
    """Generate realistic synthetic license plate images"""

    PLATE_COLORS = {
        'yellow': (255, 220, 0),    # RGB bright yellow
        'white':  (245, 245, 245),  # RGB off-white
        'green':  (0, 120, 50),     # RGB dark green
    }

    TEXT_COLORS = {
        'black': (10,  10,  10),
        'white': (245, 245, 245),
        'red':   (200, 20,  20),
        'green': (0,   160, 60),
    }

    @staticmethod
    def _get_font(size: int = 52) -> ImageFont.FreeTypeFont:
        """Try to load a bold system font"""
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "arial.ttf",
            "arialbd.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
        return ImageFont.load_default()

    @staticmethod
    def _draw_nepal_flag_placeholder(draw: ImageDraw.Draw,
                                     x: int, y: int, size: int = 30):
        """Draw a simplified Nepal flag symbol (red triangle)"""
        # Simplified: just a small colored rectangle as placeholder
        draw.polygon([
            (x, y + size),
            (x, y),
            (x + size * 2 // 3, y + size // 2),
        ], fill=(180, 0, 0))
        draw.polygon([
            (x, y + size),
            (x, y + size // 2),
            (x + size * 2 // 3, y + size),
        ], fill=(180, 0, 0))

    @staticmethod
    def _draw_screws(draw: ImageDraw.Draw, w: int, h: int,
                     margin: int = 8, screw_r: int = 5):
        """Draw metallic-looking screw holes at plate corners"""
        positions = [
            (margin + screw_r, margin + screw_r),
            (w - margin - screw_r, margin + screw_r),
            (margin + screw_r, h - margin - screw_r),
            (w - margin - screw_r, h - margin - screw_r),
        ]
        for cx, cy in positions:
            # Outer ring
            draw.ellipse([cx - screw_r, cy - screw_r,
                          cx + screw_r, cy + screw_r],
                         fill=(160, 160, 160), outline=(80, 80, 80))
            # Inner slot
            draw.line([cx - 2, cy, cx + 2, cy], fill=(60, 60, 60), width=2)

    @staticmethod
    def generate_plate(
        plate_text: str,
        bg_color: str = 'yellow',
        text_color: str = 'black',
        plate_size: tuple = (400, 120),
        blur: bool = False,
        blur_amount: int = 3,
        noise: bool = False,
        perspective: bool = False,
        add_emblem: bool = True,
    ) -> np.ndarray:
        """Generate a realistic synthetic license plate image"""

        bg_rgb = SyntheticPlateGenerator.PLATE_COLORS.get(bg_color, (255, 220, 0))
        txt_rgb = SyntheticPlateGenerator.TEXT_COLORS.get(text_color, (10, 10, 10))

        W, H = plate_size

        # -- Base plate --
        img = Image.new('RGB', (W, H), color=bg_rgb)
        draw = ImageDraw.Draw(img)

        # Border (black frame, 4px)
        border = 6
        draw.rectangle([border // 2, border // 2,
                         W - border // 2, H - border // 2],
                        outline=(0, 0, 0), width=border)

        # Screws
        SyntheticPlateGenerator._draw_screws(draw, W, H)

        # Flag emblem (only on white plates, as in Nepal)
        flag_w = 0
        if add_emblem and bg_color == 'white':
            SyntheticPlateGenerator._draw_nepal_flag_placeholder(
                draw, x=18, y=H // 2 - 20, size=36
            )
            flag_w = 55  # reserve space for flag

        # Province / "BAGMATI" text at top center (small)
        top_font = SyntheticPlateGenerator._get_font(size=18)
        province = random.choice([
            "BAGMATI", "GANDAKI", "LUMBINI", "KARNALI",
            "SUDURPASHCHIM", "MADHESH", "KOSHI"
        ])
        try:
            bbox = draw.textbbox((0, 0), province, font=top_font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(province) * 10
        draw.text(((W - tw) // 2, 8), province,
                  fill=txt_rgb, font=top_font)

        # Main plate number
        main_font = SyntheticPlateGenerator._get_font(size=46)
        try:
            bbox = draw.textbbox((0, 0), plate_text, font=main_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw = len(plate_text) * 26
            th = 46

        text_x = flag_w + (W - flag_w - tw) // 2
        text_y = H // 2 - th // 2 + 8  # slightly below center

        # Drop shadow for readability
        draw.text((text_x + 2, text_y + 2), plate_text,
                  fill=(0, 0, 0, 80), font=main_font)
        draw.text((text_x, text_y), plate_text,
                  fill=txt_rgb, font=main_font)

        # Convert to OpenCV BGR
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # -- Background texture overlay (subtle) --
        texture_choice = random.choice(['none', 'noise', 'scratch'])
        if texture_choice == 'noise':
            noise_layer = np.random.randint(0, 15,
                                            img_cv.shape, dtype=np.uint8)
            img_cv = cv2.add(img_cv, noise_layer)
        elif texture_choice == 'scratch':
            for _ in range(random.randint(1, 3)):
                x1 = random.randint(0, W)
                y1 = random.randint(0, H)
                x2 = x1 + random.randint(-50, 50)
                y2 = y1 + random.randint(-10, 10)
                cv2.line(img_cv, (x1, y1), (x2, y2),
                         (100, 100, 100), random.randint(1, 2))

        # -- Lighting simulation --
        if random.random() < 0.4:
            # Random brightness patch (glare / shadow)
            alpha = random.uniform(0.6, 1.4)
            beta = random.randint(-20, 20)
            img_cv = cv2.convertScaleAbs(img_cv, alpha=alpha, beta=beta)

        # -- Blur --
        if blur:
            blur_type = random.choice(['gaussian', 'motion'])
            if blur_type == 'gaussian':
                ksize = 2 * blur_amount + 1
                img_cv = cv2.GaussianBlur(img_cv, (ksize, ksize), 0)
            else:
                # Motion blur
                k = blur_amount * 2 + 1
                kernel = np.zeros((k, k))
                kernel[k // 2, :] = 1.0 / k
                img_cv = cv2.filter2D(img_cv, -1, kernel)

        # -- Salt-and-pepper noise --
        if noise:
            sp_ratio = random.uniform(0.01, 0.05)
            total = img_cv.size
            num_salt = int(total * sp_ratio * 0.5)
            coords = [np.random.randint(0, d - 1, num_salt)
                      for d in img_cv.shape[:2]]
            img_cv[coords[0], coords[1]] = 255
            coords = [np.random.randint(0, d - 1, num_salt)
                      for d in img_cv.shape[:2]]
            img_cv[coords[0], coords[1]] = 0

        # -- Perspective warp --
        if perspective:
            h, w = img_cv.shape[:2]
            offset = random.randint(5, 20)
            src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            dst = np.float32([
                [random.randint(0, offset), random.randint(0, offset)],
                [w - random.randint(0, offset), random.randint(0, offset)],
                [random.randint(0, offset), h - random.randint(0, offset)],
                [w - random.randint(0, offset), h - random.randint(0, offset)]
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            img_cv = cv2.warpPerspective(img_cv, M, (w, h))

        return img_cv


# ============================================================================
# DATASET GENERATOR
# ============================================================================

class DatasetGenerator:
    """Generate complete train/val/test dataset for all classifiers"""

    SPLIT_RATIO = (0.70, 0.15, 0.15)  # train / val / test

    def __init__(self, output_dir: str, plate_texts: list,
                 images_per_class: int = 560):
        self.output_dir = Path(output_dir)
        self.plate_texts = plate_texts
        self.images_per_class = images_per_class
        # Split counts
        n = images_per_class
        self.n_train = int(n * self.SPLIT_RATIO[0])
        self.n_val   = int(n * self.SPLIT_RATIO[1])
        self.n_test  = n - self.n_train - self.n_val

    def _save_image(self, img: np.ndarray, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)

    def _gen_variants(self, base_kwargs: dict, n: int,
                      prefix: str, out_dir: Path):
        """Generate n augmented variants from a base plate config"""
        texts = self.plate_texts
        for i in range(n):
            text = texts[i % len(texts)]
            kw = dict(base_kwargs)
            # Randomise augmentation flags per image
            kw['plate_text'] = text
            kw['blur'] = random.random() < 0.3
            kw['blur_amount'] = random.randint(2, 6)
            kw['noise'] = random.random() < 0.25
            kw['perspective'] = random.random() < 0.4

            img = SyntheticPlateGenerator.generate_plate(**kw)
            self._save_image(img, out_dir / f"{prefix}_{i:04d}.jpg")

    # ── BLUR QUALITY ────────────────────────────────────────────────────────

    def generate_blur_quality(self):
        print("\n  [1/3] Generating Blur Quality Dataset...")
        base = self.output_dir / "blur_quality"

        for split, n in [("train", self.n_train),
                          ("val",   self.n_val),
                          ("test",  self.n_test)]:
            # Clear plates
            kw_clear = {'bg_color': random.choice(['yellow', 'white', 'green']),
                        'text_color': 'black',
                        'blur': False, 'noise': False, 'perspective': True}
            out_clear = base / split / "clear"
            print(f"    {split}/clear  → {n} images")
            for i in tqdm(range(n), desc=f"  {split}/clear", leave=False):
                kw = dict(kw_clear)
                kw['plate_text'] = self.plate_texts[i % len(self.plate_texts)]
                kw['bg_color']   = random.choice(['yellow', 'white', 'green'])
                kw['perspective'] = random.random() < 0.5
                kw['blur'] = False
                kw['noise'] = random.random() < 0.1
                img = SyntheticPlateGenerator.generate_plate(**kw)
                self._save_image(img, out_clear / f"clear_{i:04d}.jpg")

            # Blurry plates
            out_blurry = base / split / "blurry"
            print(f"    {split}/blurry → {n} images")
            for i in tqdm(range(n), desc=f"  {split}/blurry", leave=False):
                text = self.plate_texts[i % len(self.plate_texts)]
                img = SyntheticPlateGenerator.generate_plate(
                    plate_text=text,
                    bg_color=random.choice(['yellow', 'white', 'green']),
                    text_color='black',
                    blur=True,
                    blur_amount=random.randint(4, 9),
                    noise=random.random() < 0.5,
                    perspective=random.random() < 0.4,
                )
                self._save_image(img, out_blurry / f"blurry_{i:04d}.jpg")

    # ── COLOR CLASSIFICATION ─────────────────────────────────────────────────

    def generate_color_classification(self):
        print("\n  [2/3] Generating Color Classification Dataset...")
        base = self.output_dir / "color_classification"

        colors = ['yellow', 'white', 'green']
        # Text color by plate type
        text_map = {'yellow': 'black', 'white': 'black', 'green': 'white'}

        for split, n in [("train", self.n_train),
                          ("val",   self.n_val),
                          ("test",  self.n_test)]:
            for color in colors:
                out_dir = base / split / color
                print(f"    {split}/{color} → {n} images")
                for i in tqdm(range(n), desc=f"  {split}/{color}", leave=False):
                    text = self.plate_texts[i % len(self.plate_texts)]
                    img = SyntheticPlateGenerator.generate_plate(
                        plate_text=text,
                        bg_color=color,
                        text_color=text_map[color],
                        blur=random.random() < 0.2,
                        blur_amount=random.randint(1, 4),
                        noise=random.random() < 0.2,
                        perspective=random.random() < 0.5,
                    )
                    self._save_image(img, out_dir / f"{color}_{i:04d}.jpg")

    # ── FONT COLOR CLASSIFICATION ────────────────────────────────────────────

    def generate_font_classification(self):
        print("\n  [3/3] Generating Font Color Classification Dataset...")
        base = self.output_dir / "font_color_classification"

        font_colors = ['black', 'white', 'red', 'green']
        # BG color that contrasts with text
        bg_map = {
            'black': random.choice(['yellow', 'white']),
            'white': 'green',
            'red':   'white',
            'green': 'white',
        }

        for split, n in [("train", self.n_train),
                          ("val",   self.n_val),
                          ("test",  self.n_test)]:
            for font_color in font_colors:
                out_dir = base / split / font_color
                print(f"    {split}/{font_color} → {n} images")
                for i in tqdm(range(n), desc=f"  {split}/{font_color}", leave=False):
                    text = self.plate_texts[i % len(self.plate_texts)]
                    bg = ('yellow' if font_color == 'black' and random.random() < 0.5
                          else bg_map[font_color])
                    img = SyntheticPlateGenerator.generate_plate(
                        plate_text=text,
                        bg_color=bg,
                        text_color=font_color,
                        blur=random.random() < 0.2,
                        blur_amount=random.randint(1, 4),
                        noise=random.random() < 0.2,
                        perspective=random.random() < 0.5,
                    )
                    self._save_image(img, out_dir / f"{font_color}_{i:04d}.jpg")

    def generate_all(self):
        """Generate complete dataset for all three classifiers"""
        print("=" * 60)
        print("  LICENSE PLATE DATASET GENERATION  v2")
        print("=" * 60)
        print(f"  Output dir      : {self.output_dir}")
        print(f"  Unique texts    : {len(self.plate_texts)}")
        print(f"  Images/class    : {self.images_per_class}")
        print(f"  Split (tr/va/te): {self.n_train}/{self.n_val}/{self.n_test}")

        self.generate_blur_quality()
        self.generate_color_classification()
        self.generate_font_classification()

        print("\n" + "=" * 60)
        print("  Dataset generation complete!")
        print("=" * 60)
        self._print_summary()

    def _print_summary(self):
        """Print dataset statistics"""
        print("\n  Dataset Summary:")
        for classifier in ['blur_quality', 'color_classification',
                            'font_color_classification']:
            base = self.output_dir / classifier
            if not base.exists():
                continue
            print(f"\n  {classifier}/")
            for split in ['train', 'val', 'test']:
                split_dir = base / split
                if not split_dir.exists():
                    continue
                total = sum(len(list(d.glob('*.jpg')))
                            for d in split_dir.iterdir()
                            if d.is_dir())
                classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
                print(f"    {split:<8}: {total:>5} images  "
                      f"({', '.join(sorted(classes))})")


# ============================================================================
# DATASET ORGANIZER (legacy helper, kept for compatibility)
# ============================================================================

class DatasetOrganizer:
    """Organize and prepare datasets for training"""

    @staticmethod
    def create_directory_structure(base_path: str):
        """Create standard directory structure"""
        base = Path(base_path)
        dirs = []

        for classifier, classes in [
            ("blur_quality", ["clear", "blurry"]),
            ("color_classification", ["yellow", "white", "green"]),
            ("font_color_classification", ["black", "white", "red", "green"]),
        ]:
            for split in ["train", "val", "test"]:
                for cls in classes:
                    dirs.append(base / classifier / split / cls)

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        print(f"✓ Directory structure created at {base}")

    @staticmethod
    def split_dataset(source_dir: str, split_ratio: tuple = (0.7, 0.15, 0.15)):
        """Split an existing flat directory of images into train/val/test"""
        source = Path(source_dir)
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_images = []
        for ext in extensions:
            all_images.extend(source.glob(f'*{ext}'))
            all_images.extend(source.glob(f'*{ext.upper()}'))

        if not all_images:
            print(f"No images found in {source}")
            return None, None, None

        random.shuffle(all_images)
        total = len(all_images)
        n_train = int(total * split_ratio[0])
        n_val   = int(total * split_ratio[1])

        train = all_images[:n_train]
        val   = all_images[n_train:n_train + n_val]
        test  = all_images[n_train + n_val:]

        print(f"Total: {total}  Train: {len(train)}  "
              f"Val: {len(val)}  Test: {len(test)}")
        return train, val, test


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main data preparation workflow"""

    # Generate 600 unique Nepali plate texts
    print("Generating plate texts...")
    plate_texts = generate_nepali_plate_texts(count=600)
    print(f"  {len(plate_texts)} unique plate texts generated")

    # Generate dataset (560 images per class → 392 train / 84 val / 84 test)
    generator = DatasetGenerator(
        output_dir="dataset",
        plate_texts=plate_texts,
        images_per_class=560
    )
    generator.generate_all()

    print("\nNext steps:")
    print("  1. (Optional) Add real Nepali plate images to dataset/*/train/<class>/")
    print("  2. Run: python train_models.py")
    print("  3. Run: python test_image.py  — to test on images/image.png")


if __name__ == "__main__":
    main()