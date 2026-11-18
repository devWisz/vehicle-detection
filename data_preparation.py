import cv2
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

VEHICLE_TYPES = {
    0: {"name": "Public Vehicle",     "bg": "white", "font": "black"},
    1: {"name": "Private Vehicle",    "bg": "red",   "font": "white"},
    2: {"name": "Government Vehicle", "bg": "white", "font": "red"},
    3: {"name": "Tourist Vehicle",    "bg": "green", "font": "white"},
}

PLATE_BG_COLORS_RGB = {
    "white": (245, 245, 245),
    "red":   (200, 20,  20),
    "green": (0,   120, 50),
}

FONT_COLORS_RGB = {
    "black": (10,  10,  10),
    "white": (245, 245, 245),
    "red":   (200, 20,  20),
}

BG_CLASS_IDS   = {"white": 0, "red": 1, "green": 2}
FONT_CLASS_IDS = {"black": 0, "white": 1, "red": 2}

DISTRICTS = ["BA", "KA", "PA", "JA", "GA", "LA", "MA",
             "NA", "TA", "DA", "SA", "RA", "HA", "CHA"]

# Devanagari districts used on Nepali plates
DEV_DISTRICTS = ["ग", "ब", "प", "ज", "का", "ल", "म", "न", "त", "द", "स", "र"]
DEV_DIGITS    = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]
DEV_LETTERS   = ["क", "ख", "ग", "घ", "च", "छ", "ज", "झ", "त", "थ", "द",
                 "ध", "न", "प", "ब", "भ", "म", "य", "र", "ल", "व", "श"]
LETTERS   = list("ABCDEFGHJKLMNPQRSTUVWXYZ")


def _dev_num(n):
    """Convert integer to Devanagari numeral string."""
    return "".join(DEV_DIGITS[int(d)] for d in str(n))


def generate_plate_texts(count=900):
    plates = set()
    while len(plates) < count:
        if len(plates) % 3 == 0:
            # Latin style: BA 4 AA 4777
            dist = random.choice(DISTRICTS)
            n1   = random.randint(1, 9)
            let  = random.choice(LETTERS) + random.choice(LETTERS)
            n2   = random.randint(1000, 9999)
            plates.add(f"{dist} {n1} {let} {n2}")
        else:
            # Devanagari style: ग.१ प ४७७७
            dist = random.choice(DEV_DISTRICTS)
            n1   = DEV_DIGITS[random.randint(1, 9)]
            let  = random.choice(DEV_LETTERS) + random.choice(DEV_LETTERS)
            n2   = _dev_num(random.randint(1000, 9999))
            plates.add(f"{dist}.{n1} {let} {n2}")
    return list(plates)


def _get_font(size=50):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _motion_kernel(size, angle_deg):
    kernel = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    angle  = np.deg2rad(angle_deg)
    for i in range(size):
        x = int(cx + (i - cx) * np.cos(angle))
        y = int(cy + (i - cx) * np.sin(angle))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    s = kernel.sum()
    return kernel / s if s > 0 else kernel


def apply_blur(img, blur_type="gaussian", strength=5):
    k = 2 * strength + 1
    if blur_type == "gaussian":
        return cv2.GaussianBlur(img, (k, k), strength * 0.6)
    elif blur_type == "motion_h":
        ker = np.zeros((k, k), dtype=np.float32)
        ker[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, ker)
    elif blur_type == "motion_v":
        ker = np.zeros((k, k), dtype=np.float32)
        ker[:, k // 2] = 1.0 / k
        return cv2.filter2D(img, -1, ker)
    elif blur_type == "motion_diag":
        angle = random.choice([30, 45, 60, 120, 135, 150])
        ker   = _motion_kernel(k, angle)
        return cv2.filter2D(img, -1, ker)
    elif blur_type == "defocus":
        ker  = np.zeros((k, k), dtype=np.float32)
        cv2.circle(ker, (k // 2, k // 2), strength, 1, -1)
        s = ker.sum()
        ker = ker / s if s > 0 else ker
        return cv2.filter2D(img, -1, ker)
    elif blur_type == "combined":
        # Gaussian then motion — simulates camera shake + defocus
        tmp = cv2.GaussianBlur(img, (k, k), strength * 0.4)
        ker = np.zeros((k, k), dtype=np.float32)
        ker[k // 2, :] = 1.0 / k
        return cv2.filter2D(tmp, -1, ker)
    return img


BLUR_TYPES = ["gaussian", "motion_h", "motion_v", "motion_diag", "defocus", "combined"]


def render_plate(text, bg_color, font_color,
                 blur=False, blur_strength=0,
                 noise=False, perspective=False):
    W, H    = 420, 130
    bg_rgb  = PLATE_BG_COLORS_RGB[bg_color]
    txt_rgb = FONT_COLORS_RGB[font_color]

    img  = Image.new("RGB", (W, H), bg_rgb)
    draw = ImageDraw.Draw(img)

    border = 6
    draw.rectangle([border // 2, border // 2, W - border // 2, H - border // 2],
                   outline=(0, 0, 0), width=border)

    for cx, cy in [(14, 14), (W - 14, 14), (14, H - 14), (W - 14, H - 14)]:
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5],
                     fill=(160, 160, 160), outline=(60, 60, 60))
        draw.line([cx - 2, cy, cx + 2, cy], fill=(40, 40, 40), width=2)

    province  = random.choice(["BAGMATI", "GANDAKI", "LUMBINI",
                                "KARNALI", "MADHESH", "KOSHI"])
    top_font  = _get_font(16)
    try:
        bb = draw.textbbox((0, 0), province, font=top_font)
        tw = bb[2] - bb[0]
    except Exception:
        tw = len(province) * 9
    draw.text(((W - tw) // 2, 7), province, fill=txt_rgb, font=top_font)

    main_font = _get_font(50)
    try:
        bb = draw.textbbox((0, 0), text, font=main_font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        tw, th = len(text) * 28, 50
    tx = (W - tw) // 2
    ty = H // 2 - th // 2 + 10
    draw.text((tx + 2, ty + 2), text, fill=(0, 0, 0, 80), font=main_font)
    draw.text((tx, ty), text, fill=txt_rgb, font=main_font)

    out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Colour jitter
    if random.random() < 0.35:
        out = cv2.convertScaleAbs(out,
                                   alpha=random.uniform(0.75, 1.3),
                                   beta=random.randint(-20, 20))

    # Salt-and-pepper noise
    if noise:
        n = int(out.size * random.uniform(0.01, 0.05))
        coords = [np.random.randint(0, d - 1, n) for d in out.shape[:2]]
        out[coords[0], coords[1]] = 255 if random.random() < 0.5 else 0

    # Blur — choose type randomly when not specified
    if blur and blur_strength > 0:
        btype = random.choice(BLUR_TYPES)
        out   = apply_blur(out, btype, blur_strength)

    # Perspective warp
    if perspective:
        h, w = out.shape[:2]
        off   = random.randint(4, 20)
        src   = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst   = np.float32([
            [random.randint(0, off), random.randint(0, off)],
            [w - random.randint(0, off), random.randint(0, off)],
            [random.randint(0, off), h - random.randint(0, off)],
            [w - random.randint(0, off), h - random.randint(0, off)],
        ])
        M   = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, M, (w, h))

    return out


def _save(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def generate_dataset(base="dataset", n_per_class=700):
    texts = generate_plate_texts(900)
    n_tr  = int(n_per_class * 0.70)
    n_va  = int(n_per_class * 0.15)
    n_te  = n_per_class - n_tr - n_va
    splits = [("train", n_tr), ("val", n_va), ("test", n_te)]

    print(f"Generating dataset → {base}/")
    print(f"  {n_per_class} images/class  (train={n_tr} val={n_va} test={n_te})\n")

    # ── [1/3] Blur quality ──────────────────────────────────────────────────
    print("[1/3] Blur Quality  (6 blur types, strength 3–9)")
    for split, n in splits:
        for label in ["clear", "blurry"]:
            out = Path(base) / "blur_quality" / split / label
            for i in range(n):
                vt = random.choice(list(VEHICLE_TYPES.values()))
                if label == "blurry":
                    # Vary strength: mild (3-4), medium (5-6), heavy (7-9)
                    strength = random.choices([3, 4, 5, 6, 7, 8, 9],
                                              weights=[1, 2, 3, 3, 3, 2, 1])[0]
                    img = render_plate(
                        texts[i % len(texts)],
                        bg_color=vt["bg"], font_color=vt["font"],
                        blur=True, blur_strength=strength,
                        noise=random.random() < 0.5,
                        perspective=random.random() < 0.4,
                    )
                else:
                    img = render_plate(
                        texts[i % len(texts)],
                        bg_color=vt["bg"], font_color=vt["font"],
                        blur=False,
                        noise=random.random() < 0.25,
                        perspective=random.random() < 0.4,
                    )
                _save(img, out / f"{label}_{i:04d}.jpg")
        print(f"  {split} done")

    # ── [2/3] BG colour ─────────────────────────────────────────────────────
    print("[2/3] Plate Background Color")
    for split, n in splits:
        for bg_color in ["white", "red", "green"]:
            out  = Path(base) / "bg_color" / split / bg_color
            font = {"white": "black", "red": "white", "green": "white"}[bg_color]
            for i in range(n):
                # 20% chance of mild blur (strength 1-3) to teach robustness
                do_blur = random.random() < 0.20
                img = render_plate(
                    texts[i % len(texts)],
                    bg_color=bg_color, font_color=font,
                    blur=do_blur,
                    blur_strength=random.randint(1, 3) if do_blur else 0,
                    noise=random.random() < 0.2,
                    perspective=random.random() < 0.5,
                )
                _save(img, out / f"{bg_color}_{i:04d}.jpg")
        print(f"  {split} done")

    # ── [3/3] Font colour ───────────────────────────────────────────────────
    print("[3/3] Font Color")
    for split, n in splits:
        for font_color in ["black", "white", "red"]:
            out = Path(base) / "font_color" / split / font_color
            # Consistent BG so the classifier focuses on font, not BG
            bg_map = {"black": "white", "white": "red", "red": "white"}
            bg     = bg_map[font_color]
            for i in range(n):
                do_blur = random.random() < 0.20
                img = render_plate(
                    texts[i % len(texts)],
                    bg_color=bg, font_color=font_color,
                    blur=do_blur,
                    blur_strength=random.randint(1, 3) if do_blur else 0,
                    noise=random.random() < 0.2,
                    perspective=random.random() < 0.5,
                )
                _save(img, out / f"{font_color}_{i:04d}.jpg")
        print(f"  {split} done")

    print("\nDataset Summary:")
    for d in ["blur_quality", "bg_color", "font_color"]:
        dp = Path(base) / d
        if not dp.exists():
            continue
        print(f"  {d}/")
        for sp in ["train", "val", "test"]:
            sdir = dp / sp
            if not sdir.exists():
                continue
            total   = sum(len(list(c.glob("*.jpg"))) for c in sdir.iterdir()
                          if c.is_dir())
            classes = sorted(c.name for c in sdir.iterdir() if c.is_dir())
            print(f"    {sp:<6}: {total:>5} images  ({', '.join(classes)})")


if __name__ == "__main__":
    generate_dataset()
