from pathlib import Path
import random
import cv2
import numpy as np
from tqdm import tqdm


def add_gaussian_noise(img, sigma):
    noise = np.random.normal(
        0,
        sigma,
        img.shape
    ).astype(np.float32)

    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_blur(img, kernel):
    return cv2.GaussianBlur(
        img,
        (kernel, kernel),
        0
    )


def add_occlusion(img, frac):
    h, w = img.shape[:2]

    occ_w = int(w * frac)
    occ_h = int(h * frac)

    x = random.randint(0, w - occ_w)
    y = random.randint(0, h - occ_h)

    out = img.copy()
    out[y:y+occ_h, x:x+occ_w] = 0

    return out


def blackout(img):
    return np.zeros_like(img)


def apply_curriculum(img):

    p = random.random()

    # 10% clean
    if p < 0.10:
        return img

    # 20% light noise
    elif p < 0.30:
        return add_gaussian_noise(img, sigma=10)

    # 20% noise + blur
    elif p < 0.50:
        img = add_gaussian_noise(img, sigma=20)
        return add_blur(img, kernel=5)

    # 20% strong blur
    elif p < 0.70:
        return add_blur(img, kernel=11)

    # 15% occlusion
    elif p < 0.85:
        return add_occlusion(img, frac=0.30)

    # 10% severe occlusion
    elif p < 0.95:
        return add_occlusion(img, frac=0.60)

    # 5% complete failure
    else:
        return blackout(img)


def build_dataset(input_dir, output_dir):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = list(input_dir.glob("*"))

    for f in tqdm(image_files):

        img = cv2.imread(str(f))

        if img is None:
            continue

        corrupted = apply_curriculum(img)

        cv2.imwrite(
            str(output_dir / f.name),
            corrupted
        )


if __name__ == "__main__":

    build_dataset(
        "external_images",
        "external_images_corrupted"
    )