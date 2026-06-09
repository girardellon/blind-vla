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


DEGRADATIONS = {
    "type0": lambda x: x,
    "type1": lambda x: add_gaussian_noise(x, sigma=10),
    "type2": lambda x: add_blur(
        add_gaussian_noise(x, sigma=20),
        kernel=5,
    ),
    "type3": lambda x: add_blur(x, kernel=11),
    "type4": lambda x: add_occlusion(x, frac=0.30),
    "type5": lambda x: add_occlusion(x, frac=0.60),
    "type6": blackout,
}


def build_dataset(input_root, output_root):

    input_root = Path(input_root)
    output_root = Path(output_root)

    object_dirs = [d for d in input_root.iterdir() if d.is_dir()]

    for obj_dir in object_dirs:

        object_name = obj_dir.name

        out_obj_dir = output_root / object_name
        out_obj_dir.mkdir(parents=True, exist_ok=True)

        sample_dirs = sorted(
            [d for d in obj_dir.iterdir() if d.is_dir()]
        )

        for sample_dir in tqdm(
            sample_dirs,
            desc=f"{object_name}"
        ):

            img_path = sample_dir / "external.png"
        
            if not img_path.exists():
                    continue

            img = cv2.imread(str(img_path))

            sample_name = sample_dir.name

            for degr_name, degr_fn in DEGRADATIONS.items():

                degraded = degr_fn(img)

                save_name = (
                    f"{object_name}_"
                    f"{sample_name}_"
                    f"{degr_name}.png"
                )

                cv2.imwrite(
                    str(out_obj_dir / save_name),
                    degraded
                )


if __name__ == "__main__":

    build_dataset(
        "data_collection/dataset_version1",
        "external_images_corrupted"
    )