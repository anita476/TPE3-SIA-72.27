import argparse
import ast
import csv
import sys
from pathlib import Path

import numpy as np

IMG_SIZE = 28
SEED = 1

TRANSLATION_RANGE = 2
GAMMA_RANGE = (0.5, 1.6)
CONTRAST_RANGE = (0.7, 1.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a digits CSV with contrast and translation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--labels", nargs="+", required=True, help="Labels:augmented copies per sample, e.g. 5:3 8:4.",
    )

    args = parser.parse_args()
    args.per_sample_by_label = {
        int(label): int(per_sample)
        for label, per_sample in (
            value.split(":", 1) for value in args.labels
        )
    }
    return args


def translate(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(image)
    h, w = image.shape

    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)

    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def neighbors_3x3(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    padded = np.pad(image, pad_width=1, mode="constant", constant_values=0.0)

    return np.stack([
        padded[i:i + h, j:j + w]
        for i in range(3)
        for j in range(3)
    ])


def soft_blur(image: np.ndarray) -> np.ndarray:
    neigh = neighbors_3x3(image)
    blurred = np.mean(neigh, axis=0)

    return 0.75 * image + 0.25 * blurred


def thicken(image: np.ndarray) -> np.ndarray:
    neigh = neighbors_3x3(image)
    expanded = np.max(neigh, axis=0)

    return np.maximum(image, 0.65 * expanded)


def thin(image: np.ndarray) -> np.ndarray:
    neigh = neighbors_3x3(image)

    support = np.sum(neigh > 0.15, axis=0)

    out = image.copy()

    out[support <= 3] *= 0.55

    return out


def random_contrast(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    factor = float(rng.uniform(*CONTRAST_RANGE))
    out = image.copy()
    foreground = out > 0.0

    if not np.any(foreground):
        return out

    mean = float(np.mean(out[foreground]))
    out[foreground] = (out[foreground] - mean) * factor + mean
    return out


def random_gamma(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gamma = float(rng.uniform(*GAMMA_RANGE))
    return np.clip(image, 0.0, 1.0) ** gamma


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = image.astype(np.float32)

    # 1. Translation
    dx = int(rng.integers(-TRANSLATION_RANGE, TRANSLATION_RANGE + 1))
    dy = int(rng.integers(-TRANSLATION_RANGE, TRANSLATION_RANGE + 1))
    image = translate(image, dx, dy)
    ink_mask = image > 0.0

    # 2. Contrast
    if rng.random() < 0.70:
        image = random_contrast(image, rng)

    # 3. Gamma
    if rng.random() < 0.70:
        image = random_gamma(image, rng)

    # 4. Soft blur 
    if rng.random() < 0.20:
        image = soft_blur(image)

    # 5. Thicken or thin
    r = rng.random()
    if r < 0.20:
        image = thicken(image)
    elif r < 0.35:
        image = thin(image)

    image[~ink_mask] = 0.0
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def image_to_csv_field(image: np.ndarray) -> str:
    return "[" + ", ".join(repr(float(v)) for v in image.reshape(-1)) + "]"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(SEED)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_sample_by_label = args.per_sample_by_label

    n_original = n_augmented = 0
    with input_path.open(newline="", encoding="utf-8") as f_in, \
         output_path.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["label", "image"])

        for row in reader:
            label = int(row["label"])

            image = np.asarray(
                ast.literal_eval(row["image"]),
                dtype=np.float32
            ).reshape(IMG_SIZE, IMG_SIZE)

            writer.writerow([label, image_to_csv_field(image)])
            n_original += 1

            per_sample = per_sample_by_label.get(label, 0)
            if per_sample <= 0:
                continue

            for _ in range(per_sample):
                writer.writerow([label, image_to_csv_field(augment(image, rng))])
                n_augmented += 1

    print(f"originals: {n_original}", file=sys.stderr)
    print(
        "labels: "
        + ", ".join(
            f"{label}:{count}" for label, count in sorted(per_sample_by_label.items())
        ),
        file=sys.stderr,
    )
    print(f"augmented: {n_augmented}", file=sys.stderr)
    print(f"total:     {n_original + n_augmented}", file=sys.stderr)


if __name__ == "__main__":
    main()
