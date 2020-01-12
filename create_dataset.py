from preprocessing.board import BoardAnnotation, BoardPicture
import itertools
from imgaug import augmenters as iaa
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import numpy as np

# temporary fix for imgaug
np.random.bit_generator = np.random._bit_generator


DATA_DIR = Path("data")

PIECES = {
    'p': "pawn",
    'r': "rook",
    'n': "knight",
    'b': "bishop",
    'q': "queen",
    'k': "king"
}

with open(os.path.join(DATA_DIR, "via_region_data.json"), "r") as f:
    obj = json.load(f)
    obj = {
        entry["filename"]: (entry["regions"]["0"]["shape_attributes"]["all_points_x"][:4],
                            entry["regions"]["0"]["shape_attributes"]["all_points_y"][:4])
        for entry in obj.values()
    }


def extract_images(file: Path):
    x, y = obj[file.name]
    points = np.array(list(zip(x, y)))
    img = Image.open(file)
    picture = BoardPicture(img, points)
    with open(DATA_DIR / "annotations" / f"{file.stem}.txt", "r") as f:
        description = f.read()
    annotation = BoardAnnotation(description)
    for x in range(8):
        for y in range(4):
            img = picture.crop_square(x, y, 1 + (y + 1) / 8)
            img = Image.fromarray(img)
            piece = annotation.get(x, y)
            if isinstance(piece, bytes):
                piece = piece.decode("utf-8")
            if piece != '':
                piece = ("black" if piece.islower() else "white") + \
                    "_" + PIECES[piece.lower()]
            else:
                piece = "empty"
            yield x, y, piece, img


def extract_images_for_dataset(input_folder: Path, output_folder: Path):
    for file in input_folder.glob("*.jpg"):
        print(file.name)
        for x, y, piece, img in extract_images(file):
            piece_dir = output_folder / piece
            if not piece_dir.exists():
                piece_dir.mkdir(parents=True)
            img.save(os.path.join(
                piece_dir, f"{file.stem}_{x}_{y}{file.suffix}"))


# Validation dataset
print("Processing validation dataset...")
extract_images_for_dataset(DATA_DIR / "images" / "val",
                           DATA_DIR / "pieces" / "val")

# Train dataset
print("Processing train dataset...")
extract_images_for_dataset(DATA_DIR / "images" / "train",
                           DATA_DIR / "pieces" / "train")
print("Augmenting train dataset...")
NUM_IMAGES_PER_CLASS = 500
for folder in (DATA_DIR / "pieces" / "train").iterdir():
    if not folder.is_dir():
        continue
    samples = np.array(list(folder.glob("*.jpg")))
    difference = NUM_IMAGES_PER_CLASS - len(samples)
    if difference < 0:
        print(f"Removing {-difference} images from {folder.name}")
        to_remove = np.random.choice(samples, -difference, replace=False)
        for file in to_remove:
            file.unlink()
    elif difference > 0:
        print(
            f"Producing {difference} more images for {folder.name} (augmenting) ")
        to_augment = np.random.choice(samples, difference, replace=True)
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.PerspectiveTransform(scale=(0.01, 0.15)),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Multiply((0.8, 1.2)),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.1, 0.1)},
                rotate=(-20, 20),
                shear=(-6, 6)
            )
        ])
        augmented = seq(images=[np.array(Image.open(x)) for x in to_augment])
        for file, img in zip(to_augment, augmented):
            for i in itertools.count(start=1):
                output_file = file.parent / \
                    (file.stem + f"_augmented_{i:03}" + file.suffix)
                if not output_file.exists():
                    break
            Image.fromarray(img).save(output_file)
