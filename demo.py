import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pathlib
import tensorflow as tf
import random

from chess.chess import ChessModel
from chess.board import BoardAnnotation

model = ChessModel()
model.load("model.h5")

IMAGE_DIR = pathlib.Path("data") / "images"

files = list(IMAGE_DIR.glob("**/*.jpg"))


def print_board(board: str):
    for i, row in enumerate(board.split("\n"), 1):
        if i == 9:
            break
        row = " ".join(row)
        print(f"{chr(ord('H') - i + 1)} | {row}")
    print("  +" + "--" * 8)
    print("    " + " ".join(str(x) for x in range(1, 9)))


while True:
    filename = random.choice(files)

    print(f"File: {filename}")

    img = Image.open(filename)
    plt.imshow(img)
    plt.show(block=False)

    points = np.array(plt.ginput(4, timeout=0))
    if points.shape != (4, 2):
        continue

    img = np.expand_dims(img, axis=0)
    points = np.expand_dims(points, axis=0)

    prediction = model.predict(img, points)
    print_board(BoardAnnotation.decode(prediction)[0].numpy().decode("utf-8"))

    if input("Press enter to continue or 'q' to quit: ") == "q":
        break
