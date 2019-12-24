import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from preprocessing.board import BoardAnnotation, BoardPicture

DATA_DIR = "data"

with open(os.path.join(DATA_DIR, "via_region_data.json"), "r") as f:
    obj = json.load(f)
for entry in reversed(list(obj.values())):
    attributes = entry["regions"]["0"]["shape_attributes"]
    points = np.array(
        list(zip(attributes["all_points_x"][0:4], attributes["all_points_y"][0:4])))
    filename = entry["filename"]
    print(filename)
    name, extension = os.path.splitext(os.path.basename(filename))

    img = Image.open(os.path.join(DATA_DIR, "images", filename))
    picture = BoardPicture(img, points)
    with open(os.path.join(DATA_DIR, "annotations", f"{name}.txt"), "r") as f:
        description = f.read()
    annotation = BoardAnnotation(description)
    for x in range(8):
        for y in range(4):
            img = picture.crop_square(x, y, 1 + (y + 1) / 8)
            img = Image.fromarray(img)
            piece = annotation.get(x, y)
            piece = piece.decode("utf-8") if piece != '' else 'empty'
            folder = os.path.join(DATA_DIR, "pieces", piece)
            if not os.path.exists(folder):
                os.makedirs(folder)
            img.save(os.path.join(folder, f"{name}_{x}_{y}{extension}"))
