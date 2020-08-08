import numpy as np
import logging
import tensorflow as tf
from pathlib import Path
import json

from chess.labels import LABELS

logger = logging.getLogger(__name__)


class InvalidAnnotationException(ValueError):
    pass


class Sample:

    def __init__(self, img: bytes, data: dict):
        self.img = img

        size = data["asset"]["size"]
        self.width, self.height = size["width"], size["height"]
        pieces = []
        corners = None
        for region in data["regions"]:
            if len(region["tags"]) != 1:
                logger.warn(
                    f"Region {region['id']} in {data['asset']['name']} does not have exactly one tag, skipping.")
                continue
            tag = region["tags"][0]
            if region["type"] == "RECTANGLE" and tag in LABELS and tag != "empty":
                box = region["boundingBox"]
                height, width, x1, y1 = box["height"], box["width"], box["left"], box["top"]
                x2 = x1 + width
                y2 = y1 + height
                box = np.array([x1, y1, x2, y2])
                pieces.append((tag, box))
            elif region["type"] == "POLYGON" and tag == "board":
                corners = [(p["x"], p["y"])
                           for p in region["points"]]
            else:
                logger.warning(
                    f"Ignoring region {region['id']} in {data['asset']['name']}.")
        if corners is None:
            raise InvalidAnnotationException(
                f"Board corner not annotated in {data['asset']['name']}.")

        wh = np.array([self.width, self.height])
        self.corners = np.array(corners) / wh
        if len(pieces) == 0:
            self.classes = []
            self.boxes = np.array([])
        else:
            self.classes, self.boxes = zip(*pieces)
            self.boxes = np.array(self.boxes) / np.expand_dims(np.repeat(wh, 2),
                                                               axis=0)

    def generate_example(self):
        labels = {k: i for i, k in enumerate(LABELS)}
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.img])),
                    "corners": tf.train.Feature(float_list=tf.train.FloatList(value=self.corners.flatten().tolist())),
                    "pieces": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(labels[l])
                                                                                    for l in self.classes])),
                    "boxes": tf.train.Feature(float_list=tf.train.FloatList(value=self.boxes.flatten().tolist()))
                }
            )
        )


def create_tf_dataset(export_dir: Path, dataset_file: Path):
    with (export_dir / "chess-export.json").open("r") as f:
        data = json.load(f)

    with tf.io.TFRecordWriter(str(dataset_file)) as f:
        for asset in data["assets"].values():
            name = asset["asset"]["name"]
            try:
                with (export_dir / name).open("rb") as img:
                    sample = Sample(img.read(), asset)
            except InvalidAnnotationException as e:
                logger.warning(f"Skipping {name}: {e!s}")
                continue
            f.write(sample.generate_example().SerializeToString())


if __name__ == "__main__":
    export_dir = Path("chess_data") / "labels" / "vott-json-export"
    create_tf_dataset(export_dir, export_dir / "dataset.tfrecord")
