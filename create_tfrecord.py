from pathlib import Path
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import logging

from chess.preprocessing import Sample, InvalidAnnotationException

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    export_dir = Path("chess_data") / "labels" / "vott-json-export"

    with (export_dir / "chess-export.json").open("r") as f:
        data = json.load(f)

    with tf.io.TFRecordWriter(str(export_dir / "dataset.tfrecord")) as f:
        for asset in data["assets"].values():
            name = asset["asset"]["name"]
            try:
                with (export_dir / name).open("rb") as img:
                    sample = Sample(img.read(), asset)
            except InvalidAnnotationException as e:
                logger.warning(f"Skipping {name}: {e!s}")
                continue
            f.write(sample.generate_example().SerializeToString())
    print("Done")
