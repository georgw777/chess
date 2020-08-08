from pathlib import Path
import tensorflow as tf

from preprocessing.sample import create_tf_dataset


def test_export():
    export_dir = Path("test/assets/sample_export")
    create_tf_dataset(export_dir, export_dir / "tmp.tfrecord")
    assert len(list(tf.data.TFRecordDataset(
        [str(export_dir / "tmp.tfrecord")]))) == 2
