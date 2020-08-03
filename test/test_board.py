import tensorflow as tf
import numpy as np

from chess.board import BoardAnnotation
from chess.labels import LABELS


def test_encode():
    annotation = "\n".join(["   p    ",
                            "        ",
                            "ppp pppp",
                            "rnbqkbnr"])
    board = np.array([[0, 0, 0, 4, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [4, 4, 4, 0, 4, 4, 4, 4],
                      [6, 1, 2, 5, 3, 2, 1, 6],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]])
    annotation = tf.constant(annotation)
    encoded = BoardAnnotation.encode(annotation)
    np.testing.assert_equal(tf.argmax(encoded, axis=-1), board)


def test_decode():
    annotation = "\n".join(["   p    ",
                            "        ",
                            "ppp pppp",
                            "rnbqkbnr",
                            "        ",
                            "        ",
                            "        ",
                            "        "]) + "\n"
    board = np.array([[0, 0, 0, 4, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [4, 4, 4, 0, 4, 4, 4, 4],
                      [6, 1, 2, 5, 3, 2, 1, 6],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]])
    board_ohe = tf.one_hot(board, depth=len(LABELS))
    decoded = BoardAnnotation.decode(board_ohe)
    assert decoded.shape == (1,)
    assert decoded.numpy().item().decode("utf-8") == annotation
