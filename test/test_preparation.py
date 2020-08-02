import tensorflow as tf
from chess.preparation import reshape_points

def test_reshape_points():
    points = tf.random.normal((17, 4, 2))
    points = reshape_points(points)
    assert points.shape == (17, 2, 2, 2)