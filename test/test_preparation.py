import tensorflow as tf
import numpy as np
from chess import preparation

def test_reshape_points_shape():
    points = tf.random.normal((17, 4, 2))
    points = preparation.reshape_points(points)
    assert points.shape == (17, 2, 2, 2)

def test_reshape_points_values():
    points = np.array([[
        [0., 0.],
        [1., 1.],
        [1., 0.],
        [0., 1.]
    ]])
    expected = np.array([[
        [
            [0., 0.],
            [1., 0.],
        ], [
            [0., 1.],
            [1., 1.]
        ]
    ]])
    np.testing.assert_equal(preparation.reshape_points(points), expected)

def test_get_intersection():
    a = tf.constant([[0., 0.], [1., 1.]])
    b = tf.constant([[0., 1.], [1., 0.]])
    points = tf.stack([a, b], axis=0)
    points = tf.expand_dims(points, axis=0)
    intersection = preparation.get_intersection(points)
    expected = np.array([[.5, .5]])
    np.testing.assert_almost_equal(intersection, expected)

def test_get_center_point():
    a = tf.constant([[0., 0.], [1., 0.]])
    b = tf.constant([[0., 1.], [1., 1.]])
    points = tf.stack([a, b], axis=0)
    points = tf.expand_dims(points, axis=0)
    center = preparation.get_center_point(points)
    expected = np.array([[.5, .5]])
    np.testing.assert_almost_equal(center, expected)

def test_get_squares():
    points = tf.constant([[
        [
            [0., 0.],
            [1., 0.],
        ], [
            [0., 1.],
            [1., 1.]
        ]
    ]])
    assert preparation.get_squares(points).shape == (1, 8, 8, 2, 2, 2)