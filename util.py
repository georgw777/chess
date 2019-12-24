import numpy as np


def get_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    points = np.concatenate((a, b))
    points = np.hstack((points, np.ones((4, 1))))
    l1 = np.cross(*points[:2])
    l2 = np.cross(*points[2:])
    x, y, z = np.cross(l1, l2)
    return (x/z, y/z)


def get_angle_of_vector(v: np.ndarray) -> float:
    x, y = v
    return np.arctan(y / x)


def get_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a1, a2 = a
    b1, b2 = b
    return get_angle_of_vector(b2 - b1) - get_angle_of_vector(a2 - a1)


def get_center_point(points: np.ndarray) -> np.ndarray:
    return get_intersection(points[[0, 1], [0, 1]], points[[0, 1], [1, 0]])
