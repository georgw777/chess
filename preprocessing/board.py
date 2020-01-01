import json
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from util import get_angle_between, get_angle_of_vector, get_center_point, get_intersection


class BoardAnnotation:
    def __init__(self, description: str):
        self.board = np.chararray((4, 8))
        self.board.fill('')
        for y, line in enumerate(reversed(description.split('\n'))):
            for x, piece in enumerate(line):
                self.board[y, x] = piece if piece != ' ' else ''

    def get(self, x: int, y: int) -> str:
        return self.board[y, x]


class BoardPicture:
    def __init__(self, img: Image.Image, vertices: np.ndarray):
        self.img = img
        self.vertices = self._order_points(vertices)
        self.squares = self._get_squares()

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        # order: [[tl, tr], [bl, br]]
        points = points[points[:, 1].argsort()]
        top, bottom = np.split(points, [2])
        top = top[top[:, 0].argsort()]
        bottom = bottom[bottom[:, 0].argsort()]
        return np.stack([top, bottom])

    def _get_polygon_mask(self, polygon: np.ndarray) -> np.ndarray:
        mask = Image.new('L', self.img.size, 0)
        ImageDraw.Draw(mask).polygon(
            polygon.flatten().tolist(), outline=1, fill=1)
        return np.array(mask).astype(np.bool)

    def crop_polygon(self, polygon: np.ndarray) -> np.ndarray:
        mask = self._get_polygon_mask(polygon)
        rows = mask.max(axis=-1) > 0
        cols = mask.max(axis=0) > 0
        ymin, ymax = rows.argmax(), len(rows) - rows[::-1].argmax()
        xmin, xmax = cols.argmax(), len(cols) - cols[::-1].argmax()
        indices = (slice(ymin, ymax), slice(xmin, xmax))
        mask = mask[indices]
        img = np.array(self.img)[indices]
        mask = np.broadcast_to(np.expand_dims(mask, axis=-1), img.shape)
        return np.where(mask, img, 0)

    def highlight_polygon(self, polygon: np.ndarray) -> np.ndarray:
        mask = self._get_polygon_mask(polygon)
        img = np.array(self.img)
        mask = np.broadcast_to(np.expand_dims(mask, axis=-1), img.shape)
        return np.where(mask, img, img // 2)

    @staticmethod
    def _split_square(points: np.ndarray, horizontal_intersection: np.ndarray, vertical_intersection: np.ndarray) -> np.ndarray:
        top_line, bottom_line = points
        left_line, right_line = points.swapaxes(0, 1)
        center = get_center_point(points)
        horizontal_line = (center, horizontal_intersection) if horizontal_intersection is not None else (
            center, center + top_line[1] - top_line[0])
        vertical_line = (center, vertical_intersection) if vertical_intersection is not None else (
            center, center + left_line[1] - left_line[0])
        left_point = get_intersection(left_line, horizontal_line)
        right_point = get_intersection(right_line, horizontal_line)
        top_point = get_intersection(top_line, vertical_line)
        bottom_point = get_intersection(bottom_line, vertical_line)

        return np.array([
            [points[0, 0], top_point, points[0, 1]],
            [left_point, center, right_point],
            [points[1, 0], bottom_point, points[1, 1]]
        ])

    def _get_squares(self) -> np.ndarray:  # shape: [9, 9, 2]
        points = self.vertices
        horizontal_intersection = get_intersection(*points)
        vertical_intersection = get_intersection(*np.swapaxes(points, 0, 1))
        for depth in range(1, 4):
            arr = np.zeros((2 ** depth + 1, 2 ** depth + 1, 2))
            for y in range(points.shape[0] - 1):
                for x in range(points.shape[1] - 1):
                    arr[y*2:y*2+3, x*2:x*2+3] = self._split_square(
                        points[y:y+2, x:x+2], horizontal_intersection, vertical_intersection)
            points = arr
        return points

    @staticmethod
    def _increase_square_height(square: np.ndarray, height_multiplier: float) -> np.ndarray:
        ((tl, tr), (bl, br)) = square
        return np.array([
            [bl + (tl - bl) * height_multiplier, br +
             (tr - br) * height_multiplier],
            [bl, br]
        ])

    def get_coordinates_of_square(self, x: int, y: int, height_multiplier: float = 1.) -> np.ndarray:
        y = 7 - y
        square = self.squares[y:y+2, x:x+2]
        return self._increase_square_height(square, height_multiplier)

    def crop_square(self, x: int, y: int, height_multiplier: float = 1.) -> np.ndarray:
        square = self.get_coordinates_of_square(x, y, height_multiplier)
        square = square.reshape(4, 2)[[0, 1, 3, 2]]
        return self.crop_polygon(square)

    def highlight_square(self, x: int, y: int, height_multiplier: float = 1.) -> np.ndarray:
        square = self.get_coordinates_of_square(x, y, height_multiplier)
        square = square.reshape(4, 2)[[0, 1, 3, 2]]
        return self.highlight_polygon(square)
