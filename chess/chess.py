import tensorflow as tf
import numpy as np
import typing

from .labels import LABELS
from .preparation import reshape_points, get_squares, extend_square_heights, crop_squares
from .board import BoardAnnotation


class ChessModel:
    def __init__(self, image_size: typing.Tuple[int, int] = (300, 300)):
        self.image_size = image_size

    def create_model(self):
        self.inception_model = tf.keras.applications.InceptionV3(
            input_shape=self.image_size + (3,),
            include_top=False,
            weights="imagenet"
        )

        self.model = tf.keras.Sequential([
            self.inception_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(LABELS), activation="softmax")
        ])
        self.model.compile(optimizer="adam",
                           loss="categorical_crossentropy",
                           metrics=["categorical_accuracy"])

    def save(self, file: str):
        self.model.save(file)

    def load(self, file: str):
        self.model = tf.keras.models.load_model(file)

    def fit(self, train: tf.data.Dataset, test: tf.data.Dataset, learning_rate: float, epochs: int, steps_per_epoch: int, validation_steps: int, fine_tune: bool = False) -> tf.keras.callbacks.History:
        self.inception_model.trainable = fine_tune
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                           loss="categorical_crossentropy",
                           metrics=["categorical_accuracy"])
        return self.model.fit(train,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=test,
                              validation_steps=validation_steps)

    def predict(self, imgs: tf.Tensor, points: tf.Tensor) -> tf.Tensor:
        # imgs: [..., H, W, 3]
        # points: [..., 4, 2]

        batch_dimensions = imgs.shape[:-3]
        assert len(batch_dimensions) > 0

        points = tf.cast(points, tf.float32)
        points = reshape_points(points)
        squares = get_squares(points)
        squares = extend_square_heights(squares)
        samples = crop_squares(imgs, squares, size=self.image_size)

        # Make sure we only have 1 batch dimension
        samples = tf.reshape(samples, (-1, *self.image_size, 3))

        # Preprocess
        samples = tf.keras.applications.inception_v3.preprocess_input(samples)

        # Predict
        predictions = self.model.predict(samples)
        predictions = tf.reshape(
            predictions, (*batch_dimensions, 8, 8, len(LABELS)))

        return predictions
