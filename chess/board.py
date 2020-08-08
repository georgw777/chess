import tensorflow as tf

from .labels import LABELS

labels = tf.constant(list(LABELS.keys()))
board_chars = tf.constant(list(map(ord, LABELS.values())))
board_mask = tf.reshape(board_chars, (1, 1, *board_chars.shape))


class BoardAnnotation:

    @staticmethod
    @tf.function
    def encode(description: tf.Tensor) -> tf.Tensor:
        board = tf.strings.split(description, sep="\n")
        board = tf.strings.unicode_decode(board, "UTF-8").to_tensor()
        board_shape = tf.shape(board)
        board = tf.concat([board, tf.fill((8 - board_shape[0], board_shape[1]), board_chars[0])],
                          axis=0)
        board_shape = tf.shape(board)
        board = tf.concat([board, tf.fill((board_shape[0], 8 - board_shape[1]), board_chars[0])],
                          axis=1)
        board = board_mask == tf.expand_dims(board, axis=-1)
        return board

    @staticmethod
    @tf.function
    def decode(board: tf.Tensor) -> tf.Tensor:
        # board: [..., 8, 8, len(LABELS)]
        batch_dimensions = board.shape[:-3]
        description = tf.argmax(board, axis=-1)
        description = tf.cast(description, tf.int32)
        description = tf.gather(board_chars, description)
        description = tf.concat((
            description,
            tf.broadcast_to(tf.constant([[ord('\n')]]), (*batch_dimensions, 8, 1))), axis=-1)
        description = tf.reshape(description, (-1, 64 + 8))
        description = tf.strings.unicode_encode(description, "UTF-8")
        return description
