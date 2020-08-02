import tensorflow as tf


@tf.function
def reshape_points(points: tf.Tensor) -> tf.Tensor:
    # points: [..., 4, 2]
    batch_dimensions = points.shape[:-2]
    y_sorted = tf.argsort(points[..., 1])
    points = tf.gather(points, y_sorted, batch_dims=len(batch_dimensions))
    points = tf.reshape(points, (*batch_dimensions, 2, 2, 2))
    x_sorted = tf.argsort(points[..., 0])
    points = tf.gather(points, x_sorted, batch_dims=len(batch_dimensions)+1)
    return points


@tf.function
def get_intersection(points: tf.Tensor) -> tf.Tensor:
    # points: [..., 2, 2, 2]
    #tf.debugging.assert_shapes({ points: (..., 2, 2, 2) })
    points = tf.concat(
        (points, tf.ones((*points.shape[:-1], 1), dtype=points.dtype)), axis=-1)
    l1 = tf.linalg.cross(points[..., 0, 0, :], points[..., 0, 1, :])
    l2 = tf.linalg.cross(points[..., 1, 0, :], points[..., 1, 1, :])
    result = tf.linalg.cross(l1, l2)
    xy = result[..., :2]
    z = result[..., 2:]
    return xy / z


@tf.function
def get_center_point(points: tf.Tensor) -> tf.Tensor:
    # points: [..., 2, 2, 2]
    return get_intersection(tf.stack([
        tf.stack([
            points[..., 0, 0, :],
            points[..., 1, 1, :]
        ], axis=-2),
        tf.stack([
            points[..., 0, 1, :],
            points[..., 1, 0, :]
        ], axis=-2)
    ], axis=-3))


@tf.function
def get_board_intersections(points: tf.Tensor) -> tf.Tensor:
    # points: [..., 2, 2, 2]
    batch_dimensions = points.shape[:-3]
    ndim = len(batch_dimensions)
    transposed = tf.transpose(points, tf.concat(
        (tf.range(ndim), tf.constant([1, 0, 2]) + ndim), axis=0))
    return tf.stack((get_intersection(points), get_intersection(transposed)), axis=1)


@tf.function
def split_square(points: tf.Tensor, intersections: tf.Tensor):
    # points: [..., 2, 2, 2]
    top_line, bottom_line = points[..., 0, :, :], points[..., 1, :, :]
    left_line, right_line = points[..., :, 0, :], points[..., :, 1, :]

    horizontal_intersection = intersections[..., 0, :]
    vertical_intersection = intersections[..., 1, :]

    center = get_center_point(points)

    horizontal_line = tf.stack([
        center,
        tf.where(
            tf.math.is_inf(horizontal_intersection),
            center + top_line[..., 1, :] - top_line[..., 0, :],
            horizontal_intersection
        )
    ], axis=-2)
    vertical_line = tf.stack([
        center,
        tf.where(
            tf.math.is_inf(vertical_intersection),
            center + left_line[..., 1, :] - left_line[..., 0, :],
            vertical_intersection
        )
    ], axis=-2)

    left_point = get_intersection(
        tf.stack((left_line, horizontal_line), axis=-3))
    right_point = get_intersection(
        tf.stack((right_line, horizontal_line), axis=-3))
    top_point = get_intersection(tf.stack((top_line, vertical_line), axis=-3))
    bottom_point = get_intersection(
        tf.stack((bottom_line, vertical_line), axis=-3))

    points = tf.reshape(points, (*points.shape[:-3], 4, 2))
    new_points = tf.stack(
        [left_point, right_point, top_point, bottom_point, center], axis=-2)
    all_points = tf.concat([points, new_points], axis=-2)

    TL, TR, BL, BR, LEFT, RIGHT, TOP, BOTTOM, CENTER = list(range(9))

    return tf.gather(all_points, [
        [[[TL, TOP],
          [LEFT, CENTER]],
         [[TOP, TR],
          [CENTER, RIGHT]]],
        [[[LEFT, CENTER],
          [BL, BOTTOM]],
         [[CENTER, RIGHT],
          [BOTTOM, BR]]]
    ], axis=-2)


@tf.function
def get_squares(points: tf.Tensor) -> tf.Tensor:
    intersections = get_board_intersections(points)
    broadcasted_intersections = intersections
    batch_dimensions = points.shape[:-3]
    for i in range(1, 4):
        points = split_square(points, broadcasted_intersections)
        points = tf.reshape(points, (*batch_dimensions, 4**i, 2, 2, 2))
        broadcasted_intersections = tf.broadcast_to(
            intersections, (*batch_dimensions, 4**i, 2, 2))
    points = tf.reshape(points, (*batch_dimensions, *((2,) * 6), 2, 2, 2))
    ndim = tf.rank(points) - 6 - 3
    axis_permutation = tf.concat([
        tf.range(ndim),
        tf.constant([0, 2, 4, 1, 3, 5]) + ndim,
        tf.range(3) + tf.rank(points) - 3
    ], axis=0)
    points = tf.transpose(points, axis_permutation)
    points = tf.reshape(points, (*batch_dimensions, 8, 8, 2, 2, 2))
    return points


@tf.function
def extend_square_heights(squares: tf.Tensor) -> tf.Tensor:
    # squares: [..., 8, 8, 2, 2, 2]
    batch_dimensions = squares.shape[:-5]
    height_multiplier = tf.linspace(2., 1, 9)[1:]
    height_multiplier = tf.broadcast_to(tf.reshape(
        height_multiplier, (*batch_dimensions, 8, 1, 1, 1)), (*batch_dimensions, 8, 8, 1, 1))

    top = squares[..., 1, :, :] + \
        (squares[..., 0, :, :] - squares[..., 1, :, :]) * height_multiplier
    bottom = squares[..., 1, :, :]
    return tf.stack((top, bottom), axis=-3)


@tf.function
def crop_squares(img: tf.Tensor, squares: tf.Tensor, size=(300, 300)) -> tf.Tensor:
    # img: [..., H, W, 3]
    # squares: [..., 8, 8, 2, 2, 2]
    #                h, w, Y, X, coordinates (x, y)
    H, W = img.shape[-3:-1]
    batch_dimensions = squares.shape[:-5]

    squares = tf.reshape(squares, (*squares.shape[:-3], 4, 2))
    x1 = tf.reduce_min(squares[..., 0], axis=-1)
    y1 = tf.reduce_min(squares[..., 1], axis=-1)
    x2 = tf.reduce_max(squares[..., 0], axis=-1)
    y2 = tf.reduce_max(squares[..., 1], axis=-1)

    # Normalize
    x1 /= W - 1
    x2 /= W - 1
    y1 /= H - 1
    y2 /= H - 1

    bounding_boxes = tf.stack((y1, x1, y2, x2), axis=-1)

    bounding_boxes = tf.reshape(bounding_boxes, (-1, 4))
    img = tf.reshape(img, (-1, H, W, 3))
    indices = tf.reshape(tf.broadcast_to(
        tf.range(img.shape[0]), (img.shape[0], 64)), (-1,))

    result = tf.image.crop_and_resize(
        image=img,
        boxes=bounding_boxes,
        box_indices=indices,
        crop_size=size
    )

    # Reshape to original batch dimensions
    result = tf.reshape(result, (*batch_dimensions, 8, 8, *size, 3))

    return result
