"""This module bundles methods to add online data augmentation to a TensorFlow graph.

"Online" means that this got nothing to do with any data augmentation which may have been done before caching the
dataset.
"""
import math

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import config as cf
from utils import log


def add_augmentation_operations(tf_imgs_placeholder, tf_labels_placeholder):
    """Add TensorFlow operations to the given placeholders and return two tensors which hold the final result.

    :param tf_imgs_placeholder: The original placeholder for the image input.
    :param tf_labels_placeholder: The original placeholder for the label input.
    :return: The new placeholders for images and labels.
    """
    # Randomly distort the colors. There are 1 or 4 ways to do it.
    if cf.get("dao_color_distortion"):
        log.log("adding color distortions")

        # the TensorFlow color distortions require an image in the [0, 1] range, but our data is currently in [-1, 1].
        tf_imgs_placeholder = tf.add(tf_imgs_placeholder, 1)
        tf_imgs_placeholder = tf.divide(tf_imgs_placeholder, 2.0)

        num_distort_cases = 1 if cf.get("dao_color_distortion_fast_mode") else 4
        tf_imgs_placeholder = _apply_with_random_selector(
            tf_imgs_placeholder,
            lambda x, ordering: _add_augmentation_color_distortions(x, ordering, cf.get("dao_color_distortion_fast_mode")),
            num_cases=num_distort_cases)

        # we need to change the value range from [0, 1] to [-1, 1] as done in the original inception code, too.
        # otherwise the weighted cross entropy won't converge smoothly in the cascade's very last net.
        # note that we have been using [-1, 1] before applying any color distortions, if cf.get("standardization"). so
        # it makes absolutely sense to revert the changes caused by the color distortions. it's still a bit weird
        # though, because the original inception code is doing this, too, although the documentation states that
        # _add_augmentation_color_distortions() accepts inputs in [0, 1].
        tf_imgs_placeholder = tf.subtract(tf_imgs_placeholder, 0.5)
        tf_imgs_placeholder = tf.multiply(tf_imgs_placeholder, 2.0)

    # the following params refer mainly to the background class. so some of them will be replaced in the
    # _augment function
    # e.g. foreground won't be flipped, if not cf.get("dao_allow_vertical_flipping_of_foreground")
    tf_imgs_placeholder, tf_labels_placeholder = _add_augmentation_operations_basic(tf_imgs_placeholder,
                                                                                    tf_labels_placeholder,
                                                                                    horizontal_flip=cf.get("dao_horizontal_flip"),
                                                                                    vertical_flip=cf.get("dao_vertical_flip"),
                                                                                    rotate=cf.get("dao_max_rotation_angle"),
                                                                                    crop_probability=cf.get("dao_crop_probability"),
                                                                                    crop_min_percent=cf.get("dao_crop_min_percent"))

    return tf_imgs_placeholder, tf_labels_placeholder


def _add_augmentation_operations_basic(images, labels,
                                       resize=None,  # (width, height) tuple or None
                                       horizontal_flip=False,
                                       vertical_flip=False,
                                       rotate=0,  # Maximum rotation angle in degrees
                                       crop_probability=0,  # How often we do crops
                                       crop_min_percent=0.6,  # Minimum linear dimension of a crop
                                       crop_max_percent=1.):  # Maximum linear dimension of a crop
    """Add a combination of resizing, flipping, rotating and cropping.

    This method is based on the one from: https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
    :param images:
    :param labels:
    :param resize:
    :param horizontal_flip:
    :param vertical_flip:
    :param rotate:
    :param crop_probability:
    :param crop_min_percent:
    :param crop_max_percent:
    :return:
    """
    if resize is not None:
        images = tf.image.resize_bilinear(images, resize)

    # My experiments showed that casting on GPU improves training performance
    # if images.dtype != tf.float32:
    #     images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    #     images = tf.subtract(images, 0.5)
    #     images = tf.multiply(images, 2.0)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # a tensor with shape [batch_size] stating whether a specific sample belongs to the foreground class or not
        # (this assumes label.KEY_FOREGROUND == 1 and label.KEY_BACKGROUND == 0)
        is_foreground = tf.cast(labels, tf.bool)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)

            # vertical flipping may be disabled for foreground samples
            # if that's the case, we will just replace the coins for all foreground samples
            if not cf.get("dao_allow_vertical_flipping_of_foreground"):
                coin = tf.logical_and(coin, tf.logical_not(is_foreground))

            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        # rotating
        # TODO instead of using the already cropped window, we could rotate the meta data only and use this
        # meta data to re-crop the new window directly from the larger original image. so we could prevent almost
        # any black bars which are caused by missing image data. of course, windows which are very close to the
        # image border will still contain black bars.
        if cf.get("dao_rotation_mode") != cf.DAO_ROTATION_MODE_DISABLED:

            # we got two different rotation modes
            rotation_90_enabled = cf.get("dao_rotation_mode") == cf.DAO_ROTATION_MODE_90
            rotation_continuous_enabled = cf.get("dao_rotation_mode") == cf.DAO_ROTATION_MODE_CONTINUOUS and rotate > 0

            if rotation_90_enabled or rotation_continuous_enabled:

                if rotation_90_enabled:
                    k_all = tf.random_uniform([batch_size], minval=0, maxval=4, dtype=tf.int32)  # maxval is exclusive
                    comparison1 = tf.equal(k_all, tf.constant(1))
                    comparison2 = tf.equal(k_all, tf.constant(2))  # foreground shouldn't be rotated by 180°
                    comparison3 = tf.equal(k_all, tf.constant(3))
                    # k_all = tf.where(tf.logical_and(is_foreground, comparison2), tf.zeros_like(k_all), k_all)
                    k_all = tf.where(is_foreground, tf.zeros_like(k_all), k_all)

                    degrees = tf.cast(k_all, tf.float32)
                    zeros = tf.zeros_like(k_all, dtype=tf.float32)
                    ones = zeros + 90.0
                    twos = zeros + 180.0
                    threes = zeros + 270.0
                    # degrees = tf.where(comparison0, zeros, degrees)  # 0 remains 0
                    degrees = tf.where(comparison1, ones, degrees)
                    degrees = tf.where(comparison2, twos, degrees)
                    degrees = tf.where(comparison3, threes, degrees)
                    angles = degrees / 180 * math.pi
                elif rotation_continuous_enabled:
                    # the given rotation angle is mainly used for the background samples
                    angle_rad_base = rotate / 180 * math.pi
                    angles_base = tf.random_uniform([batch_size], -angle_rad_base, angle_rad_base)

                    # we may want to use another angle for foreground samples
                    if cf.get("dao_max_foreground_rotation_angle") is not None:
                        angle_rad_foreground = cf.get("dao_max_foreground_rotation_angle") / 180 * math.pi
                        angles_foreground = tf.random_uniform([batch_size], -angle_rad_foreground, angle_rad_foreground)
                        angles = tf.where(is_foreground,
                                          angles_foreground,
                                          angles_base)
                    else:
                        # using the same angle for foreground and background
                        angles = angles_base

                transforms.append(
                    tf.contrib.image.angles_to_projective_transforms(
                        angles, height, width))

        # cropping
        if crop_probability > 0:
            crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                         crop_max_percent)
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
            crop_transform = tf.stack([
                crop_pct,
                tf.zeros([batch_size]), top,
                tf.zeros([batch_size]), crop_pct, left,
                tf.zeros([batch_size]),
                tf.zeros([batch_size])
            ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

    return images, labels


def _apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    This method has been taken from:
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

    Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

    Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def _add_augmentation_color_distortions(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    This method has been taken from:
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

    Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
    Returns:
    3-D Tensor color-distorted image on range [0, 1]
    Raises:
    ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)
