import numpy as np
import tensorflow as tf

from .utils import matrices_to_flat_transforms, tf_transform, gaussian_blur, enforce_4D


def tf_radians(degrees):
    return degrees * (np.pi / 180.0)


@tf.function
def rotation_matrix(center_x, center_y, rotation):
    """Get the 3x3 rotation transformation matrix

    Parameters
    ----------
    center: (int, int)
        The center point of the rotation
    rotation: int
        The amount of rotation in degrees, positive is clock-wise
    """
    alpha = tf.math.cos(tf_radians(rotation))
    beta = tf.math.sin(tf_radians(rotation))
    return tf.convert_to_tensor([[alpha, beta, (1 - alpha) * center_x - beta * center_y],
                                 [-beta, alpha, beta * center_y + (1 - alpha) * center_y],
                                 [0, 0, 1]])


@tf.function
def shear_matrix(shear):
    """Get the 3x3 shear transformation matrix

    Parameters
    ----------
    shear: int
        The amount of shear in degrees

    NOTE
    ----
    The shear holds the top of the image stationary and shears the rest of the image left/right
    """
    shear = tf_radians(shear)
    return tf.convert_to_tensor([[1, -1 * tf.math.sin(shear), 0],
                                 [0, tf.math.cos(shear), 0],
                                 [0, 0, 1]])


@tf.function
def shift_matrix(x_shift=0, y_shift=0):
    """Get the 3x3 translation transformation matrix

    Parameters
    ----------
    x_shift: int
        The number of pixels to shift the image towards the left (the default is 0)
    y_shift: int
        The number of pixels to shift the image upwards (the default is 0)
    """
    # transform = np.eye(3, dtype=np.float32)
    # transform[0, 2] = x_shift
    # transform[1, 2] = y_shift
    # return transform
    return tf.convert_to_tensor([[1., 0., x_shift],
                                 [0., 1., y_shift],
                                 [0., 0., 1.]])


@tf.function
def scaling_matrix(height=0, width=0, width_scale=1.0, height_scale=1.0):
    """Get the 3x3 scaling transformation matrix

    Parameters
    ----------
    height: int
        The height of an input image (Set this to use a centered scaling) (the default is 0)
    width: int
        The width of an input image (set this to use a centered scaling) (the default is 0)
    width_scale: float
        The width scaling factor (1.5 = 1.5x wider image) (the default is 1.0)
    height_scale: float
        The height scaling factor (1.5 = 1.5x taller image) (the default is 1.0)
    """
    width_scale = 1.0 / width_scale
    height_scale = 1.0 / height_scale
    x_shift = (width * 0.5) * (1 - width_scale)
    y_shift = (height * 0.5) * (1 - height_scale)
    return tf.convert_to_tensor([[width_scale, 0, x_shift],
                                 [0, height_scale, y_shift],
                                 [0, 0, 1]])


@tf.function
def flip_matrix(height, width, v_flip=False, h_flip=True):
    """Get the 3x3 flip transformation matrix

    Parameters
    ----------
    height: int
        The height of the input image
    width: int
        The width of the input image
    v_flip: bool
        Whether to apply a vertical flip (the default is False)
    h_flip: bool
        Whether to apply a horizontal flip (the default is True)

    Note
    ----
    Height and width are required, otherwise the image will be flipped along the axis and remain out of frame
    """
    # transform = np.eye(3)
    # if h_flip:
    #     transform[0, 0] = -1
    #     transform[0, 2] = width
    # if v_flip:
    #     transform[1, 1] = -1
    #     transform[1, 2] = height
    # return transform
    transform = [[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]
    if h_flip:
        transform[0] = [-1., 0., width]
    if v_flip:
        transform[1] = [0., -1., height]
    return tf.convert_to_tensor(transform)


def wrap_tf_augment(augment_func):
    def tf_augment(image, label=None):
        if label is None:
            return tf.py_function(func=augment_func, inp=[image], Tout=tf.float32)
        return tf.py_function(func=augment_func, inp=[image, label], Tout=(tf.float32, tf.float32))
    return tf_augment


def get_image_augmenter(
    image_height,
    image_width,
    augment_chance=[1.0, 0.5],
    flip_h=False,
    flip_v=False,
    width_scaling=(1.0, 1.0),
    height_scaling=(1.0, 1.0),
    rotation_range=(0.0, 0.0),
    shear_range=(0.0, 0.0),
    x_shift=(0, 0),
    y_shift=(0, 0),
    brightness_range=(0.0, 0.0),
    contrast_range=(1.0, 1.0),
    random_distribution=tf.random.uniform,
    run_on_batch=True,
    as_tf_pyfunc=False,
    **kwargs
):
    """Get an augmentation function for TensorFlow pipelines

    augment_chance: list
        A list of probabilities for applying augments (ie: odds of the first augment being applied, odds of the second augment being applied if the first was applied, and so on) (the default is [1.0, 0.5])
    flip_h: bool
        Whether horizontal flips are allowed (the default is False)
    flip_v: bool
        Whether vertical flips are allowed (the default is False)
    width_scaling: (float, float)
        The minimum and maximum range to scale the width to (the default is (1.0, 1.0))
    height_scaling: (float, float)
        The minimum and maximum range to scale the height to (the default is (1.0, 1.0))
    rotation_range: (float, float)
        The minimum and maximum rotation in degrees that can be applied (the default is (0, 0))
    shear_range: (float, float)
        The minimum and maximum shear in degrees that can be applied (the default is (0, 0))
    x_shift: (float, float)
        The minimum and maximum pixels to shift left that can be applied (the default is (0, 0))
    y_shift: (float, float)
        The minimum and maximum pixels to shift upwards that can be applied (the default is (0, 0))
    brightness_range: (float, float)
        The minimum and maximum delta to apply to image brightness (the default is (0, 0))
    contrast_range: (float, float)
        The minimum and maximum factor to change image contrast by (the default is (1, 1))
    random_distribution: function
        The function for the random distribution to use for augmentation values (the default is numpy.random.uniform)
    run_on_batch: bool
        Whether the augment func will be called on batches (the default is True)

    Notes
    -----
    All default values result in no change to the image. Only augmentations that can result in a change in the image will be drawn from when randomly applying augmentations.
    Labels sent to the augmentation function are assumed to be label-maps with the same shape as the input image.
    random_distribution assumes input in the form of random_distribution(min_val, max_val). It's main purpose is to allow seeded random generators rather than different distribution functions.
    If a single value is passed for width_scaling, height_scaling, or contrast_range, the range becomes (1 - value, 1 + value).
    If a single value is passed for rotation_range, shear_range, x_shift, y_shift, or brightness_range, the range becomes (-value, value).

    """
    # rotation_range = np.radians(rotation_range)  # removed - convert to radians later
    # shear_range = np.radians(shear_range)  # removed - convert to radians later
    if not hasattr(width_scaling, '__len__') or len(width_scaling) == 1:
        width_scaling = (1 - width_scaling, 1 + width_scaling)
    if not hasattr(height_scaling, '__len__') or len(height_scaling) == 1:
        height_scaling = (1 - height_scaling, 1 + height_scaling)
    if not hasattr(rotation_range, '__len__') or len(rotation_range) == 1:
        rotation_range = (-rotation_range, rotation_range)
    if not hasattr(shear_range, '__len__') or len(shear_range) == 1:
        shear_range = (-shear_range, shear_range)
    if not hasattr(x_shift, '__len__') or len(x_shift) == 1:
        x_shift = (-x_shift, x_shift)
    if not hasattr(y_shift, '__len__') or len(y_shift) == 1:
        y_shift = (-y_shift, y_shift)
    if not hasattr(brightness_range, '__len__') or len(brightness_range) == 1:
        brightness_range = (-brightness_range, brightness_range)
    if not hasattr(contrast_range, '__len__') or len(contrast_range) == 1:
        contrast_range = (1 - contrast_range, 1 + contrast_range)

    choices = []
    if flip_h:
        choices.append('flip_h')
    if flip_v:
        choices.append('flip_v')
    if width_scaling[0] < 1.0 or width_scaling[1] > 1.0:
        choices.append('width_scaling')
    if height_scaling[0] < 1.0 or height_scaling[1] > 1.0:
        choices.append('height_scaling')
    if rotation_range[0] != 0 or rotation_range[1] != 0:
        choices.append('rotation')
    if shear_range[0] != 0 or shear_range[1] != 0:
        choices.append('shear')
    if x_shift[0] != 0 or x_shift[1] != 0:
        choices.append('x_shift')
    if y_shift[0] != 0 or y_shift[1] != 0:
        choices.append('y_shift')
    if brightness_range[0] < 0 or brightness_range[1] > 0:
        choices.append('brightness')
    if contrast_range[0] < 1.0 or contrast_range[1] > 1.0:
        choices.append('contrast')
    max_choices = len(choices)
    augment_chance = np.array(augment_chance)

    def _quick_random(vals):
        return random_distribution([], minval=vals[0], maxval=vals[1])

    def augment_func(image, label=None):
        transform = tf.eye(3, dtype=tf.float32)
        affine_changed = False
        num_choices = tf.math.count_nonzero(tf.random.uniform(augment_chance.shape, 0, 1) < augment_chance)
        num_choices = tf.minimum(num_choices, max_choices)
        for augment in np.random.choice(choices, num_choices, replace=False):
            if augment == 'flip_h':
                transform = transform @ flip_matrix(image_height, image_width, False, True)
                affine_changed = True
            elif augment == 'flip_v':
                transform = transform @ flip_matrix(image_height, image_width, True, False)
                affine_changed = True
            elif augment == 'width_scaling':
                transform = transform @ scaling_matrix(image_height, image_width, _quick_random(width_scaling), 1.0)
                affine_changed = True
            elif augment == 'height_scaling':
                transform = transform @ scaling_matrix(image_height, image_width, 1.0, _quick_random(height_scaling))
                affine_changed = True
            elif augment == 'rotation':
                transform = transform @ rotation_matrix(image_width // 2, image_height // 2, _quick_random(rotation_range))
                affine_changed = True
            elif augment == 'shear':
                transform = transform @ shear_matrix(_quick_random(shear_range))
                affine_changed = True
            elif augment == 'x_shift':
                transform = transform @ shift_matrix(_quick_random(x_shift), 0)
                affine_changed = True
            elif augment == 'y_shift':
                transform = transform @ shift_matrix(0, _quick_random(y_shift))
                affine_changed = True
            elif augment == 'brightness':
                image = tf.image.adjust_brightness(image, _quick_random(brightness_range))
            elif augment == 'contrast':
                image = tf.image.adjust_contrast(image, _quick_random(contrast_range))
            else:
                raise ValueError("Unknown augmentation selected?")
        if affine_changed:
            transform = tf.cast(transform, tf.float32)
            transform = matrices_to_flat_transforms(transform, invert=True)
            image = tf_transform(image, transform, interpolation='BILINEAR')
            if label is not None:
                label = tf_transform(label, transform, interpolation='NEAREST')
            if not run_on_batch:
                image = image[0]
                if label is not None:
                    label = label[0]
        if label is not None:
            return image, label
        return image
    if as_tf_pyfunc:
        return wrap_tf_augment(augment_func)
    return augment_func


def get_blur_augmenter(
    image_height,
    image_width,
    filter_range=[3, 5],
    sigma_range=[1, 5],
    random_distribution=tf.random.uniform,
    run_on_batch=True,
    as_tf_pyfunc=False,
):
    def _rand_int(vals):
        return random_distribution([], minval=vals[0], maxval=vals[1], dtype=tf.int32)

    def augment_func(image, label=None):
        image = enforce_4D(image)
        filter_size = random_distribution([], minval=filter_range[0], maxval=filter_range[1], dtype=tf.int32)
        sigma = random_distribution([], minval=sigma_range[0], maxval=sigma_range[1], dtype=tf.float32)
        blurred = gaussian_blur(image, filter_size=filter_size, sigma=sigma)
        if not run_on_batch:
            blurred = blurred[0]
        if label is not None:
            return blurred, label
        return blurred

    if as_tf_pyfunc:
        return wrap_tf_augment(augment_func)
    return augment_func


def get_normal_noise_augmenter(mean=0.0, stddev=0.05, as_tf_pyfunc=False):
    def augment_func(image, label=None):
        image += tf.random.normal(image.shape, mean=mean, stddev=stddev, dtype=image.dtype)
        if label is not None:
            return image, label
        return image

    if as_tf_pyfunc:
        return wrap_tf_augment(augment_func)
    return augment_func
