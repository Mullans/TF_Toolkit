import os

import gouda
import logging
import numpy as np
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE


def allow_gpu_growth():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


def set_tf_loglevel(level):
    if isinstance(level, str):
        level_dict = {
            'fatal': 3,
            'error': 2,
            'warning': 1,
            'info': 0
        }
        if level.lower() in level_dict:
            level = level_dict[level]
        else:
            raise ValueError('Unknown logging level: "{}"'.format(level))
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    elif level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    elif level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def register_goudapath_tensor():
    """Registers gouda.GoudaPath with tensorflow, so it can be converted to a Tensor object (only the absolute path is used)"""
    def goudapath_to_tensor(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(value.abspath, dtype=dtype, name=name)
    tf.register_tensor_conversion_function(gouda.GoudaPath, goudapath_to_tensor)


def enforce_4D(image, dtype=tf.float32):
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image)

    ndims = image.get_shape().ndims
    if ndims == 2:
        image = image[None, :, :, None]
    elif ndims == 3 and image.shape[0] == 1:
        image = image[:, :, :, None]
    elif ndims == 3:
        image = image[None, :, :, :]
    elif ndims != 4:
        raise ValueError('Unknown image shape: {}'.format(image.shape))
    return tf.cast(image, dtype)


@tf.function
def count_nonzero(x, y):
    """Nonzero counter for input/label paired datasets"""
    return tf.cast(tf.math.count_nonzero(y), tf.int32), tf.size(y)


def get_binary_class_weights(dataset, return_initial_bias=False):
    """Get suggested class weights for binary classifications.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset with items in form (x, y) where y is a binary label
    return_initial_bias : bool
        Whether a suggested initial bias for the output layer weights should be returned (the default is False)
    """
    start_val = tf.Variable([0, 0], tf.int32)
    counted = dataset.map(count_nonzero, num_parallel_calls=AUTOTUNE)
    counted = counted.reduce(start_val, lambda x, y: x + y)
    pos, total = np.array(counted)
    zero_weight = (1 / (total - pos)) * (total / 2.0)
    one_weight = (1 / pos) * (total / 2.0)
    if return_initial_bias:
        return [zero_weight, one_weight], np.log([pos / (total - pos)])
    else:
        return [zero_weight, one_weight]


def get_image_augmenter_lite(random_crop=[30, 30], flip_h=True, flip_v=True, after_batching=False):
    """Return an augmentation method for TF image dataset pipelines

    Parameters
    ----------
    random_crop : [int, int]
        The amount to increase and then randomly crop away in height and width (the default is [30, 30])
    flip_h : bool
        Whether the image should be horizontally flipped with a 50% chance (the default is True)
    flip_v : bool
        Whether the image should be vertically flipped with a 50% chance (the default is True)
    after_batching : bool
        Whether the augmentation will be used after images are batched (the default is False)
    """
    def augment_func(image, label):
        image_shape = tf.shape(image)
        if after_batching:
            height = image_shape[1]
            width = image_shape[2]
        else:
            height = image_shape[0]
            width = image_shape[1]
        image = tf.image.resize_with_crop_or_pad(image, height + random_crop[0], width + random_crop[1])
        image = tf.image.random_crop(image, image_shape)
        if flip_h:
            image = tf.image.random_flip_left_right(image)
        if flip_v:
            image = tf.image.random_flip_up_down(image)
        return image, label
    return tf.function(augment_func)


def tf_transform(images, transforms, interpolation='NEAREST', output_shape=None, name=None):
    """Applies the given transform(s) to the image(s).

    Note: This is just a quick copy of
    tensorflow_addons.image.transform_ops.transform for the augmenter.
    """
    with tf.name_scope(name or 'transform'):
        images = enforce_4D(images)
        if len(transforms.get_shape()) == 1:
            transforms = transforms[None]
        elif transforms.get_shape().ndims is None:
            raise ValueError("transforms rank must be statically known")
        elif len(transforms.get_shape()) == 2:
            transforms = transforms
        else:
            transforms = transforms
            raise ValueError(
                "transforms should have rank 1 or 2, but got rank %d"
                % len(transforms.get_shape())
            )

        output = tf.raw_ops.ImageProjectiveTransformV2(
            images=images,
            transforms=transforms,
            output_shape=tf.shape(images)[1:3],
            interpolation=interpolation.upper(),
        )
        return output


def matrices_to_flat_transforms(transform_matrices, invert=False, name=None):
    """Converts affine matrices to projective transforms.

    Note: This is just a quick copy of
    tensorflow_addons.image.transform_ops.matrices_to_flat_transforms
    for the augmenter.
    """
    with tf.name_scope(name or "matrices_to_flat_transforms"):
        if invert:
            transform_matrices = tf.linalg.inv(transform_matrices)
        transform_matrices = tf.convert_to_tensor(
            transform_matrices, name="transform_matrices"
        )
        if transform_matrices.shape.ndims not in (2, 3):
            raise ValueError(
                "Matrices should be 2D or 3D, got: %s" % transform_matrices
            )
        transforms = tf.reshape(transform_matrices, tf.constant([-1, 9]))
        transforms /= transforms[:, 8:9]
        return transforms[:, :8]


def get_image_augmenter(flip_h=True,
                        flip_v=False,
                        width_scale_range=(1.0, 1.0),
                        height_scale_range=(1.0, 1.0),
                        max_rotation=0.0,
                        max_shear=0.0,
                        x_max_shift=30,
                        y_max_shift=30,
                        as_degrees=True,
                        label_as_map=True,
                        **kwargs):
    """Get an augmentation function for image/label paired data points

    Parameters
    ----------
    flip_h : bool
        Whether to randomly apply horizontal flips (the default is True)
    flip_v : bool
        Whether to randomly apply vertical flips (the default is True)
    width_scale_range : (float, float) | float
        The minimum and maximum scaling factor for image width OR the factor in either direction ie. 0.05 -> [0.95, 1.05] (the default is (1.0, 1.0))
    height_scale_range : (float, float) | float
        The minimum and maximum scaling factor for image height OR the factor in either direction ie. 0.05 -> [0.95, 1.05] (the default is (1.0, 1.0))
    max_rotation : float
        The maximum rotation (in either direction) for the image (the default is 0)
    max_shear : float
        The maximum shear angle (in either direction) for the image (the default is 0)
    x_max_shift : int
        The maximum shift in pixels for the image (the default is 30)
    y_max_shift : int
        The maximum shift in pixels for the image (the default is 30)
    as_degrees : bool
        Whether the input rotation/shear values are in degrees rather than radians (the default is True)
    label_as_map : bool
        Whether to treat the label as a class map with the same size as the image or as a scalar class (the default is True).
    """
    if as_degrees:
        max_rotation = np.radians(max_rotation)
        max_shear = np.radians(max_shear)
    if not hasattr(width_scale_range, '__len__') or len(width_scale_range) == 1:
        width_scale_range = (1 - width_scale_range, 1 + width_scale_range)
    if not hasattr(height_scale_range, '__len__') or len(height_scale_range) == 1:
        height_scale_range = (1 - height_scale_range, 1 + height_scale_range)

    def augment_func(image, label):
        if flip_h:
            if tf.random.normal([1]) > 0:
                image = tf.image.flip_left_right(image)
                if label_as_map:
                    label = tf.image.flip_left_right(label)
        if flip_v:
            if tf.random.normal([1]) > 0:
                image = tf.image.flip_up_down(image)
                if label_as_map:
                    label = tf.image.flip_up_down(label)

        width_scale = tf.random.uniform([], *width_scale_range)
        height_scale = tf.random.uniform([], *height_scale_range)
        rotation = tf.random.uniform([], -max_rotation, max_rotation)
        shear = tf.random.uniform([], -max_shear, max_shear)
        shift_x = tf.random.uniform([], -x_max_shift, x_max_shift)
        shift_y = tf.random.uniform([], -y_max_shift, y_max_shift)

        row_1 = tf.stack([width_scale * tf.cos(rotation), -height_scale * tf.sin(rotation + shear), shift_x])
        row_2 = tf.stack([width_scale * tf.sin(rotation), height_scale * tf.cos(rotation + shear), shift_y])
        row_3 = tf.stack([0.0, 0.0, 1.0])
        params = tf.stack([row_1, row_2, row_3])

        flat_transform = matrices_to_flat_transforms(tf.linalg.inv(params))
        image = tf_transform(image, flat_transform, interpolation='BILINEAR')
        if label_as_map:
            label = tf_transform(label, flat_transform, interpolation='NEAREST')
        return image, label
    return tf.function(augment_func)


@tf.function
def sobel_converter(image, label):
    """Convert the image to a gradient magnitude estimation using sobel edge detection"""
    out_shape = tf.shape(image)
    image = tf.reshape(image, [-1, 1024, 1360, 1])
    sobels = tf.image.sobel_edges(image)
    sobels = tf.reduce_sum(sobels ** 2, axis=-1)
    sobels = tf.sqrt(sobels)

    # rescale back to [0, 1]
    min_pix = tf.reduce_min(sobels, axis=(1, 2, 3), keepdims=True)
    max_pix = tf.reduce_max(sobels, axis=(1, 2, 3), keepdims=True)
    sobels = tf.divide(tf.subtract(sobels, min_pix), tf.subtract(max_pix, min_pix))
    sobels = tf.reshape(sobels, out_shape)
    return sobels, label


@tf.function
def rescale_image(image, dtype=tf.float32):
    """Rescale the image to have a minimum of 0 and maximum of 1"""
    image = tf.cast(image, dtype)
    min_pix = tf.reduce_min(image)
    max_pix = tf.reduce_max(image)
    image = (image - min_pix) / (max_pix - min_pix)
    return image


def numpy_to_native(item):
    return getattr(item, 'tolist', lambda: item)()


# def animate_samples(model_name, tofile=None, from_gan=True):
#     """From gan makes it so that few frames are used from later epochs as training slows."""
#     if tofile is None:
#         tofile = './{}/animated.gif'.format(model_name)
#     gif_path = tofile.rsplit('.', 1)[0] + '.gif'
# #     png_path = gif_path + '.png'
#     with imageio.get_writer(gif_path, mode='I') as writer:
#         filenames = sorted(glob.glob('./{}/examples/example_*.jpg'.format(model_name)))
#         last = -1
#         for i, filename in enumerate(filenames):
#             frame = 2 * (i ** 0.5)
#             if round(frame) > round(last):
#                 last = frame
#             else:
#                 continue
#             image = imageio.imread(filename)
#             writer.append_data(image)
#         image = imageio.imread(filenames[-1])
#         writer.append_data(image)
# #     os.system("cp {} {}".format(gif_path, png_path))


def elastic_transform(image, label=None, alpha=2, sigma=0.04):
    """Random elastic distortion as described in - Best Practices for Convolutional Neural Networks applied to Visual Document Analysis by Simard et al. 2003

    Parameters
    ----------
    image : np.ndarray
        Input image to augment
    label : np.ndarray
        An optional label mask that maps to the image - this ensures the label transforms the same way as the image (the default is None)
    alpha : int
        I don't remember - something to do with the blur strength (the default is 2)
    sigma : float
        I don't remember - something to do with the blur distance (the default is 0.04)

    NOTE:
    OpenCV is imported locally to this function so that utils.py can be used without it
    """
    import cv2

    alpha = image.shape[1] * alpha
    sigma = image.shape[1] * sigma

    dx = cv2.GaussianBlur((np.random.rand(*image.shape[:2]) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*image.shape[:2]) * 2 - 1), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    x = (x + dx).astype(np.float32)
    y = (y + dy).astype(np.float32)

    new_image = cv2.remap(image, x, y, cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    if label is None:
        return new_image
    new_label = cv2.remap(label, x, y, cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    return new_image, new_label

# TODO - replace the opencv calls in elastic_transform with tensorflow like below
# def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
#     x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
#     g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
#     g_norm2d = tf.pow(tf.reduce_sum(g), 2)
#     g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
#     g_kernel = tf.expand_dims(g_kernel, axis=-1)
#     return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)
# def apply_blur(img):
#     blur = _gaussian_kernel(3, 2, 3, img.dtype)
#     img = tf.nn.depthwise_conv2d(img[None], blur, [1,1,1,1], 'SAME')
#     return img[0]


def get_image_and_label_loader(image_type='jpg', dtype=tf.float32):
    """Get the tf.function to load an image/label pair.

    NOTE: tf.io.decode_jpeg/png is slightly faster than decode_image (~0.0002 sec/image)

    NOTE: The function will take a single tensor of size [2] with form [image_path, label_path]
    """
    if image_type == 'jpg':
        decoder = tf.io.decode_jpeg
    elif image_type == 'png':
        decoder = tf.io.decode_png
    else:
        decoder = tf.io.decode_image

    def loader(paths):
        image_path = paths[0]
        label_path = paths[1]
        image = decoder(tf.io.read_file(image_path))
        image = rescale_image(image)
        label = decoder(tf.io.read_file(label_path))
        label = rescale_image(label)
        return image, label

    return tf.function(loader)


def get_pad_func(vert_padding=0, horiz_padding=0):
    """Shortcut function for adding equal vertical or horizontal padding to an image/label paired dataset

    NOTE
    ----
    Assumes data is of shape [batch, x, y, channels]
    """
    padding = [[0, 0], [vert_padding, vert_padding], [horiz_padding, horiz_padding], [0, 0]]

    def pad_func(image, label):
        image = tf.pad(image, padding)
        label = tf.pad(label, padding)
        return image, label
    return tf.function(pad_func)


def get_center_of_mass(input_arr):
    """Find the center of mass for the input n-dimensional array.

    NOTE:
    Assumes that input_array has shape [batch, ...]
    If a sample in the batch has a mass of 0, the coordinates will be nan
    """
    grids = tf.meshgrid(*[tf.range(axis_size) for axis_size in input_arr.shape[1:]], indexing='ij')
    coords = tf.stack([tf.reshape(grid, (-1,)) for grid in grids], axis=-1)
    coords = tf.cast(coords, tf.float32)
    flat_mass = tf.reshape(input_arr, [-1, tf.reduce_prod(input_arr.shape[1:]), 1])
    total_mass = tf.reduce_sum(flat_mass, axis=1)
    return tf.math.divide(tf.reduce_sum(flat_mass * coords, axis=1), total_mass)


def rotation_matrix(center_x, center_y, rotation):
    """Get the 3x3 rotation transformation matrix

    Parameters
    ----------
    center: (int, int)
        The center point of the rotation
    rotation: int
        The amount of rotation in degrees, positive is clock-wise
    """
    alpha = np.cos(np.radians(rotation))
    beta = np.sin(np.radians(rotation))
    transform = np.array([[alpha, beta, (1 - alpha) * center_x - beta * center_y], [-beta, alpha, beta * center_y + (1 - alpha) * center_y], [0, 0, 1]])
    return transform


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
    shear = np.radians(shear)
    return np.array([[1, -1 * np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])


def shift_matrix(x_shift=0, y_shift=0):
    """Get the 3x3 translation transformation matrix

    Parameters
    ----------
    x_shift: int
        The number of pixels to shift the image towards the left (the default is 0)
    y_shift: int
        The number of pixels to shift the image upwards (the default is 0)
    """
    transform = np.eye(3, dtype=np.float32)
    transform[0, 2] = x_shift
    transform[1, 2] = y_shift
    return transform


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
    width_scale = 1 / width_scale
    height_scale = 1 / height_scale
    x_shift = (width * 0.5) * (1 - width_scale)
    y_shift = (height * 0.5) * (1 - height_scale)
    return np.array([[width_scale, 0, x_shift], [0, height_scale, y_shift], [0, 0, 1]])


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
    transform = np.eye(3)
    if h_flip:
        transform[0, 0] = -1
        transform[0, 2] = width
    if v_flip:
        transform[1, 1] = -1
        transform[1, 2] = height
    return transform


def get_image_augmenter_v2(
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
    random_distribution=np.random.uniform,
    run_on_batch=True,
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
    rotation_range = np.radians(rotation_range)
    shear_range = np.radians(shear_range)
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

    def augment_func(image, label=None):
        # shape = tf.shape(image)
        # if len(shape) == 4:
        #     height, width = float(shape[1]), float(shape[2])
        # elif len(shape) == 3:
        #     height, width = float(shape[0]), float(shape[1])
        # else:
        #     raise ValueError('Image/Label must have shape [batch, height, width, channels] or [height, width, channels], not {}'.format(image.shape))
        height = image_height
        width = image_width
        # print(shape, height, width)
        # print(height.shape, width.shape)
        transform = np.eye(3, dtype=np.float32)
        affine_changed = False
        num_choices = 0
        for p in augment_chance:
            if np.random.random() <= p:
                num_choices += 1
            else:
                break
        num_choices = min(num_choices, len(choices))
        for augment in np.random.choice(choices, num_choices, replace=False):
            if augment == 'flip_h':
                transform = transform @ flip_matrix(height, width, False, True)
                affine_changed = True
            elif augment == 'flip_v':
                transform = transform @ flip_matrix(height, width, True, False)
                affine_changed = True
            elif augment == 'width_scaling':
                transform = transform @ scaling_matrix(height, width, random_distribution(*width_scaling), 1.0)
                affine_changed = True
            elif augment == 'height_scaling':
                transform = transform @ scaling_matrix(height, width, 1.0, random_distribution(*height_scaling))
                affine_changed = True
            elif augment == 'rotation':
                transform = transform @ rotation_matrix(width // 2, height // 2, random_distribution(*rotation_range))
                affine_changed = True
            elif augment == 'shear':
                transform = transform @ shear_matrix(random_distribution(*shear_range))
                affine_changed = True
            elif augment == 'x_shift':
                transform = transform @ shift_matrix(random_distribution(*x_shift), 0)
                affine_changed = True
            elif augment == 'y_shift':
                transform = transform @ shift_matrix(0, random_distribution(*y_shift))
                affine_changed = True
            elif augment == 'brightness':
                image = tf.image.adjust_brightness(image, random_distribution(*brightness_range))
            elif augment == 'contrast':
                image = tf.image.adjust_contrast(image, random_distribution(*contrast_range))
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
                label = label[0]
        return image, label
    return augment_func
