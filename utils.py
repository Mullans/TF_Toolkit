import gouda
import numpy as np
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE


def register_goudapath_tensor():
    """Registers gouda.GoudaPath with tensorflow, so it can be converted to a Tensor object (only the absolute path is used)"""
    def goudapath_to_tensor(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(value.abspath, dtype=dtype, name=name)
    tf.register_tensor_conversion_function(gouda.GoudaPath, goudapath_to_tensor)


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


def get_image_augmenter(random_crop=[30, 30], flip_h=True, flip_v=True, after_batching=False):
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
            height = shape[1]
            width = shape[2]
        else:
            height = shape[0]
            width = shape[1]
        image = tf.image.resize_with_crop_or_pad(image, height + random_crop[0], width + random_crop[1])
        image = tf.image.random_crop(image, image_shape)
        if flip_h:
            image = tf.image.random_flip_left_right(image)
        if flip_v:
            image = tf.image.random_flip_up_down(image)
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
def scale_image(image, label):
    """Rescale the image to have a minimum of 0 and maximum of 1"""
    min_pix = tf.reduce_min(image)
    max_pix = tf.reduce_max(image)
    image = (image - min_pix) / (max_pix - min_pix)
    return image, label


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
    """
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
