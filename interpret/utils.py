"""Utility methods for model interprebility"""
import cv2
import numpy as np
from scipy import ndimage
import skimage.transform
import skimage.segmentation
import tensorflow as tf


def resize(image, output_shape, method='fast', anti_aliasing=False):
    """Resize image

    Parameters
    ----------
    image : np.ndarray
        Image to resize
    output_shape : [int, int]
        Shape for the resized image
    method : str
        Method to use when resizing - use `fast` for the fastest option or `auto` for the option with the least reconstruction error (the default is `fast`)
    anti_aliasing : bool
        Whether to apply anti-aliasing before downsampling - only applied to downsampling skimage methods

    NOTE
    ----
    Possible methods are: [cv2_area, cv2_linear, skimage_0, skimage_3, auto, fast].
    `auto` uses skimage_0 for upsampling and cv2_area for downsampling, and `fast` always uses cv_area
    """
    height, width = image.shape[:2]
    dest_height, dest_width = output_shape[:2]
    upsampling = height > dest_height  # assumes height & width are bigger on one
    if method == 'fast':
        method = 'cv2_area'
    elif method == 'auto':
        if upsampling:
            method = 'skimage_0'  # skimage_3 has slightly better reconstruction accuracy for upsizing, but skimage_0 is ~8-12x faster
        else:
            method = 'cv2_area'

    if method == 'cv2_area':
        return cv2.resize(image, (dest_width, dest_height), interpolation=cv2.INTER_AREA)
    elif method == 'cv2_linear':
        return cv2.resize(image, (dest_width, dest_height), interpolation=cv2.INTER_LINEAR)
    elif method == 'skimage_0':
        return skimage.transform.resize(
            image,
            (dest_height, dest_width),
            order=0,
            preserve_range=True,
            mode='constant',
            anti_aliasing=anti_aliasing and not upsampling
        )
    elif method == 'skimage_3':
        return skimage.transform.resize(
            image,
            (dest_height, dest_width),
            order=3,
            preserve_range=True,
            mode='constant',
            anti_aliasing=anti_aliasing and not upsampling
        )


def normalize_image(image, out_range, out_shape=None):
    """Normalize and resize an image

    Parameters
    ----------
    image : np.ndarray
        Image to normalize/reshape
    out_range : [int, int]
        Minimum and maximum values for the output image
    out_shape : None | (int, int)
    """
    image = np.squeeze(image)
    if out_shape is not None:
        image = resize(image, out_shape, method='skimage_0')
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (out_range[1] - out_range[0]) + out_range[0]
    return image


def get_felzenszwalb(image,
                     resize_image=True,
                     rescaled_size=(224, 224),
                     sigma_values=[0.8],
                     rescale_range=[-1.0, 1.0],
                     dilation_radius=5):
    """Compute image segments based on Felzenszwalb's algorithm

    Parameters
    ----------
    image : np.ndarray
        Input image to segment
    resize : bool
        Whether to resize the image for the segmentation - rescaled back to source afterwards
    rescaled_size : (int, int)
        If resize, what size should the image be rescaled to?
    scale_range : [float, float]
        Value min and max to rescale the image to
    dilation_radius : int
        How much each segment is dilated to include edges - larger=blobbier, smaller=sharper

    Note
    ----
    https://doi.org/10.1023/B:VISI.0000022288.19776.77
    """
    parameters = {
        # 'image_rescale': (224, 224),
        'scale_values': [50, 10, 150, 250, 500, 1200],
        'sigma_values': [0.8],
        'min_segment_size': 150
    }

    height, width = image.shape[:2]
    if resize_image:
        image = normalize_image(image, rescale_range, rescaled_size)
    else:
        image = normalize_image(image, rescale_range)

    segments = []
    for scale in parameters['scale_values']:
        for sigma in parameters['sigma_values']:
            segment = skimage.segmentation.felzenszwalb(image, scale=scale, sigma=sigma, min_size=parameters['min_segment_size'])
            segment = segment.astype(np.uint8)  # required for cv2.resize
            if resize_image:
                segment = resize(segment, (height, width), method='auto')
            segments.append(segment)
    masks = []
    for segment in segments:
        for value in range(segment.min(), segment.max() + 1):
            masks.append(segment == value)
    if dilation_radius:
        selem = skimage.morphology.disk(dilation_radius)
        masks = [skimage.morphology.dilation(mask, selem=selem) for mask in masks]
    return masks


def mask_difference(mask_1, mask_2, as_count=False):
    # Return where mask_1 is True and mask_2 is False
    diff = np.logical_and(mask_1, np.logical_not(mask_2))
    if as_count:
        return diff.sum()
    else:
        return diff


def gain_density(mask_1, attr, mask_2=None):
    if mask_2 is None:
        added_mask = mask_1
    else:
        added_mask = np.logical_and(mask_1, np.logical_not(mask_2))
    if not np.any(added_mask):
        return float('-inf')
    else:
        return attr[added_mask].mean()


def gaussian_blur(image, sigma):
    if sigma == 0:
        return image
    return ndimage.gaussian_filter(image, sigma=[sigma, sigma, 0], mode='constant')


def tf_rescale(image, keep_sign=False):
    if keep_sign:
        sign = tf.sign(image)
        image = tf.abs(image)
    min_pix = tf.reduce_min(image)
    max_pix = tf.reduce_max(image)
    if max_pix == min_pix:
        return tf.zeros(image.shape)
    image = (image - min_pix) / (max_pix - min_pix)
    if keep_sign:
        image = image * sign
    return image
