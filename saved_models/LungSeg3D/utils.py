# ct_utils.clean_segmentation

import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import skimage.measure
import warnings


MAX_INTENSITY = 500
MIN_INTENSITY = -1000


def clean_segmentation(lung_image, lung_threshold=0.1, body_mask=None, ignore_extra_labels=False):
    """Clean a segmentation of the lungs

    Parameters
    ----------
    lung_image: SimpleITK.Image
        The segmentation of the lung
    lung_threshold: float
        The minimum threshold for a positive label (the default is 0.1)
    body_mask: sitk.Image or numpy.ndarray
        An optional boolean mask of the body (the default is None)
    ignore_extra_labels: bool
        Whether to ignore stray labels or to raise a ValueError (the default is False)
    """
    if isinstance(lung_image, sitk.Image):
        source_arr = sitk.GetArrayFromImage(lung_image)
    else:
        source_arr = np.copy(lung_image)
    if body_mask is not None:
        if isinstance(body_mask, sitk.Image):
            body_mask = sitk.GetArrayFromImage(body_mask)
        source_arr[~body_mask.astype(np.bool)] = 0
    if 'float' in str(source_arr.dtype):
        lung_arr = source_arr > lung_threshold
    else:
        lung_arr = source_arr

    labels = skimage.measure.label(lung_arr, connectivity=1, background=0)
    new_labels = np.zeros_like(lung_arr)
    bins = np.bincount(labels.flat)
    pos_lungs = (np.argwhere(bins[1:] > 100000) + 1).flatten()
    if len(pos_lungs) > 2 and not ignore_extra_labels:
        raise ValueError('Too Many Lungs')
    elif len(pos_lungs) == 1:
        warnings.warn('Only single segmented object detected')
        lungs = pos_lungs[0]
        new_labels[labels == lungs] = source_arr[labels == lungs]
    elif len(pos_lungs) == 2:
        lung1, lung2 = pos_lungs[:2]
        new_labels[labels == lung1] = source_arr[labels == lung1]
        new_labels[labels == lung2] = source_arr[labels == lung2]
    else:
        raise ValueError('No segmentated objects found')
    if isinstance(lung_image, sitk.Image):
        clean_image = sitk.GetImageFromArray(new_labels)
        clean_image.CopyInformation(lung_image)
        return clean_image
    return new_labels


def clip_image(image, low=MIN_INTENSITY, high=MAX_INTENSITY):
    """NOTE: UI-Lung values: [-1024, 1024], NVidia Lung Values [-1000, 500]"""
    image = sitk.Threshold(image, -32768, high, high)
    image = sitk.Threshold(image, low, high, low)
    # sitk.IntensityWindowing(image, -1000, 500, -1000, 500)
    return image


def crop_image_to_mask(image, mask=None, crop_z=False, crop_quantile=50, return_bounds=False):
    if mask is None:
        mask = mask_body(image)
    elif isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask)

    front_view = mask.max(axis=1)
    side_view = mask.max(axis=2)
    x_starts = []
    x_stops = []
    y_starts = []
    y_stops = []
    for idx in range(front_view.shape[0]):
        xrow = front_view[idx]
        if xrow.sum() == 0:
            continue
        xstart, xstop = np.where(xrow > 0)[0][[0, -1]]
        x_starts.append(xstart)
        x_stops.append(xstop)

        yrow = side_view[idx]
        if yrow.sum() == 0:
            continue
        ystart, ystop = np.where(yrow > 0)[0][[0, -1]]
        y_starts.append(ystart)
        y_stops.append(ystop)

    if crop_z:
        column = mask.max(axis=(1, 2))
        zstart, zstop = np.where(column > 0)[0][[0, -1]]
        z_slice = slice(zstart, zstop)

    sizex, sizey, _ = image.GetSize()
    sizex = int(sizex)
    sizey = int(sizey)
    x_stop = sizex - np.percentile(x_starts, crop_quantile)
    x_start = sizex - np.percentile(x_stops, 100 - crop_quantile)
    y_stop = np.percentile(y_stops, 100 - crop_quantile)
    y_start = np.percentile(y_starts, crop_quantile)
    lengthx = x_stop - x_start
    lengthy = y_stop - y_start
    cropx = sizex - lengthx
    cropy = sizey - lengthy
    to_crop = min(cropx, cropy)

    left = sizex - x_stop
    right = x_start
    if right > left:
        diff = right - left
        extra = to_crop - diff
        x_slice = slice(int(extra / 2), sizex - int(diff + extra / 2))
    else:
        diff = left - right
        extra = to_crop - diff
        x_slice = slice(int(diff + extra / 2), sizex - int(extra / 2))

    front = y_start
    back = sizey - y_stop
    if front > back:
        diff = min(to_crop, front - back)
        extra = to_crop - diff
        y_slice = slice(int(diff + extra / 2), sizey - int(extra / 2))
    else:
        diff = min(to_crop, back - front)
        extra = to_crop - diff
        y_slice = slice(int(extra / 2), sizey - int(diff + extra / 2))

    if crop_z:
        bounds = (x_slice, y_slice, z_slice)
    else:
        bounds = (x_slice, y_slice)

    if return_bounds:
        return bounds

    cropped = image[bounds]
    return cropped


def fill2d(arr):
    if isinstance(arr, sitk.Image):
        arr = sitk.ConstantPad(arr, [0, 0, 1], [0, 0, 1], 1)
        arr = sitk.BinaryFillhole(arr)
        return arr[:, :, 1:-1]
    output = np.zeros_like(arr)
    for idx in range(arr.shape[0]):
        check_slice = arr[idx]
        # check_slice = scipy.ndimage.binary_dilation(check_slice)
        filled = scipy.ndimage.binary_fill_holes(check_slice)
        output[idx] = filled
    return output


def mask_body(image, opening_size=1):
    """Generate a mask of the body in a 3D CT"""
    if not isinstance(image, sitk.Image):
        raise ValueError("mask_body requires a SimpleITK.Image object")
    bin_img = sitk.RecursiveGaussian(image, 3)
    bin_img = sitk.BinaryThreshold(bin_img, -500, 10000, 1, 0)
    if opening_size > 0:
        bin_img = sitk.BinaryMorphologicalOpening(bin_img, [opening_size] * 3, sitk.sitkBall, 0, 1)
    labels = sitk.ConnectedComponent(bin_img)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.Execute(labels)
    body_label = [-1, -1]
    for label in lfilter.GetLabels():
        label_area = lfilter.GetNumberOfPixels(label)
        if label_area > body_label[1]:
            body_label = [label, label_area]
    bin_img = sitk.Equal(labels, body_label[0])
    bin_img = sitk.BinaryMorphologicalClosing(bin_img, [3, 3, 3], sitk.sitkBall, 1)
    filled_labels = fill2d(bin_img)
    return filled_labels


def faster_mask_body(image, resample=True):
    src_image = image
    if resample:
        image = resample_iso_by_slice_size(image, 128, interp=sitk.sitkLinear)
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius(3)
    image = median_filter.Execute(image)
    bin_img = sitk.Greater(image, -500)
    labels = sitk.ConnectedComponent(bin_img)
    lfilter = sitk.LabelShapeStatisticsImageFilter()
    lfilter.Execute(labels)
    body_label = [-1, -1]
    for label in lfilter.GetLabels():
        label_area = lfilter.GetNumberOfPixels(label)
        if label_area > body_label[1]:
            body_label = [label, label_area]
    bin_img = sitk.Equal(labels, body_label[0])
    bin_img = sitk.ConstantPad(bin_img, [0, 0, 1], [0, 0, 1], 1)
    bin_img = sitk.BinaryFillhole(bin_img)[:, :, 1:-1]
    bin_img = resample_to_ref(bin_img, src_image, sitk.sitkNearestNeighbor)
    return bin_img


def resample_iso_by_slice_size(image, output_size, outside_val=MIN_INTENSITY, interp=sitk.sitkBSpline):
    """Resample an image to a given slice size and enforce equal spacing in all dimensions"""
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    input_origin = image.GetOrigin()
    input_direction = image.GetDirection()
    if not hasattr(output_size, '__len__'):
        output_size = [output_size, output_size]
    if len(output_size) == 2:
        output_size = list(output_size) + [input_size[2]]

    output_spacing = (np.array(input_size) * np.array(input_spacing)) / np.array(output_size)
    output_spacing[2] = output_spacing[0]
    output_size[2] = (input_size[2] * input_spacing[2]) / output_spacing[2]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interp)
    resample_filter.SetOutputDirection(input_direction)
    resample_filter.SetOutputOrigin(input_origin)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetSize([int(s) for s in output_size])
    resample_filter.SetDefaultPixelValue(outside_val)
    resampled_image = resample_filter.Execute(image)
    return resampled_image


def resample_to_ref(im, ref, outside_val=0, interp=sitk.sitkNearestNeighbor):
    """Resample an image to match a reference image"""
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetInterpolator(interp)
    resampleFilter.SetDefaultPixelValue(outside_val)
    if isinstance(ref, sitk.Image):
        resampleFilter.SetReferenceImage(ref)
    elif isinstance(ref, sitk.ImageFileReader):
        resampleFilter.SetSize(ref.GetSize())
        resampleFilter.SetOutputOrigin(ref.GetOrigin())
        resampleFilter.SetOutputSpacing(ref.GetSpacing())
        resampleFilter.SetOutputDirection(ref.GetDirection())
    elif isinstance(ref, dict):
        resampleFilter.SetSize(ref['size'])
        resampleFilter.SetOutputOrigin(ref['origin'])
        resampleFilter.SetOutputSpacing(ref['spacing'])
        resampleFilter.SetOutputDirection(ref['direction'])
    else:
        raise ValueError("Unknown reference type: '{}'".format(type(ref)))
    resampleIm = resampleFilter.Execute(im)
    return resampleIm
