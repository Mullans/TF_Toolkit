import os
import sys
import warnings
sys.path.append('../..')

from TF_Toolkit.core_model import CoreModel
import gouda
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from .utils import clean_segmentation, clip_image, crop_image_to_mask, faster_mask_body, resample_iso_by_slice_size, resample_to_ref

try:
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
except RuntimeError as _:
    warnings.warn("Physical devices cannot be modified after being initialized.")


class LungSegmenter(CoreModel):
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), 'model_args.json')
        self.model_args = gouda.load_json(config_path)
        self.compile_model()
        self.model.load_weights(os.path.join(os.path.dirname(__file__), 'model_weights/model_weights.tf'))

    def segment(self, image, crop_image=True, lung_threshold=0.1, clean=True, premask_body=False, use_max=False, cleaning_kwargs={}):
        """Segment the lungs in a ct image
            Parameters
            ----------
            image : str | SimpleITK.Image
                The source CT image to segment
            crop_image : bool
                Whether to crop the x and y planes of the image into the body (the default is True)
            lung_threshold : float
                The threshold for the predicted lung values (the default is 0.1)
            clean : bool
                Whether to apply post-prediction cleaning to the image (the default is True)
            premask_body : bool
                If clean is True and crop_image is True
        """
        if isinstance(image, str):
            image = sitk.ReadImage(image)
        image_size = image.GetSize()
        body_mask = None
        if crop_image:
            body_mask = faster_mask_body(image)
            x_slice, y_slice = crop_image_to_mask(image, mask=body_mask, crop_quantile=50, return_bounds=True)
            image = image[x_slice, y_slice, :]
            if not premask_body:
                body_mask = None  # clear the body mask so that it isn't used in cleaning
        elif premask_body:
            body_mask = faster_mask_body(image)

        src_image = image
        image = resample_iso_by_slice_size(image, [256, 256], outside_val=-1000)
        image = clip_image(image, -1000, 500)
        image_arr = sitk.GetArrayFromImage(image)

        image_arr = (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min())
        samples = tf.signal.frame(image_arr, 8, 4, pad_end=False, axis=0)
        batches = np.array_split(samples, int(np.ceil(samples.shape[0] / 32)))
        lung_output = np.zeros(image_arr.shape, dtype=np.float32)
        idx = 0
        for batch in batches:
            prediction = np.array(self.model(batch, training=False))
            for i in range(prediction.shape[0]):
                image_slice = slice(4 * (i + idx), 4 * (i + idx) + 8)
                if use_max:
                    lung_output[image_slice] += np.max(prediction[i], axis=-1)
                else:
                    lung_output[image_slice] += prediction[i, :, :, :, 0]
            idx += prediction.shape[0]
        lung_output = (lung_output > (lung_threshold * 2)).astype(np.uint8)
        lung_image = sitk.GetImageFromArray(lung_output)
        lung_image.CopyInformation(image)
        lung_image = resample_to_ref(lung_image, src_image)
        lung_image = sitk.BinaryFillhole(lung_image)

        if crop_image:
            pad_start = [x_slice.start, y_slice.start, 0]
            pad_end = [image_size[0] - x_slice.stop, image_size[1] - y_slice.stop, 0]
            lung_image = sitk.ConstantPad(lung_image, pad_start, pad_end, 0)
        if clean:
            lung_image = clean_segmentation(lung_image, body_mask=body_mask, **cleaning_kwargs)
        return lung_image
