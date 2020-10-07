import numpy as np
import tensorflow as tf


class Occlusion(object):
    def __init__(self, model):
        self.model = model
        self.dtype = model.dtype

    def __enforce_shape(self, input_value):
        if not isinstance(input_value, tf.Tensor):
            input_value = tf.convert_to_tensor(input_value)

        if input_value.ndim == 2:
            input_value = tf.expand_dims(input_value, -1)

        if input_value.ndim == 3 and input_value.shape[-1] == 1:
            input_value = tf.expand_dims(input_value, 0)
        elif input_value.ndim == 3 and input_value.shape[0] == 1:
            input_value = tf.expand_dims(input_value, -1)

        return tf.cast(input_value, self.dtype)

    def __occlude2D(self, image, num_patches=10, overlap=0.5, background_value=0):
        """Get occluded versions of given image

        Parameters
        ----------
        image : np.ndarray | tf.tensor
            Image to occlude
        num_patches : int | [int, int]
            Number of horizontal and vertical patches to use (the default is 10)
        overlap : float
            Amount of overlap between neighboring patches as a fraction (the default is 0.5)
        background_value : int | float
            The value to use for occlusion patches
        """
        if overlap >= 1:
            raise ValueError("Overlap must be a fraction between 0 and 1")
        if not hasattr(num_patches, '__len__'):
            num_patches = [num_patches, num_patches]
        num_patches = np.array(num_patches).astype(int)
        image_shape = image.shape[:2]
        patch_size = image_shape / (num_patches - overlap * num_patches + overlap)

        locations = []
        images = []
        row_step, col_step = patch_size * (1 - overlap)
        for row in range(num_patches[0]):
            if row == num_patches[0] - 1:
                row_slice = slice(int(row * row_step), int(image_shape[0]))
            else:
                row_slice = slice(int(row * row_step), int(row * row_step + patch_size[0]))
            for col in range(num_patches[1]):
                if col == num_patches[1] - 1:
                    col_slice = slice(int(col * col_step), int(image_shape[1]))
                else:
                    col_slice = slice(int(col * col_step), int(col * col_step + patch_size[1]))
                locations.append([row_slice, col_slice])
                new_img = np.copy(image)
                new_img[row_slice, col_slice] = background_value
                images.append(new_img)
        images = np.stack(images, axis=0)
        return images, locations

    def get_mask(self, x_value, num_patches=10, overlap=0.5, class_of_interest=0, background_value=0, batch_size=32):
        """Get occlusion-based heatmap of input influence on prediction

        Parameters
        ----------
        x_value : np.ndarray | tf.tensor
            Input image to occlude
        label : int
            Ground truth label for input
        num_patches : int | [int, int]
            Number of horizontal and vertical patches to use (the default is 10)
        overlap : float
            Amount of overlap between neighboring patches as a fraction (the default is 0.5)
        background_value : int | float
            The value to use for occlusion patches
        batch_size : int
            The number of occluded samples to process at a time (the default is 32)
        """
        x_value = self.__enforce_shape(x_value)
        images, locations = self.__occlude2D(x_value[0], num_patches=num_patches, overlap=overlap, background_value=background_value)
        baseline = self.model(x_value)[:, class_of_interest]
        batches = [images[i:i + batch_size] for i in range(0, images.shape[0], batch_size)]
        results = np.concatenate([self.model(batch)[:, class_of_interest] for batch in batches], axis=0)

        heat_map = np.zeros(x_value.shape[1:3])
        counts = np.zeros(x_value.shape[1:3])
        for idx, loc in enumerate(locations):
            change = results[idx]
            heat_map[loc[0], loc[1]] += baseline - results[idx]
            counts[loc[0], loc[1]] += 1
        heat_map = np.divide(heat_map, counts)

        return heat_map
