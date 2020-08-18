"""Code for integrated gradients -- code adapted from https://github.com/PAIR-code/saliency"""
import numpy as np
import tensorflow as tf


class SaliencyMask(object):
    def __init__(self, model):
        self.model = model
        self.dtype = model.dtype
        self.num_outputs = model.outputs[0].shape[-1]

    def enforce_shape(self, input_value):
        if not isinstance(input_value, tf.Tensor):
            input_value = tf.convert_to_tensor(input_value)

        if input_value.ndim == 2:
            input_value = tf.expand_dims(input_value, -1)

        if input_value.ndim == 3 and input_value.shape[-1] == 1:
            input_value = tf.expand_dims(input_value, 0)
        elif input_value.ndim == 3 and input_value.shape[0] == 1:
            input_value = tf.expand_dims(input_value, -1)

        return tf.cast(input_value, self.dtype)

    def get_mask(self, x_value, output_of_interest=-1, as_tensor=False):
        x_value = self.enforce_shape(x_value)

        if output_of_interest > -1:
            # Return the gradient for a specific output
            with tf.GradientTape() as tape:
                tape.watch(x_value)
                output = tf.split(self.model(x_value), self.num_outputs, axis=-1)
            grads = tape.gradient(output[output_of_interest], x_value)[0]
        else:
            # Return the gradient for the entire output
            with tf.GradientTape() as tape:
                tape.watch(x_value)
                output = self.model(x_value)
            grads = tape.gradient(output, x_value)[0]
        if as_tensor:
            return grads
        else:
            return grads.numpy()

    def get_smoothed_mask(self, x_value, stdev_spread=0.15, nsamples=25, magnitude=True, output_of_interest=-1):
        stddev = stdev_spread * (np.max(x_value) - np.min(x_value))
        total_gradients = np.zeros_like(x_value)
        for i in range(nsamples):
            noise = np.random.normal(0, stddev, x_value.shape)
            x_plus_noise = x_value + noise
            grad = self.get_mask(x_plus_noise, output_of_interest=output_of_interest)
            if magnitude:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad
        return total_gradients / nsamples

    def get_integrated_mask(self, input_value, baseline=None, output_of_interest=-1, steps=25):
        """Get a saliency mask using the integrated gradients method

        Parameters
        ----------
        input_value : np.ndarray
            Input to find the gradients for
        baseline : np.ndarray
            Baseline value used in the integration - defaults to 0
        steps : number of integrated steps between the baseline and input

        https://arxiv.org/pdf/1703.01365.pdf
        """
        input_value = self.enforce_shape(input_value)

        if baseline is None:
            baseline = tf.zeros_like(input_value)
        else:
            baseline = self.enforce_shape(baseline)
        if baseline.shape != input_value.shape:
            raise ValueError("Baseline shape {} must match input shape {}".format(baseline.shape, input_value.shape))

        diff = input_value - baseline
        # total_gradients = np.zeros(input_value.shape, dtype=self.dtype.as_numpy_dtype)
        total_gradients = tf.zeros(input_value.shape, dtype=self.dtype)

        # TODO - batch these operations (?)
        for alpha in np.linspace(0, 1, steps):
            step = baseline + alpha * diff
            total_gradients += self.get_mask(step, as_tensor=True, output_of_interest=output_of_interest)
        return (total_gradients * diff / steps).numpy()
