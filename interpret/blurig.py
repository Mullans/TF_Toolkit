"""Not sure if this works - seems to be pre-publication https://github.com/PAIR-code/saliency/blob/master/saliency/blur_ig.py"""

import math
import numpy as np
import tqdm
from .saliencymask import SaliencyMask
from .utils import gaussian_blur


class BlurIG(SaliencyMask):
    def get_mask(self, x_value, max_sigma=50, steps=100, grad_step=0.01, sqrt=False, output_of_interest=-1):
        if sqrt:
            sigmas = [math.sqrt(float(i) * max_sigma / float(steps)) for i in range(0, steps + 1)]
        else:
            sigmas = [float(i) * max_sigma / float(steps) for i in range(0, steps + 1)]
        step_vector_diff = [sigmas[i + 1] - sigmas[i] for i in range(0, steps)]

        total_gradients = np.zeros_like(x_value)
        for i in tqdm.trange(steps):
            x_step = gaussian_blur(x_value, sigmas[i])
            gaussian_gradient = (gaussian_blur(x_value, sigmas[i] + grad_step) - x_step) / grad_step
            total_gradients += step_vector_diff[i] * np.multiply(gaussian_gradient, super(BlurIG, self).get_mask(x_step, output_of_interest=output_of_interest))

        total_gradients *= -1.0
        return total_gradients
