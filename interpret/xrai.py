"""Code for XRAI alorithm -- adapted from https://github.com/PAIR-code/saliency"""

import numpy as np
import tqdm

from .utils import gain_density, get_felzenszwalb
from .saliencymask import SaliencyMask


class XRAI(SaliencyMask):
    def __init__(self,
                 model,
                 integration_steps=50,
                 minimum_coverge=1.0,
                 min_mask_pixel_difference=50,
                 gain_function=gain_density,
                 flatten_output_segments=True,
                 felzenszwalb_rescale=(224, 224),
                 felzenszwalb_sigmas=[0.8],
                 use_fast_xrai=False,
                 ):
        super(XRAI, self).__init__(model)
        self.__integrated_gradients = SaliencyMask(model)
        self.steps = integration_steps
        self.coverage = minimum_coverge
        self.min_mask_difference = min_mask_pixel_difference
        self.gain_function = gain_function
        self.flatten_segments = flatten_output_segments
        self.felzenszwalb_sigmas = felzenszwalb_sigmas
        if isinstance(felzenszwalb_rescale, float):
            height, width = model.input_shape[1:3]
            scale_height = int(height * felzenszwalb_rescale)
            scale_width = int(width * felzenszwalb_rescale)
            self.felzenszwalb_rescale = (scale_height, scale_width)
        else:
            self.felzenszwalb_rescale = felzenszwalb_rescale
        self.use_fast_xrai = use_fast_xrai

    def __get_integrated_gradients(self, image, baselines, steps=None, output_of_interest=-1):
        if steps is None:
            steps = self.steps
        gradients = []
        for baseline in baselines:
            gradient = self.__integrated_gradients.get_integrated_mask(
                image,
                baseline=baseline,
                steps=steps,
                output_of_interest=output_of_interest,
            )
            if np.ndim(gradient) == 4 and gradient.shape[0] == 1:
                gradient = gradient[0]
            gradients.append(gradient)
        return gradients

    def __make_baselines(self, x_value, x_baselines):
        if x_baselines is None:
            x_baselines = []
            x_baselines.append(np.min(x_value) * np.ones_like(x_value))
            x_baselines.append(np.max(x_value) * np.ones_like(x_value))
        else:
            for baseline in x_baselines:
                if baseline.shape != x_value.shape:
                    raise ValueError("Baseline size {} does not match input size {}".format(baseline.shape, x_value.shape))
        return x_baselines

    def get_mask(self,
                 x_value,
                 baselines=None,
                 segments=None,
                 base_attribution=None,
                 return_ig_attributions=False,
                 return_attr_data=True,
                 return_gains=False,
                 output_of_interest=-1,
                 verbose=False,
                 **kwargs
                 ):

        min_mask_difference = kwargs['min_mask_difference'] if 'min_mask_difference' in kwargs else self.min_mask_difference
        flatten_segments = kwargs['flatten_segments'] if 'flatten_segments' in kwargs else self.flatten_segments
        use_fast_xrai = kwargs['use_fast_xrai'] if 'use_fast_xrai' in kwargs else self.use_fast_xrai

        if verbose:
            print("Getting baseline gradients...")
        if base_attribution is not None:
            base_attribution = np.array(base_attribution)
            if base_attribution.shape != x_value.shape:
                raise ValueError("Base attribution shape {} does not match input shape {}".format(base_attribution.shape, x_value.shape))
            x_baselines = None
            attrs = base_attribution
            attr = base_attribution
        else:
            x_baselines = self.__make_baselines(x_value, baselines)
            attrs = self.__get_integrated_gradients(
                x_value,
                x_baselines,
                self.steps,
                output_of_interest=output_of_interest
            )
            attr = np.mean(np.stack(attrs, axis=0), axis=0)
        attr = attr.max(axis=-1)

        if verbose:
            print("Getting image segments...")
        if segments is None:
            if np.ndim(x_value) == 4 and x_value.shape[0] == 1:
                x_value = x_value[0]
            segments = get_felzenszwalb(x_value,
                                        rescaled_size=self.felzenszwalb_rescale,
                                        sigma_values=self.felzenszwalb_sigmas
                                        )
        if verbose:
            print("{} segments found...".format(len(segments)))

        if use_fast_xrai:
            attr_map, attr_data = self.__xrai_fast(
                attr,
                segments,
                min_mask_difference=min_mask_difference,
                integer_segments=flatten_segments,
                verbose=verbose
            )
        else:
            attr_map, attr_data = self.__xrai(
                attr,
                segments,
                min_mask_difference=min_mask_difference,
                integer_segments=flatten_segments,
                verbose=verbose
            )

        if return_attr_data:
            return attr_map, attr_data
        if return_ig_attributions:
            return attr_map, attrs
        else:
            return attr_map

    def __xrai(self,
               attr,
               segments,
               min_mask_difference=50,
               integer_segments=True,
               return_gains=False,
               verbose=False,
               **kwargs
               ):
        output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)
        n_masks = len(segments)
        current_area_perc = 0.0
        current_mask = np.zeros(attr.shape, dtype=np.bool)
        masks_trace = []
        remaining_masks = {index: mask for index, mask in enumerate(segments)}

        if verbose:
            print("Finding gain masks...")
        added_masks_count = 1
        pbar = tqdm.tqdm(total=self.coverage, leave=verbose)
        while current_area_perc <= self.coverage:
            best_gain = [-np.inf, None]
            remove_queue = []
            for mask_key in remaining_masks:
                mask = remaining_masks[mask_key]
                mask_pixel_diff = np.logical_and(mask, np.logical_not(current_mask)).sum()
                if mask_pixel_diff < min_mask_difference:
                    remove_queue.append(mask_key)
                    continue
                gain = self.gain_function(mask, attr, mask_2=current_mask)
                if gain > best_gain[0]:
                    best_gain = [gain, mask_key]
            for key in remove_queue:
                del remaining_masks[key]
            if len(remaining_masks) == 0:
                break
            added_mask = remaining_masks[best_gain[1]]
            mask_diff = np.logical_and(added_mask, np.logical_not(current_mask))
            masks_trace.append((mask_diff, best_gain[0]))

            current_mask = np.logical_or(current_mask, added_mask)
            increase_perc = np.mean(current_mask) - current_area_perc
            pbar.update(increase_perc)
            current_area_perc += increase_perc
            output_attr[mask_diff] = best_gain[0]
            del remaining_masks[best_gain[1]]
            added_masks_count += 1
        pbar.close()

        uncomputed_mask = output_attr == -np.inf
        output_attr[uncomputed_mask] = self.gain_function(uncomputed_mask, attr)
        if return_gains:
            return output_attr, masks_trace
        masks_trace = [val[0] for val in sorted(masks_trace, key=lambda x: -x[1])]
        if np.any(uncomputed_mask):
            masks_trace.append(uncomputed_mask)

        output_attr = output_attr.astype(self.dtype)
        if integer_segments:
            attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)
            for i, mask in enumerate(masks_trace):
                attr_ranks[mask] = i + 1
            return output_attr, attr_ranks
        else:
            return output_attr, masks_trace

    def __xrai_fast(
        self,
        attr,
        segments,
        min_mask_difference=50,
        integer_segments=True,
        verbose=False,
        **kwargs
    ):
        output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)
        n_masks = len(segments)
        current_mask = np.zeros(attr.shape, dtype=np.bool)

        masks_trace = []

        if verbose:
            print('Finding gain masks...')
        seg_attrs = [gain_fun(segment_mask, attr) for segment_mask in segments]
        segments, seg_attrs = list(
            zip(*sorted(zip(segments, seg_attrs), key=lambda x: -x[1]))
        )
        for i, added_mask in enumerate(segments):
            mask_diff = np.logical_and(added_mask, np.logical_not(current_mask))
            mask_pixel_diff = mask_diff.sum()
            if mask_pixel_diff < min_mask_difference:
                continue
            mask_gain = self.gain_function(mask_diff, attr)
            masks_trace.append((mask_diff, mask_gain))
            output_attr[mask_diff] = mask_gain
            current_mask = np.logical_or(current_mask, added_mask)
        uncomputed_mask = output_attr == -np.inf
        output_attr[uncomputed_mask] = self.gain_function(uncomputed_mask, attr)
        masks_trace = [val[0] for val in sorted(masks_trace, key=lambda x: -x[1])]
        if np.any(uncomputed_mask):
            masks_trace.append(uncomputed_mask)

        output_attr = output_attr.astype(self.dtype)
        if integer_segments:
            attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)
            for i, mask in enumerate(masks_trace):
                attr_ranks[mask] = i + 1
            return output_attr, attr_ranks
        else:
            return output_attr, masks_trace
