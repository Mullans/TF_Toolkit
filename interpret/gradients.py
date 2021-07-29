import numpy as np
import tensorflow as tf
from ..utils import enforce_4D, rescale


def simple_gradients(model, input_image, layer_name, logit_layer_name=None, **kwargs):
    logits = model.output if logit_layer_name is None else model.get_layer(logit_layer_name)

    gradient_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, logits])
    with tf.GradientTape() as tape:
        conv_output, predictions = gradient_model(input_image)
    grads = tape.gradient(predictions, conv_output)
    return conv_output, grads, predictions


def integrated_gradients(model, input_image, layer_name, num_steps=25, baseline_image=None, logit_layer_name=None, max_batch_size=16, **kwargs):
    """

    Axiomatic Attribution for Deep Networks
    Sundararajan et al. 2017
    """
    logits = model.output if logit_layer_name is None else model.get_layer(logit_layer_name)
    input_image = enforce_4D(input_image)

    gradient_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, logits])
    baseline_image = tf.zeros_like(input_image) if baseline_image is None else enforce_4D(baseline_image)

    diff_image = input_image - baseline_image
    all_grads = None
    steps = tf.linspace(0.0, 1.0, num_steps)
    for min_idx in range(0, num_steps, max_batch_size):
        max_idx = min(min_idx + max_batch_size, num_steps)
        step_images = baseline_image + tf.concat([steps[inner_idx] * diff_image for inner_idx in range(min_idx, max_idx)], axis=0)
        with tf.GradientTape() as tape:
            conv_output, model_output = gradient_model(step_images)
        grads = tape.gradient(model_output, conv_output)
        grads = tf.reduce_sum(grads, axis=0, keepdims=True)
        all_grads = grads if all_grads is None else all_grads + grads
    conv_output, model_output = gradient_model(input_image)
    return conv_output, tf.nn.relu(all_grads / num_steps), model_output


def integrated_gradients_to_input(model, input_image, num_steps=25, baseline_image=None, logit_layer_name=None, max_batch_size=16, **kwargs):
    """Integrated gradients, but specific to the input"""
    input_image = enforce_4D(input_image)

    baseline_image = tf.zeros_like(input_image) if baseline_image is None else enforce_4D(baseline_image)
    diff_image = input_image - baseline_image
    all_grads = None
    steps = tf.linspace(0.0, 1.0, num_steps)
    for min_idx in range(0, num_steps, max_batch_size):
        max_idx = min(min_idx + max_batch_size, num_steps)
        step_images = baseline_image + tf.concat([steps[inner_idx] * diff_image for inner_idx in range(min_idx, max_idx)], axis=0)
        with tf.GradientTape() as tape:
            tape.watch(step_images)
            model_output = model(step_images)
        grads = tape.gradient(model_output, step_images)
        grads = tf.reduce_sum(grads, axis=0, keepdims=True)
        all_grads = grads if all_grads is None else all_grads + grads
    return rescale(np.squeeze(tf.nn.relu(all_grads / num_steps)))


def smooth_gradients(model, input_image, layer_name, num_steps=25, noise_percent=0.15, baseline_image=None, logit_layer_name=None, max_batch_size=16, **kwargs):
    """Work in progress...

    SmoothGrad: removing noise by adding noise
    Smilkov et al 2017
    """
    logits = model.output if logit_layer_name is None else model.get_layer(logit_layer_name)
    input_image = enforce_4D(input_image)
    noise_stddev = (tf.reduce_max(input_image) - tf.reduce_min(input_image)) * noise_percent
    gradient_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, logits])
    all_grads = None
    for idx in range(0, num_steps, max_batch_size):
        group_size = min(idx + max_batch_size, num_steps) - idx
        step_images = tf.tile(input_image, [group_size, 1, 1, 1])
        step_images += tf.random.normal(step_images.shape, 0, noise_stddev)
        with tf.GradientTape() as tape:
            conv_output, model_output = gradient_model(step_images)
        grads = tape.gradient(model_output, conv_output)
        grads = tf.reduce_sum(grads, axis=0, keepdims=True)
        all_grads = grads if all_grads is None else all_grads + grads
    conv_output, model_output = gradient_model(input_image)
    return conv_output, all_grads / num_steps, model_output


def guided_integrated_gradients(model, input_image, num_steps=25, percentile=10, baseline_image=None, **kwargs):
    """Currently a work in progress:
        Only works based on the input, not on intermediate layers
        Doesn't seem to output anything useful for segmentation models... implementation error?

    Guided Integrated Gradients: an Adaptive Path Method for Removing Noise
    Kapishnikov et al 2021
    """
    input_image = enforce_4D(input_image)
    baseline_image = tf.zeros_like(input_image) if baseline_image is None else enforce_4D(baseline_image)

    X_i = input_image
    X_b = baseline_image

    d_total = tf.reduce_sum(tf.abs(X_b, X_i))
    x = X_b
    attr = tf.zeros_like(X_i)

    for t in range(1, num_steps + 1):
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = model(x)
        y = tape.gradient(output, x)
        while True:
            y = tf.where(x == X_i, np.inf, y)
            d_target = d_total * (1 - t / num_steps)
            d_current = tf.reduce_sum(tf.abs(x, X_i))
            if d_target == d_current:
                print('break1')
                break
            p_val = np.percentile(tf.abs(y), percentile)
            S = tf.abs(y) <= p_val
            d_S = tf.reduce_sum(tf.where(S, tf.abs(x - X_i), 0))
            delta = (d_current - d_target) / d_S
            temp = tf.identity(x)
            if delta > 1:
                x = tf.where(S, X_i, x)
            else:
                x = tf.where(S, (1 - delta) * x + delta * X_i, x)
            y = tf.where(tf.math.is_inf(y), 0, y)
            attr = attr + tf.where(S, (x - temp) * y, 0)
            if delta <= 1:
                tf.print(delta)
                break
    return attr
