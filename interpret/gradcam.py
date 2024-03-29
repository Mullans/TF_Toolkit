from .gradients import simple_gradients, integrated_gradients
import tensorflow as tf


def GradCAM(model, input_image, layer_name, class_of_interest=0, gradient_type='simple', resize_to_input=False, **gradient_kwargs):
    # gradient_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])
    # with tf.GradientTape() as tape:
    #     conv_output, predictions = gradient_model(input_image)
    # grads = tape.gradient(predictions, conv_output)
    if gradient_type == 'simple':
        conv_output, grads, _ = simple_gradients(model, input_image, layer_name, class_of_interest=class_of_interest, **gradient_kwargs)
    elif gradient_type == 'integrated':
        conv_output, grads, _ = integrated_gradients(model, input_image, layer_name, class_of_interest=class_of_interest, **gradient_kwargs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    if resize_to_input:
        if heatmap.ndim == 2:
            heatmap = tf.expand_dims(heatmap, -1)
        return tf.image.resize(heatmap, (input_image.shape[1], input_image.shape[2]))
    return heatmap


def GradCAMPlus(model, input_image, layer_name, gradient_type='simple', segmentation=False, resize_to_input=False, class_of_interest=0, **gradient_kwargs):
    """GradCAM++ algorithm for segmentations

    NOTE
    ----
    This assumes a segmentation model with [0, 1] range outputs

    Adapted from: https://github.com/adityac94/Grad_CAM_plus_plus/blob/master/misc/utils.py
    """
    # logits = model.output if logit_layer_name is None else model.get_layer(logit_layer_name)

    # gradient_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, logits])
    # with tf.GradientTape() as tape:
    #     conv_output, predictions = gradient_model(input_image)
    # grads = tape.gradient(predictions, conv_output)
    if gradient_type == 'simple':
        conv_output, grads, predictions = simple_gradients(model, input_image, layer_name, class_of_interest=class_of_interest, **gradient_kwargs)
    elif gradient_type == 'integrated':
        conv_output, grads, predictions = integrated_gradients(model, input_image, layer_name, class_of_interest=class_of_interest, **gradient_kwargs)
    # This is only needed for segmentation
    if segmentation:
        small_pred = tf.image.resize(predictions, grads.shape[1:3])
    else:
        small_pred = predictions
    # small_pred should be class-specific score
    first_deriv = tf.exp(small_pred) * grads
    second_deriv = first_deriv * grads
    third_deriv = second_deriv * grads

    global_sum = tf.reduce_sum(conv_output, axis=(0, 1, 2), keepdims=True)
    weights = tf.maximum(first_deriv, 0.0)
    alpha_denom = second_deriv * 2.0 + third_deriv * global_sum
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, 1)
    alphas = tf.divide(second_deriv, alpha_denom)

    weights = tf.maximum(first_deriv, 0.0)
    alphas_thresholding = tf.where(weights != 0, alphas, 0.0)
    alpha_normalization_constant = tf.reduce_sum(alphas_thresholding, axis=(0, 1), keepdims=True)
    alpha_normalization_constant = tf.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, 1)
    alphas /= alpha_normalization_constant

    deep_linearization_weights = tf.reduce_sum(weights * alphas, axis=(0, 1, 2), keepdims=True)
    heatmap = tf.reduce_sum(deep_linearization_weights * conv_output, axis=(0, 3))
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    if resize_to_input:
        if heatmap.ndim == 2:
            heatmap = tf.expand_dims(heatmap, -1)
        return tf.image.resize(heatmap, (input_image.shape[1], input_image.shape[2]))
    return heatmap
