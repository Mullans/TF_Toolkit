import tensorflow as tf
from ..utils import rescale


def ScoreCAM(model, input_image, layer_name, class_of_interest=None, resize_to_input=False, max_batch_size=8):
    """An approximate adaptation of ScoreCAM for segmentation models

    Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
    Wang et al. 2020
    """
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])
    activations, model_output = activation_model(input_image)
    if class_of_interest is not None:
        model_output = model_output[..., class_of_interest]

    scaled_activations = rescale(tf.nn.relu(activations), axis=(0, 1, 2), keepdims=True)
    resized_activations = tf.image.resize(scaled_activations, input_image.shape[1:3], method='bilinear')
    masked_inputs = tf.transpose(input_image * resized_activations, [3, 1, 2, 0])
    weights = []
    for idx in range(0, masked_inputs.shape[0], max_batch_size):
        masked_predictions = model(masked_inputs[idx:idx + max_batch_size])
        if class_of_interest is not None:
            masked_predictions = masked_predictions[..., class_of_interest]
        # This is the difference... no real analog to increase in confidence for segmentation
        if model_output.ndim > 1:
            weights.append(tf.reduce_sum(masked_predictions - model_output, axis=(1, 2, 3)))
        else:
            weights.append(masked_predictions - model_output)
    weights = rescale(tf.concat(weights, 0))
    attr_map = tf.reduce_sum(activations * weights, axis=(0, 3), keepdims=True)
    if resize_to_input:
        if attr_map.ndim == 2:
            attr_map = tf.expand_dims(attr_map, -1)
        return tf.image.resize(attr_map, (input_image.shape[1], input_image.shape[2]))
    return attr_map
