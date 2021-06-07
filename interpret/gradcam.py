import tensorflow as tf


def GradCAM(model, input_image, layer_name, resize_to_input=False):
    gradient_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = gradient_model(input_image)
    grads = tape.gradient(predictions, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    if resize_to_input:
        return tf.image.resize(heatmap, (input_image.shape[2], input_image.shape[1]))
    return heatmap
