import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, GlobalAveragePooling2D, Layer, ReLU


# Templates
class TemplateLayer(Layer):
    def __init__(self, name='template', batchnorm=True, activation=ReLU):
        super().__init__(name=name)
        self.batchnorm = BatchNormalization(name=name + '/bn') if batchnorm else None
        if activation is None:
            self.activation = None
        elif isinstance(activation, str):
            self.activation = Activation(activation, name=name + '/act')
        else:
            self.activation = activation(name=name + '/act')

    def call(self, x, training=True):
        x = self.main_layer(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x


# Activations
class Mish(Layer):
    def __init__(self, name='mish_activation'):
        super().__init__(name=name)

    def call(self, x, *args, **kwargs):
        return x * tf.math.tanh(tf.math.softplus(x))


# Layers
class ConvLayer2D(TemplateLayer):
    def __init__(self, filters, kernel=(3, 3), strides=(2, 2), padding='same', batchnorm=True, activation=ReLU, name='conv_layer', **kwargs):
        super().__init__(batchnorm=batchnorm, activation=activation, name=name)
        self.main_layer = Conv2D(filters, kernel, strides=strides, padding=padding, name=name, **kwargs)


class DeconvLayer2D(TemplateLayer):
    def __init__(self, filters, kernel=(3, 3), strides=(2, 2), padding='same', batchnorm=True, activation=ReLU, name='deconv_layer', **kwargs):
        super().__init__(batchnorm=batchnorm, activation=activation, name=name)
        self.main_layer = Conv2DTranspose(filters, kernel, strides=strides, padding=padding, **kwargs)


class DenseLayer(TemplateLayer):
    def __init__(self, nodes, batchnorm=True, activation=ReLU, name='dense_layer'):
        super().__init__(batchnorm=batchnorm, activation=activation, name=name)
        self.main_layer = Dense(nodes, name=name)


class DualAttentionLayer2D(Layer):
    def __init__(self, num_input_filters, squeeze_ratio=2, name='attention_layer'):
        super().__init__(name=name)
        squeeze_channels = max(num_input_filters // squeeze_ratio, 1)
        self.channel_pool = GlobalAveragePooling2D(name=name + '/channel_GAP')
        self.channel_squeeze = Dense(squeeze_channels, activation='relu', name=name + '/channel_dense_squeeze')
        self.channel_expand = Dense(num_input_filters, activation='sigmoid', name=name + '/channel_dense_expand')
        self.spatial_squeeze_1 = Conv2D(max(num_input_filters // 2, 1), (1, 1), padding='same', activation='relu', name=name + '/spatial_squeeze_1')
        self.spatial_squeeze_2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name=name + '/spatial_squeeze_2')

    def call(self, x, *args, **kwargs):
        channel_block = self.channel_pool(x)
        channel_block = self.channel_squeeze(channel_block, *args, **kwargs)
        channel_block = self.channel_expand(channel_block, *args, **kwargs)
        channel_block = tf.keras.layers.multiply([x, channel_block])

        spatial_block = self.spatial_squeeze_1(x, *args, **kwargs)
        spatial_block = self.spatial_squeeze_2(spatial_block, *args, **kwargs)
        return tf.keras.layers.multiply([channel_block, spatial_block])


# Combination Blocks
class DoubleConvBlock(Layer):
    def __init__(self, *args, name='conv_layer', **kwargs):
        super().__init__(name=name)
        self.conv1 = ConvLayer2D(*args, strides=(1, 1), name=name + '/conv1', **kwargs)
        self.conv2 = ConvLayer2D(*args, strides=(2, 2), name=name + '/conv2', **kwargs)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DoubleDeconvBlock(Layer):
    def __init__(self, filters, *args, name='deconv_layer', use_attention=True, **kwargs):
        super().__init__(name=name)
        self.deconv1 = DeconvLayer2D(filters, *args, strides=(2, 2), name=name + '/deconv1', **kwargs)
        self.deconv2 = ConvLayer2D(filters, *args, strides=(1, 1), name=name + '/deconv2', **kwargs)
        self.attention = DualAttentionLayer2D(filters, squeeze_ratio=2, name=name + '/attention') if use_attention else None

    def call(self, x, skip_connection=None):
        x = self.deconv1(x)
        if skip_connection is not None:
            x = tf.concat([x, skip_connection], -1)
        x = self.deconv2(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


# Models
class EncoderModel(Model):
    def __init__(self, filter_scale=0, num_layers=5, forward_connections=0, activation=Mish, name='encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer_list = [
            DoubleConvBlock(2 ** (filter_scale + 3 + i),
                            activation=activation,
                            name=name + '/conv_layer_{}'.format(i + 1)) for i in range(num_layers)]
        self.forward_connections = forward_connections

    def call(self, x, *args, **kwargs):
        layer_results = []
        N = len(self.layer_list)
        for i, layer in enumerate(self.layer_list):
            x = layer(x, *args, **kwargs)
            if i > N - self.forward_connections - 1:
                layer_results.append(x)
        return x, layer_results


class DecoderModel(Model):
    def __init__(self, filter_scale=0, output_channels=1, num_layers=5 + 1, use_attention=False, name='decoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer_list = [
            DoubleDeconvBlock(2 ** (filter_scale + 6 - i),
                              use_attention=use_attention,
                              name=name + '/deconv_layer_{}'.format(i + 1)) for i in range(num_layers - 1)]
        self.output_layer = ConvLayer2D(output_channels, strides=(1, 1), activation='sigmoid', name=name + '/output_conv')

    def call(self, x, skip_layers=[]):
        for i, layer in enumerate(self.layer_list):
            skip_connection = skip_layers[i] if i < len(skip_layers) else None
            x = layer.call(x, skip_connection=skip_connection)
        x = self.output_layer(x)
        return x
