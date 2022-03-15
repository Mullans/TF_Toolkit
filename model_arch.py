import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPool2D, GlobalAveragePooling3D, MaxPooling3D, LeakyReLU, Multiply, ReLU, DepthwiseConv2D, AveragePooling2D, Softmax, ZeroPadding2D)
from tensorflow.keras.regularizers import L2


# Layer Definitions
def dense_layer(layer_input, nodes, batchnorm=True, activation=ReLU, name='dense'):
    dense = Dense(nodes, name=name)(layer_input)
    if batchnorm:
        dense = BatchNormalization(name=name + '/bn')(dense)
    if activation is not None:
        dense = activation(name=name + '/act')(dense)
    return dense


def conv_layer(layer_input, filters, kernel=(3, 3), strides=(2, 2), padding='same', use_bias=True, batchnorm=True, regularize=False, activation=ReLU, name='conv'):
    regularize_weight = 1e-4 if isinstance(regularize, bool) else regularize
    conv = Conv2D(filters,
                  kernel,
                  strides=strides,
                  padding=padding,
                  use_bias=use_bias,
                  kernel_regularizer=L2(regularize_weight) if regularize else None,
                  name=name)(layer_input)
    if batchnorm:
        conv = BatchNormalization(name=name + '/bn')(conv)
    if activation is not None:
        if isinstance(activation, str):
            conv = Activation(activation, name=name + '/act')(conv)
        else:
            conv = activation(name=name + '/act')(conv)
    return conv


def conv_layer3d(layer_input, filters, kernel=(3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=True, batchnorm=True, activation=ReLU, name='conv3d'):
    conv = Conv3D(filters, kernel, strides=strides, padding=padding, use_bias=use_bias, name=name)(layer_input)
    if batchnorm:
        conv = BatchNormalization(name=name + '/bn')(conv)
    if activation is not None:
        conv = activation(name=name + '/act')(conv)
    return conv


def se_block(layer_input, num_channels, ratio=16, name='se_block'):
    """Squeeze-excitation block"""
    block = GlobalAveragePooling2D(name=name + '/GAP')(layer_input)
    squeeze_channels = max(num_channels // ratio, 1)
    block = Dense(squeeze_channels, activation='relu', name=name + '/dense1')('block')
    block = Dense(num_channels, activation='sigmoid', name=name + '/dense2')(block)
    return Multiply(name=name + '/multiply')([layer_input, block])


def se_block3D(layer_input, ratio=16, name='se_block'):
    """Squeeze-excitation block, extended to 3D"""
    num_channels = layer_input.shape[-1]
    block = GlobalAveragePooling3D(name=name + '/GAP')(layer_input)
    squeeze_channels = max(num_channels // ratio, 1)
    block = Dense(squeeze_channels, activation='relu', name=name + '/dense1')(block)
    block = Dense(num_channels, activation='sigmoid', name=name + '/dense2')(block)
    return Multiply(name=name + '/multiply')([layer_input, block])


def dual_attention_block(layer_input, ratio=16, name='sau_block'):
    """Dual attention decoder block from SAUNet https://arxiv.org/pdf/2001.07645.pdf"""
    num_channels = layer_input.shape[-1]
    squeeze_channels = max(num_channels // ratio, 1)
    channel_block = GlobalAveragePooling2D(name=name + '/channel_GAP')(layer_input)
    channel_block = Dense(squeeze_channels, activation='relu', name=name + '/channel_dense1')(channel_block)
    channel_block = Dense(num_channels, activation='sigmoid', name=name + '/channel_dense2')(channel_block)
    channel_block = Multiply(name=name + '/channel_multiply')([layer_input, channel_block])

    spatial_block = Conv2D(num_channels // 2, (1, 1), padding='same', activation='relu', name=name + '/spatial_conv1')(layer_input)
    spatial_block = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name=name + '/spatial_conv2')(spatial_block)
    return Multiply(name=name + '/spatial_multiply')([channel_block, spatial_block])


def dual_attention_blockv2(layer_input, ratio=16, name='sau_blockv2'):
    """Dual attention decoder block from SAUNet https://arxiv.org/pdf/2001.07645.pdf

    NOTE
    ----
    The v2 block uses a mask of the sigmoid + 1 so that signal is only enhanced, not suppressed. The other block uses a mask of just the sigmoid so that signal is only suppressed, not enhanced."""
    num_channels = layer_input.shape[-1]
    squeeze_channels = max(num_channels // ratio, 1)
    channel_block = GlobalAveragePooling2D(name=name + '/channel_GAP')(layer_input)
    channel_block = Dense(squeeze_channels, activation='relu', name=name + '/channel_dense1')(channel_block)
    channel_block = Dense(num_channels, activation='sigmoid', name=name + '/channel_dense2')(channel_block) + 1
    channel_block = Multiply(name=name + '/channel_multiply')([layer_input, channel_block])

    spatial_block = Conv2D(num_channels // 2, (1, 1), padding='same', activation='relu', name=name + '/spatial_conv1')(layer_input)
    spatial_block = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name=name + '/spatial_conv2')(spatial_block) + 1
    return Multiply(name=name + '/spatial_multiply')([channel_block, spatial_block])


def dual_attention_block3D(layer_input, ratio=16, name='sau_block'):
    """Dual attention decoder block from SAUNet, extended to 3D https://arxiv.org/pdf/2001.07645.pdf"""
    num_channels = layer_input.shape[-1]
    squeeze_channels = max(num_channels // ratio, 1)
    channel_block = GlobalAveragePooling3D(name=name + '/channel_GAP')(layer_input)
    channel_block = Dense(squeeze_channels, activation='relu', name=name + '/channel_dense1')(channel_block)
    channel_block = Dense(num_channels, activation='sigmoid', name=name + '/channel_dense2')(channel_block)
    channel_block = Multiply(name=name + '/channel_multiply')([layer_input, channel_block])

    spatial_block = Conv3D(num_channels // 2, (1, 1, 1), padding='same', activation='relu', name=name + '/spatial_conv1')(layer_input)
    spatial_block = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid', name=name + '/spatial_conv2')(spatial_block)
    return Multiply(name=name + '/spatial_multiply')([channel_block, spatial_block])


def dual_attention_block3Dv2(layer_input, ratio=16, name='sau_blockv2'):
    """Dual attention decoder block from SAUNet, extended to 3D https://arxiv.org/pdf/2001.07645.pdf

    NOTE
    ----
    The v2 block uses a mask of the sigmoid + 1 so that signal is only enhanced, not suppressed. The other block uses a mask of just the sigmoid so that signal is only suppressed, not enhanced.
    """
    num_channels = layer_input.shape[-1]
    squeeze_channels = max(num_channels // ratio, 1)
    channel_block = GlobalAveragePooling3D(name=name + '/channel_GAP')(layer_input)
    channel_block = Dense(squeeze_channels, activation='relu', name=name + '/channel_dense1')(channel_block)
    channel_block = Dense(num_channels, activation='sigmoid', name=name + '/channel_dense2')(channel_block) + 1
    channel_block = Multiply(name=name + '/channel_multiply')([layer_input, channel_block])

    spatial_block = Conv3D(num_channels // 2, (1, 1, 1), padding='same', activation='relu', name=name + '/spatial_conv1')(layer_input)
    spatial_block = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid', name=name + '/spatial_conv2')(spatial_block) + 1
    return Multiply(name=name + '/spatial_multiply')([channel_block, spatial_block])


def deconv_layer(layer_input, filters, kernel=(3, 3), strides=(2, 2), padding='same', use_bias=True, batchnorm=True, regularize=False, activation=ReLU, name='deconv'):
    regularize_weight = 1e-4 if isinstance(regularize, bool) else regularize
    conv = Conv2DTranspose(filters,
                           kernel_size=kernel,
                           strides=strides,
                           padding=padding,
                           use_bias=use_bias,
                           kernel_regularizer=L2(regularize_weight) if regularize else None,
                           name=name)(layer_input)
    if batchnorm:
        conv = BatchNormalization(name=name + '/bn')(conv)
    if activation is not None:
        if isinstance(activation, str):
            conv = Activation(activation, name=name + '/act')(conv)
        else:
            conv = activation(name=name + '/act')(conv)
    return conv


def deconv_layer3d(layer_input, filters, kernel=(3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=True, batchnorm=True, activation=ReLU, name='deconv3d'):
    conv = Conv3DTranspose(filters, kernel, strides=strides, padding=padding, use_bias=use_bias, name=name)(layer_input)
    if batchnorm:
        conv = BatchNormalization(name=name + '/bn')(conv)
    if activation is not None:
        conv = activation(name=name + '/act')(conv)
    return conv


def dense_block3d(layer_input, filters_per_layer, num_layers, kernel=(3, 3, 3), name='dense_block'):
    """Denseblock from Densenet 121, extended to 3D"""
    current_input = layer_input
    values = [layer_input]
    layer = BatchNormalization(name=name + '/bn0')(current_input)
    layer = ReLU(name=name + '/act0')(layer)
    layer = Conv3D(filters_per_layer, kernel, padding='same', name=name + '/conv0')(layer)
    layer = Dropout(0.2, name=name + '/drop0')(layer)
    values.append(layer)
    current_input = Concatenate(axis=-1, name='/input_concat0')(values)
    for i in range(1, num_layers):
        layer = BatchNormalization(name=name + '/bn{}'.format(i))(current_input)
        layer = ReLU(name=name + '/act{}'.format(i))(layer)
        layer = Conv3D(filters_per_layer, kernel, padding='same', name=name + '/conv{}'.format(i))(layer)
        layer = Dropout(0.2, name=name + '/drop{}'.format(i))(layer)
        values.append(layer)
        current_input = Concatenate(axis=-1, name=name + '/input_concat{}'.format(i))(values)
    return Concatenate(axis=-1, name=name + '/output_concat')(values[1:])


def transition_down3d(layer_input, pool_size=(2, 2, 2), name='transition_down'):
    """Transition Down block from Densenet 121, extended to 3D"""
    num_layers = layer_input.shape[-1]
    td = BatchNormalization(name=name + '/bn')(layer_input)
    td = ReLU(name=name + '/act')(td)
    td = Conv3D(num_layers, (1, 1, 1), name=name + '/pointconv')(td)
    td = Dropout(0.2, name=name + '/dropout')(td)
    td = MaxPooling3D(pool_size=pool_size, padding='same', name=name + '/maxpool')
    return td


def spatial_attention_layer3d(layer_input, filters, kernel=(3, 3, 3), padding='same', use_bias=True, batchnorm=True, activation=ReLU, name='spatialConv3d'):
    common_args = {'strides': (1, 1, 1), 'padding': padding, 'use_bias': use_bias, 'batchnorm': batchnorm, 'activation': activation}
    input_conv = conv3d_layer(layer_input, filters, kernel=kernel, name=name + '/input_conv', **common_args)

    spatial_path = conv3d_layer(input_conv, filters, kernel=(1, 1, 1), name='/spatial_conv1', **common_args)
    spatial_path = conv3d_layer(spatial_path, filters, kernel=(1, 1, 1), name='/spatial_conv2', **common_args)
    spatial_path = Activation('sigmoid', name='spatial_activation')(spatial_path)

    # TODO: add squeeze path once that's figured out
    output_conv = tf.multiply(input_conv, spatial_path)
    return output_conv


def separable_conv_layer(layer_input, filters, kernel=(3, 3), strides=(2, 2), norm_scaling=False, padding='same', use_bias=False, name='sepconv'):
    conv1 = DepthwiseConv2D(kernel, strides=strides, depth_multiplier=1, padding=padding, use_bias=use_bias, name=name + '/depthConv')(layer_input)
    norm1 = BatchNormalization(center=True, scale=norm_scaling, epsilon=1e-4, name=name + '/depthConv/bn')(conv1)
    act1 = ReLU(name=name + '/depthConv/act')(norm1)
    conv2 = Conv2D(filters, (1, 1), strides=1, padding=padding, use_bias=use_bias, name=name + '/pointConv')(act1)
    norm2 = BatchNormalization(center=True, scale=norm_scaling, epsilon=1e-4, name=name + '/pointConv/bn')(conv2)
    act2 = ReLU(name=name + '/pointConv/act')(norm2)
    return act2


def pad_and_merge2D(layers, merge_op=Concatenate, name='pad_and_merge'):
    with tf.name_scope(name) as scope:
        max_size = np.array([-1, -1])
        for item in layers:
            max_size = np.maximum(item.shape[1:3], max_size)
        out_layers = []
        for idx, item in enumerate(layers):
            total_pad = max_size - item.shape[1:3]
            front_pad = total_pad // 2
            back_pad = total_pad - front_pad
            padding = ((front_pad[0], back_pad[0]), (front_pad[1], back_pad[1]))
            item = ZeroPadding2D(padding=padding, name='padding_{}'.format(idx))(item)
            out_layers.append(item)
        return merge_op(name='merge')(out_layers)


# Classification Models
def yamnet(input_shape=[32, 64, 1], num_outputs=1, scale=0, mid_size=5, out_pool=False, norm_scaling=False, **kwargs):
    """Copy of the yamnet architecture for audio classification

    Parameters
    ----------
    input_shape : list of ints
        The shape of the model input (the default is [32, 64, 1])
    num_outputs : int
        The number of outputs from the final layer of the model
    scale : int
        Scaling factor for convolutional layers by powers of 2 (the default is 0)
    out_pool : bool
        Whether to apply average pooling to the last convolutional layer, otherwise a large kernel convolution is used (the default is False)
    norm_scaling : bool
        Whether to use scaling in the batch normalizations (the default is False)
    """
    inputs = Input(shape=input_shape, name='inputs')

    conv1 = conv_layer(inputs, 2 ** (4 + scale), strides=2, name='conv1')
    conv2 = separable_conv_layer(conv1, 2 ** (5 + scale), strides=1, norm_scaling=norm_scaling, name='sepconv1')
    conv3 = separable_conv_layer(conv2, 2 ** (6 + scale), strides=2, norm_scaling=norm_scaling, name='sepconv2')
    conv4 = separable_conv_layer(conv3, 2 ** (6 + scale), strides=1, norm_scaling=norm_scaling, name='sepconv3')
    conv5 = separable_conv_layer(conv4, 2 ** (7 + scale), strides=2, norm_scaling=norm_scaling, name='sepconv4')

    mid_conv = conv5
    for i in range(mid_size):
        mid_conv = separable_conv_layer(mid_conv, 2 ** (7 + scale), strides=1, norm_scaling=norm_scaling, name='sepconv{}'.format(5 + i))

    conv13 = separable_conv_layer(mid_conv, 2 ** (8 + scale), strides=2, norm_scaling=norm_scaling, name='sepconv10')
    if out_pool:
        pool_size = conv13.shape[1:3]
        conv14 = AveragePooling2D(pool_size=pool_size, strides=1, padding='valid')(conv13)
    else:
        conv14 = conv_layer(conv13, 2 ** (8 + scale), kernel=(2, 4), strides=(2, 4), name='conv_out')

    flat_outputs = Flatten(name='out_flat')(conv14)
    output = Dense(num_outputs, name='outputs')(flat_outputs)
    output = Activation('sigmoid', name='outputs_act')(output)
    return Model(inputs=inputs, outputs=output)


def multires_classifier(out_classes=2, filter_scale=1, dense_size=128, input_shape=[1024, 1360, 1], out_layers=0, out_drop=0.2, out_pool=True, **kwargs):
    """A multi-resolution network made for classifying different neuron types

    Parameters
    ----------
    out_classes : int
        The number of output binary predictions (the default is 2)
    filter_scale : int
        The power of 2 scaling factor to add to convolution layer filter counts (the default is 1)
    dense_size : int
        The number of nodes in the dense layer before the output (the default is 128)
    input_shape : list of ints
        The shape of the input (the default is [1024, 1360, 1])
    out_layers : int
        The number of extra dense layers before the output layer, with size decreasing from dense_size by half for each (the default is 0)
    out_drop : float
        The dropout rate on the dense layer before the output (the default is 0.2)
    out_pool : bool
        Whether to use an average pooling layer between the convolutional layers and dense layers (the default is True)
    """
    inputs = tf.keras.Input(shape=(tuple(input_shape)), name='input_image')

    conv1 = conv_layer(inputs, 2 ** (filter_scale + 4), kernel=5, name='conv1')
    conv1b = conv_layer(conv1, 2 ** (filter_scale + 4), strides=1, name='conv1b')
    down1 = MaxPool2D(pool_size=16, name='down1_pool')(conv1b)
    down1 = conv_layer(down1, 16, strides=1, name='down1_conv')
    down1 = ZeroPadding2D(padding=((0, 0), (1, 0)))(down1)

    conv2 = conv_layer(conv1b, 2 ** (filter_scale + 5), name='conv2')
    conv2b = conv_layer(conv2, 2 ** (filter_scale + 5), strides=1, name='conv2b')
    down2 = MaxPool2D(pool_size=8, name='down2_pool')(conv2b)
    down2 = conv_layer(down2, 16, strides=1, name='down2_conv')
    down2 = ZeroPadding2D(padding=((0, 0), (1, 0)))(down2)

    conv3 = conv_layer(conv2b, 2 ** (filter_scale + 6), name='conv3')
    conv3b = conv_layer(conv3, 2 ** (filter_scale + 6), strides=1, name='conv3b')
    down3 = MaxPool2D(pool_size=4, name='down3_pool')(conv3b)
    down3 = conv_layer(down3, 16, strides=1, name='down3_conv')
    down3 = ZeroPadding2D(padding=((0, 0), (1, 0)))(down3)

    conv4 = conv_layer(conv3b, 2 ** (filter_scale + 7), name='conv4')
    conv4b = conv_layer(conv4, 2 ** (filter_scale + 7), strides=1, name='conv4b')
    down4 = MaxPool2D(pool_size=2, name='down4_pool')(conv4b)
    down4 = conv_layer(down4, 16, strides=1, name='down4_conv')
    down4 = ZeroPadding2D(padding=((0, 0), (1, 0)))(down4)

    conv5 = conv_layer(conv4b, 2 ** (filter_scale + 8), name='conv5')
    conv5b = conv_layer(conv5, 2 ** (filter_scale + 8), strides=1, name='conv5b')
    down5 = conv_layer(conv5b, 64, strides=1, name='down5_conv')

    cat_layer = Concatenate(name='cat_layer')([down1, down2, down3, down4, down5])

    conv_flat = conv_layer(cat_layer, 128, strides=1, name='conv_flat')

    if out_pool:
        pool_size = conv_flat.shape[1:3]
        # conv_flat = AveragePooling2D(pool_size=[32, 43], strides=1)(conv_flat)
        conv_flat = AveragePooling2D(pool_size=pool_size, strides=1)(conv_flat)

    flat = Flatten(name='flatten')(conv_flat)
    dense1 = dense_layer(flat, dense_size, name='dense1')

    for i in range(out_layers - 1):
        dense1 = dense_layer(dense1, dense_size / (2 ** i), name='dense{}'.format(2 + i))
    if out_drop > 0:
        dense1 = Dropout(out_drop, name="last_drop")(dense1)
    output = Dense(out_classes, bias_initializer=Constant(0.), name='output')(dense1)

    output = Activation('sigmoid', name='output_act')(output)
    return Model(inputs=inputs, outputs=output)


# Segmentation Models
def like_unet(input_shape=[1024, 1024, 1], output_channels=1, filter_scale=0, tight_middle=False, **kwargs):
    """A quick model inspired by unet

    Parameters
    ----------
    input_shape : list of ints
        Shape of the input (the default is [1024, 1024, 1])
    filter_scale : int
        The scaling factor for the number of convolutional layer filters (the default is 0)
    tight_middle : bool
        Whether each layer in the middle should concatenate its output to its input
    """
    input_image = tf.keras.Input(shape=input_shape, name='input_image')

    down1 = conv_layer(input_image, 2 ** (3 + filter_scale), strides=2, name='down1')  # -> 512, 8
    down1b = separable_conv_layer(down1, 2 ** (3 + filter_scale), strides=1, name='down1b')

    down2 = separable_conv_layer(down1b, 2 ** (4 + filter_scale), strides=2, name='down2')  # -> 256, 16
    down2b = separable_conv_layer(down2, 2 ** (4 + filter_scale), strides=1, name='down2b')

    down3 = separable_conv_layer(down2b, 2 ** (5 + filter_scale), strides=2, name='down3')  # -> 128, 32
    down3b = separable_conv_layer(down3, 2 ** (5 + filter_scale), strides=1, name='down3b')

    down4 = separable_conv_layer(down3b, 2 ** (6 + filter_scale), strides=2, name='down4')  # -> 64, 64
    down4b = separable_conv_layer(down4, 2 ** (6 + filter_scale), strides=1, name='down4b')

    down5 = separable_conv_layer(down4b, 2 ** (7 + filter_scale), strides=2, name='down5')  # -> 32, 128
    down5b = separable_conv_layer(down5, 2 ** (7 + filter_scale), strides=1, name='down5b')

    mid_layer = down5b
    for i in range(5):
        if tight_middle:
            mid_layer2 = separable_conv_layer(mid_layer, 2 ** (7 + filter_scale), strides=1, name='midlayer_{}'.format(i + 1))
            mid_layer = Concatenate(name='mid_concat_{}'.format(i))([mid_layer, mid_layer2])
        else:
            mid_layer = separable_conv_layer(mid_layer, 2 ** (7 + filter_scale), strides=1, name='midlayer_{}'.format(i + 1))

    uplayer1 = deconv_layer(mid_layer, 2 ** (6 + filter_scale), strides=2, name='up1')  # -> 64, 64
    upconcat1 = Concatenate(name='up_concat_1')([uplayer1, down4b])
    uplayer1b = separable_conv_layer(upconcat1, 2 ** (6 + filter_scale), strides=1, name='up1b')

    uplayer2 = deconv_layer(uplayer1b, 2 ** (5 + filter_scale), strides=2, name='up2')  # -> 128, 32
    upconcat2 = Concatenate(name='up_concat_2')([uplayer2, down3b])
    uplayer2b = separable_conv_layer(upconcat2, 2 ** (5 + filter_scale), strides=1, name='up2b')

    uplayer3 = deconv_layer(uplayer2b, 2 ** (4 + filter_scale), strides=2, name='up3')  # ->256, 16
    upconcat3 = Concatenate(name='up_concat_3')([uplayer3, down2b])
    uplayer3b = separable_conv_layer(upconcat3, 2 ** (4 + filter_scale), strides=1, name='up3b')

    uplayer4 = deconv_layer(uplayer3b, 2 ** (3 + filter_scale), strides=2, name='up4')  # ->512, 8
    upconcat4 = Concatenate(name='up_concat_4')([uplayer4, down1b])
    uplayer4b = separable_conv_layer(upconcat4, 2 ** (3 + filter_scale), strides=1, name='up4b')

    uplayer5 = deconv_layer(uplayer4b, 2 ** (2 + filter_scale), strides=2, name='up5')  # ->1024, 4
    upconcat5 = Concatenate(name='up_concat_5')([uplayer5, input_image])

    output_conv = Conv2D(output_channels, kernel_size=3, padding='same', name='output_conv')(upconcat5)
    output_act = Activation('sigmoid', name='output_act')(output_conv)
    return Model(inputs=input_image, outputs=output_act)


def mobilenet(input_shape=[1024, 1024, 1], filter_scale=0, drop_rate=0.2, dense_out=1024, **kwargs):
    """Covid5K"""
    input_image = tf.keras.Input(shape=input_shape, name='input_image')

    layer1 = conv_layer(input_image, 2 ** (3 + filter_scale), strides=2, name='layer1')
    layer2 = separable_conv_layer(layer1, 2 ** (4 + filter_scale), strides=1, name='layer2')
    layer3 = separable_conv_layer(layer2, 2 ** (5 + filter_scale), strides=2, name='layer3')
    layer4 = separable_conv_layer(layer3, 2 ** (5 + filter_scale), strides=1, name='layer4')
    layer5 = separable_conv_layer(layer4, 2 ** (6 + filter_scale), strides=2, name='layer5')
    layer6 = separable_conv_layer(layer5, 2 ** (6 + filter_scale), strides=1, name='layer6')
    layer7 = separable_conv_layer(layer6, 2 ** (7 + filter_scale), strides=2, name='layer7')

    mid_layers = separable_conv_layer(layer7, 2 ** (7 + filter_scale), strides=1, name='mid_layer_1')
    for i in range(4):
        mid_layers = separable_conv_layer(mid_layers, 2 ** (7 + filter_scale), strides=1, name='mid_layer_{}'.format(i + 2))

    layer8 = separable_conv_layer(mid_layers, 2 ** (8 + filter_scale), strides=2, name='layer8')
    layer9 = separable_conv_layer(layer8, 2 ** (8 + filter_scale), strides=2, name='layer9')

    pool_size = layer9.shape[1:3]

    avg_pool = AveragePooling2D(pool_size=pool_size, strides=1, padding='valid', name='avg_pool')(layer9)
    flat = Flatten(name='flat_layer')(avg_pool)
    dense = dense_layer(flat, dense_out, name='dense_layer')
    if drop_rate > 0:
        flat = Dropout(drop_rate, name='dense_layer/dropout')(dense)
    output = Dense(1, name='output')(dense)
    output = Activation('sigmoid', name='output_act')(output)
    return Model(inputs=input_image, outputs=output)


def combined_networks(input_shape=[512, 512, 1], drop_rate=0.2, dense_out=128, filter_scale=0, tight_middle=False, **kwargs):
    """Covid5K"""
    input_image = tf.keras.Input(shape=input_shape, name='input_image')

    down1 = conv_layer(input_image, 2 ** (3 + filter_scale), strides=2, name='down1')  # -> 512, 8
    down1b = separable_conv_layer(down1, 2 ** (3 + filter_scale), strides=1, name='down1b')

    down2 = separable_conv_layer(down1b, 2 ** (4 + filter_scale), strides=2, name='down2')  # -> 256, 16
    down2b = separable_conv_layer(down2, 2 ** (4 + filter_scale), strides=1, name='down2b')

    down3 = separable_conv_layer(down2b, 2 ** (5 + filter_scale), strides=2, name='down3')  # -> 128, 32
    down3b = separable_conv_layer(down3, 2 ** (5 + filter_scale), strides=1, name='down3b')

    down4 = separable_conv_layer(down3b, 2 ** (6 + filter_scale), strides=2, name='down4')  # -> 64, 64
    down4b = separable_conv_layer(down4, 2 ** (6 + filter_scale), strides=1, name='down4b')

    down5 = separable_conv_layer(down4b, 2 ** (7 + filter_scale), strides=2, name='down5')  # -> 32, 128
    down5b = separable_conv_layer(down5, 2 ** (7 + filter_scale), strides=1, name='down5b')

    mid_layer = down5b
    for i in range(5):
        if tight_middle:
            mid_layer2 = separable_conv_layer(mid_layer, 2 ** (7 + filter_scale), strides=1, name='midlayer_{}'.format(i + 1))
            mid_layer = Concatenate(name='mid_concat_{}'.format(i))([mid_layer, mid_layer2])
        else:
            mid_layer = separable_conv_layer(mid_layer, 2 ** (7 + filter_scale), strides=1, name='midlayer_{}'.format(i + 1))

    uplayer1 = deconv_layer(mid_layer, 2 ** (6 + filter_scale), strides=2, name='up1')  # -> 64, 64
    upconcat1 = Concatenate(name='up_concat_1')([uplayer1, down4b])
    uplayer1b = separable_conv_layer(upconcat1, 2 ** (6 + filter_scale), strides=1, name='up1b')

    uplayer2 = deconv_layer(uplayer1b, 2 ** (5 + filter_scale), strides=2, name='up2')  # -> 128, 32
    upconcat2 = Concatenate(name='up_concat_2')([uplayer2, down3b])
    uplayer2b = separable_conv_layer(upconcat2, 2 ** (5 + filter_scale), strides=1, name='up2b')

    uplayer3 = deconv_layer(uplayer2b, 2 ** (4 + filter_scale), strides=2, name='up3')  # ->256, 16
    upconcat3 = Concatenate(name='up_concat_3')([uplayer3, down2b])
    uplayer3b = separable_conv_layer(upconcat3, 2 ** (4 + filter_scale), strides=1, name='up3b')

    uplayer4 = deconv_layer(uplayer3b, 2 ** (3 + filter_scale), strides=2, name='up4')  # ->512, 8
    upconcat4 = Concatenate(name='up_concat_4')([uplayer4, down1b])
    uplayer4b = separable_conv_layer(upconcat4, 2 ** (3 + filter_scale), strides=1, name='up4b')

    uplayer5 = deconv_layer(uplayer4b, 2 ** (2 + filter_scale), strides=2, name='up5')  # ->1024, 4
    upconcat5 = Concatenate(name='up_concat_5')([uplayer5, input_image])

    output_conv = Conv2D(1, kernel_size=3, padding='same', name='output_conv')(upconcat5)
    output_mask = Activation('sigmoid', name='output_mask/act')(output_conv)

    pool_size = mid_layer.shape[1:3]
    avg_pool = AveragePooling2D(pool_size=pool_size, strides=1, padding='valid', name='avg_pool')(mid_layer)
    flat = Flatten(name='flat_layer')(avg_pool)
    dense = dense_layer(flat, dense_out, name='dense_layer')
    if drop_rate > 0:
        dense = Dropout(drop_rate, name='dense_layer/dropout')(dense)
    output_dense = Dense(1, name='output_dense')(dense)
    output_label = Activation('sigmoid', name='output_label/act')(output_dense)

    return Model(inputs=input_image, outputs=[output_mask, output_label])


def uncombined_network(input_shape=[512, 512, 1], drop_rate=0.2, dense_out=128, filter_scale=0, tight_middle=False, **kwargs):
    """Same as only the label prediction of combined_networks"""
    input_image = tf.keras.Input(shape=input_shape, name='input_image')

    down1 = conv_layer(input_image, 2 ** (3 + filter_scale), strides=2, name='down1')  # -> 512, 8
    down1b = separable_conv_layer(down1, 2 ** (3 + filter_scale), strides=1, name='down1b')

    down2 = separable_conv_layer(down1b, 2 ** (4 + filter_scale), strides=2, name='down2')  # -> 256, 16
    down2b = separable_conv_layer(down2, 2 ** (4 + filter_scale), strides=1, name='down2b')

    down3 = separable_conv_layer(down2b, 2 ** (5 + filter_scale), strides=2, name='down3')  # -> 128, 32
    down3b = separable_conv_layer(down3, 2 ** (5 + filter_scale), strides=1, name='down3b')

    down4 = separable_conv_layer(down3b, 2 ** (6 + filter_scale), strides=2, name='down4')  # -> 64, 64
    down4b = separable_conv_layer(down4, 2 ** (6 + filter_scale), strides=1, name='down4b')

    down5 = separable_conv_layer(down4b, 2 ** (7 + filter_scale), strides=2, name='down5')  # -> 32, 128
    down5b = separable_conv_layer(down5, 2 ** (7 + filter_scale), strides=1, name='down5b')

    mid_layer = down5b
    for i in range(5):
        if tight_middle:
            mid_layer2 = separable_conv_layer(mid_layer, 2 ** (7 + filter_scale), strides=1, name='midlayer_{}'.format(i + 1))
            mid_layer = Concatenate(name='mid_concat_{}'.format(i))([mid_layer, mid_layer2])
        else:
            mid_layer = separable_conv_layer(mid_layer, 2 ** (7 + filter_scale), strides=1, name='midlayer_{}'.format(i + 1))

    pool_size = mid_layer.shape[1:3]
    avg_pool = AveragePooling2D(pool_size=pool_size, strides=1, padding='valid', name='avg_pool')(mid_layer)
    flat = Flatten(name='flat_layer')(avg_pool)
    dense = dense_layer(flat, dense_out, name='dense_layer')
    if drop_rate > 0:
        dense = Dropout(drop_rate, name='dense_layer/dropout')(dense)
    output_dense = Dense(1, name='output_dense')(dense)
    output_label = Activation('sigmoid', name='output_label/act')(output_dense)
    return Model(inputs=input_image, outputs=output_label)


def segnet_3d(input_shape=(8, 256, 256, 1), output_classes=2, filter_scale=0, attention_type=1, **kwargs):
    """3d segmentation network for lungs + lung nodules"""
    inputs = tf.keras.Input(shape=(tuple(input_shape)), name='input_image')

    # Encoder
    if attention_type == 1:
        att_block = dual_attention_block3D
    elif attention_type == 2:
        att_block = dual_attention_block3Dv2
    layer1 = conv_layer3d(inputs, 2 ** (filter_scale + 2), strides=(1, 1, 1), name='conv1')
    layer1b = conv_layer3d(layer1, 2 ** (filter_scale + 2), strides=(2, 2, 2), name='conv1b')

    layer2 = conv_layer3d(layer1b, 2 ** (filter_scale + 3), strides=(1, 1, 1), name='conv2')
    layer2b = conv_layer3d(layer2, 2 ** (filter_scale + 3), strides=(1, 2, 2), name='conv2b')

    layer3 = conv_layer3d(layer2b, 2 ** (filter_scale + 4), strides=(1, 1, 1), name='conv3')
    layer3b = conv_layer3d(layer3, 2 ** (filter_scale + 4), strides=(2, 2, 2), name='conv3b')

    layer4 = conv_layer3d(layer3b, 2 ** (filter_scale + 5), strides=(1, 1, 1), name='conv4')
    layer4b = conv_layer3d(layer4, 2 ** (filter_scale + 5), strides=(1, 2, 2), name='conv4b')

    layer5 = conv_layer3d(layer4b, 2 ** (filter_scale + 6), strides=(1, 1, 1), name='conv5')
    layer5b = conv_layer3d(layer5, 2 ** (filter_scale + 6), strides=(2, 2, 2), name='conv5b')

    # Mid Flow
    layer6 = conv_layer3d(layer5b, 2 ** (filter_scale + 6), strides=(1, 1, 1), name='conv6')
    # layer6b = conv_layer3d(layer6, 2 ** (filter_scale + 5), strides=(1, 1, 1), name='conv6b')

    # Decoder
    layer7 = deconv_layer3d(layer6, 2 ** (filter_scale + 5), strides=(2, 2, 2), name='deconv7')
    layer7 = Concatenate(name='concat7')([layer4b, layer7])
    layer7b = conv_layer3d(layer7, 2 ** (filter_scale + 4), strides=(1, 1, 1), name='conv7b')
    layer7b = att_block(layer7b, ratio=2, name='DAP7b')

    layer8 = deconv_layer3d(layer7b, 2 ** (filter_scale + 4), strides=(1, 2, 2), name='conv8')
    layer8 = Concatenate(name='concat8')([layer3b, layer8])
    layer8b = conv_layer3d(layer8, 2 ** (filter_scale + 3), strides=(1, 1, 1), name='conv8b')
    layer8b = att_block(layer8b, ratio=2, name='DAP8b')

    layer9 = deconv_layer3d(layer8b, 2 ** (filter_scale + 3), strides=(2, 2, 2), name='conv9')
    layer9 = Concatenate(name='concat9')([layer2b, layer9])
    layer9b = conv_layer3d(layer9, 2 ** (filter_scale + 2), strides=(1, 1, 1), name='conv9b')
    layer9b = att_block(layer9b, 1, name='DAP9b')

    layer10 = deconv_layer3d(layer9b, 2 ** (filter_scale + 2), strides=(1, 2, 2), name='conv10')
    layer10 = Concatenate(name='concat10')([layer1b, layer10])
    layer10b = conv_layer3d(layer10, 2 ** (filter_scale + 1), strides=(1, 1, 1), name='conv10b')
    layer10b = att_block(layer10b, 1, name='DAP10b')

    layer11 = deconv_layer3d(layer10b, 2 ** (filter_scale + 1), strides=(2, 2, 2), name='conv11')
    layer11 = Concatenate(name='concat11')([inputs, layer11])
    layer11b = conv_layer3d(layer11, output_classes, strides=(1, 1, 1), activation=None, name='conv11b')
    output_value = Activation('sigmoid', name='output')(layer11b)
    return Model(inputs=inputs, outputs=output_value)


def segnet_2d(input_shape=(800, 640, 3), output_classes=1, filter_scale=0, attention_type=1, **kwargs):
    """2d segmentation network for feet and wounds"""
    inputs = tf.keras.Input(shape=(tuple(input_shape)), name='input_image')

    if attention_type == 1:
        att_block = dual_attention_block
    elif attention_type == 2:
        att_block = dual_attention_blockv2

    # Encoder
    layer1 = conv_layer(inputs, 2 ** (filter_scale + 3), strides=(1, 1), name='conv1')
    layer1b = conv_layer(layer1, 2 ** (filter_scale + 3), strides=(2, 2), name='conv1b')

    layer2 = conv_layer(layer1b, 2 ** (filter_scale + 4), strides=(1, 1), name='conv2')
    layer2b = conv_layer(layer2, 2 ** (filter_scale + 4), strides=(2, 2), name='conv2b')

    layer3 = conv_layer(layer2b, 2 ** (filter_scale + 5), strides=(1, 1), name='conv3')
    layer3b = conv_layer(layer3, 2 ** (filter_scale + 5), strides=(2, 2), name='conv3b')

    layer4 = conv_layer(layer3b, 2 ** (filter_scale + 6), strides=(1, 1), name='conv4')
    layer4b = conv_layer(layer4, 2 ** (filter_scale + 6), strides=(2, 2), name='conv4b')

    layer5 = conv_layer(layer4b, 2 ** (filter_scale + 7), strides=(1, 1), name='conv5')
    layer5b = conv_layer(layer5, 2 ** (filter_scale + 7), strides=(2, 2), name='conv5b')

    # Mid Flow
    layer6 = conv_layer(layer5b, 2 ** (filter_scale + 7), strides=(1, 1), name='conv6')
    layer6b = conv_layer(layer6, 2 ** (filter_scale + 6), strides=(1, 1), name='conv6b')

    # Decoder
    layer7 = deconv_layer(layer6, 2 ** (filter_scale + 6), strides=(2, 2), name='deconv7')
    layer7 = Concatenate(name='concat7')([layer4b, layer7])
    layer7b = conv_layer(layer7, 2 ** (filter_scale + 5), strides=(1, 1), name='conv7b')
    layer7b = att_block(layer7b, ratio=2, name='DAP7b')

    layer8 = deconv_layer(layer7b, 2 ** (filter_scale + 5), strides=(2, 2), name='conv8')
    layer8 = Concatenate(name='concat8')([layer3b, layer8])
    layer8b = conv_layer(layer8, 2 ** (filter_scale + 4), strides=(1, 1), name='conv8b')
    layer8b = att_block(layer8b, ratio=2, name='DAP8b')

    layer9 = deconv_layer(layer8b, 2 ** (filter_scale + 4), strides=(2, 2), name='conv9')
    layer9 = Concatenate(name='concat9')([layer2b, layer9])
    layer9b = conv_layer(layer9, 2 ** (filter_scale + 2), strides=(1, 1), name='conv9b')
    layer9b = att_block(layer9b, 1, name='DAP9b')

    layer10 = deconv_layer(layer9b, 2 ** (filter_scale + 3), strides=(2, 2), name='conv10')
    layer10 = Concatenate(name='concat10')([layer1b, layer10])
    layer10b = conv_layer(layer10, 2 ** (filter_scale + 2), strides=(1, 1), name='conv10b')
    layer10b = att_block(layer10b, 1, name='DAP10b')

    layer11 = deconv_layer(layer10b, 2 ** (filter_scale + 2), strides=(2, 2), name='conv11')
    layer11 = Concatenate(name='concat11')([inputs, layer11])
    layer11b = conv_layer(layer11, output_classes, strides=(1, 1), activation=None, name='conv11b')
    output_value = Activation('sigmoid', name='output')(layer11b)
    return Model(inputs=inputs, outputs=output_value)


def covidnet(input_shape=(300, 300, 300, 1), output_classes=2, filter_scale=0, **kwargs):
    inputs = tf.keras.Input(shape=(tuple(input_shape)), name='input_image')

    layer1 = conv_layer3d(inputs, 2 ** (filter_scale + 3), padding='valid', kernel=(5, 5, 5), strides=(3, 3, 3), name='conv_1')
    layer1b = dual_attention_block3D(layer1, ratio=1, name='attention_1')

    layer2 = conv_layer3d(layer1b, 2 ** (filter_scale + 4), padding='valid', strides=(1, 1, 1), name='conv_2')
    layer2b = dual_attention_block3D(layer2, ratio=2, name='attention_2')

    layer3 = conv_layer3d(layer2b, 2 ** (filter_scale + 4), padding='valid', strides=(2, 2, 2), name='conv_3')
    layer3b = dual_attention_block3D(layer3, ratio=2, name='attention_3')

    layer4 = conv_layer3d(layer3b, 2 ** (filter_scale + 5), padding='valid', strides=(1, 1, 1), name='conv_4')
    layer4b = dual_attention_block3D(layer4, ratio=4, name='attention_4')

    layer5 = conv_layer3d(layer4b, 2 ** (filter_scale + 6), padding='valid', strides=(1, 1, 1), name='conv_5')
    layer5b = dual_attention_block3D(layer5, name='attention_5')

    layer6 = conv_layer3d(layer5b, 2 ** (filter_scale + 6), padding='valid', strides=(1, 1, 1), name='conv_6')
    layer6b = dual_attention_block3D(layer6, name='attention_6')

    pooling_layer = GlobalAveragePooling3D(name='avgpool')(layer6b)
    dense_first = dense_layer(pooling_layer, 2 ** (filter_scale + 7), name='dense_1')
    dense_out = dense_layer(dense_first, 2, activation=Softmax)
    return Model(inputs=inputs, outputs=dense_out)


def coughnet(input_shape=(155, 40, 2), filter_scale=0, drop_rate=0, **kwargs):
    inputs = tf.keras.Input(shape=(tuple(input_shape)), name='input_image')

    global_args = {'padding': 'valid', 'kernel': (3, 3)}
    layer1 = conv_layer(inputs, 2 ** (filter_scale + 4), strides=(1, 1), name='conv_1', **global_args)
    layer2 = conv_layer(layer1, 2 ** (filter_scale + 4), strides=(2, 1), name='conv_2', **global_args)
    layer3 = conv_layer(layer2, 2 ** (filter_scale + 5), strides=(1, 1), name='conv_3', **global_args)
    layer4 = conv_layer(layer3, 2 ** (filter_scale + 5), strides=(2, 2), name='conv_4', **global_args)
    layer5 = conv_layer(layer4, 2 ** (filter_scale + 6), strides=(1, 1), name='conv_5', **global_args)
    layer6 = conv_layer(layer5, 2 ** (filter_scale + 6), strides=(2, 2), name='conv_6', **global_args)
    layer7 = conv_layer(layer6, 2 ** (filter_scale + 6), strides=(1, 1), name='conv_7', **global_args)

    pool = GlobalAveragePooling2D(name='avgpool')(layer7)
    dense1 = dense_layer(pool, 2 ** (filter_scale + 7), name='dense_8')
    if drop_rate > 0:
        dense1 = Dropout(drop_rate, name='dense_8/dropout')(dense1)
    dense2 = dense_layer(pool, 1, name='dense_out', activation=None)
    output = Activation('sigmoid', name='act_out')(dense2)
    return Model(inputs=inputs, outputs=output)


# Convenience method for model lookup
def get_model_func(model_type):
    models = {
        'yamnet': yamnet,
        'multires': multires_classifier,
        'unet': like_unet,
        'mobilenet': mobilenet,
        'combonet': combined_networks,
        'combonet-label': uncombined_network,
        'segnet3d': segnet_3d,
        'segnet2d': segnet_2d,
        'covidnet': covidnet,
        'coughnet': coughnet
    }
    if model_type not in models:
        raise NotImplementedError("Model type '{}' does not exist".format(model_type))
    return models[model_type]
