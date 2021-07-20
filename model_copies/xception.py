from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, MaxPool2D, ReLU, SeparableConv2D
from tensorflow.keras.regularizers import L2
from ..model_arch import conv_layer


def separable_conv_layer2(layer_input, filters, kernel=(3, 3), strides=(1, 1), padding='same', use_bias=True, batchnorm=True, regularize=False, activation=ReLU, name='conv'):
    regularize_weight = 1e-4 if isinstance(regularize, bool) else regularize
    conv = SeparableConv2D(filters,
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


def strided_xception_block(x, filters, output_filters=None, kernel=(3, 3), padding='same', use_bias=True, regularize=False, activation=ReLU, initial_activation=True, name='xception_block'):
    conv_args = {'kernel': kernel, 'strides': 1, 'padding': padding, 'use_bias': use_bias, 'regularize': regularize, 'batchnorm': True}
    if output_filters is None:
        output_filters = filters
    residual = Conv2D(output_filters, 3, 2, padding=padding, name=name + '/residualConv')(x)
    residual = BatchNormalization(name=name + '/residualConv/bn')(residual)
    if initial_activation and activation is not None:
        if isinstance(activation, str):
            x = Activation(activation, name=name + '/initial_act')(x)
        else:
            x = activation(name=name + '/initial_act')(x)
    sepconv1 = separable_conv_layer2(x, filters, activation=activation, name=name + '/sepconv1', **conv_args)
    sepconv2 = separable_conv_layer2(sepconv1, output_filters, activation=None, name=name + '/sepconv2', **conv_args)
    pool = MaxPool2D(pool_size=2, padding=padding, name=name + '/pool')(sepconv2)
    return Add(name=name + '/add')([pool, residual])


def xception_block(x, filters, kernel=(3, 3), padding='same', use_bias=True, regularize=False, activation=ReLU, initial_activation=True, name='xception_block'):
    conv_args = {'kernel': kernel, 'strides': 1, 'padding': padding, 'use_bias': use_bias, 'regularize': regularize, 'batchnorm': True}
    if initial_activation and activation is not None:
        if isinstance(activation, str):
            x_act = Activation(activation, name=name + '/initial_act')(x)
        else:
            x_act = activation(name=name + '/initial_act')(x)
    else:
        x_act = x
    sepconv1 = separable_conv_layer2(x_act, filters, activation=activation, name=name + '/sepconv1', **conv_args)
    sepconv2 = separable_conv_layer2(sepconv1, filters, activation=activation, name=name + '/sepconv2', **conv_args)
    sepconv3 = separable_conv_layer2(sepconv2, filters, activation=None, name=name + '/sepconv3', **conv_args)
    return Add(name=name + '/add')([sepconv3, x])


def Xception(input_shape=[299, 299, 3], filter_scale=0, num_classes=1000, **kwargs):
    inputs = Input(shape=(tuple(input_shape)), name='input_image')

    # block1
    b1c1 = conv_layer(inputs, 2 ** (5 + filter_scale), strides=(2, 2), name='block1_conv1')
    b1c2 = conv_layer(b1c1, 2 ** (6 + filter_scale), strides=(1, 1), name='block1_conv2')

    block2 = strided_xception_block(b1c2, 2 ** (7 + filter_scale), initial_activation=False, name='block2')
    block3 = strided_xception_block(block2, 2 ** (8 + filter_scale), name='block3')
    block4 = strided_xception_block(block3, 728 * 2 ** filter_scale, name='block4')
    block5 = xception_block(block4, 728 * 2 ** filter_scale, name='block5')
    block6 = xception_block(block5, 728 * 2 ** filter_scale, name='block6')
    block7 = xception_block(block6, 728 * 2 ** filter_scale, name='block7')
    block8 = xception_block(block7, 728 * 2 ** filter_scale, name='block8')
    block9 = xception_block(block8, 728 * 2 ** filter_scale, name='block9')
    block10 = xception_block(block9, 728 * 2 ** filter_scale, name='block10')
    block11 = xception_block(block10, 728 * 2 ** filter_scale, name='block11')
    block12 = xception_block(block11, 728 * 2 ** filter_scale, name='block12')
    block13 = strided_xception_block(block12, 728 * 2 ** filter_scale, output_filters=1024 * 2 ** filter_scale, name='block13')

    block14 = separable_conv_layer2(block13, 1536 * 2 ** filter_scale, name='block14/sepconv1')
    block14 = separable_conv_layer2(block14, 2048 * 2 ** filter_scale, name='block14/sepconv2')
    output_pool = GlobalAveragePooling2D(name='output_pool')(block14)
    logits = Dense(num_classes, name='logits')(output_pool)
    return Model(inputs=inputs, outputs=logits)
