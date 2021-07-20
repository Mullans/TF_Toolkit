import tensorflow as tf


def get_reduction(reduction, batch_size=None):
    if reduction == 'sum_over_batch_size' or reduction == 'auto':
        if batch_size is None:
            def reduction_func(x):
                return tf.reduce_mean(x)
        else:
            def reduction_func(x):
                return tf.reduce_sum(x) * (1. / batch_size)
    elif reduction == 'sum':
        reduction_func = tf.reduce_sum
    elif reduction == 'none':
        def reduction_func(x):
            return x
    else:
        raise ValueError("Unknown reduction method")
    return reduction_func


def IOU_loss(**kwargs):
    """Intersection over union loss"""
    def loss(y_true, y_pred):
        # TODO: This doesn't seem super usefull...
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(tf.minimum(y_true, y_pred))
        union = tf.reduce_sum(tf.maximum(y_true, y_pred))
        return 1 - tf.divide(intersection, union)
    return tf.function(loss)


def dice_loss(as_scalar=True, reduction='sum_over_batch_size', **kwargs):
    reduction_func = get_reduction(reduction)
    if as_scalar:
        def loss(y_true, y_pred):
            numerator = 2 * tf.reduce_sum(tf.reshape(y_true * y_pred, [y_true.shape[0], -1]), axis=1)
            denominator = tf.reduce_sum(tf.reshape(y_true + y_pred, [y_true.shape[0], -1]), axis=1)
            dice = 1 - numerator / denominator
            return reduction_func(dice)
    else:
        def loss(y_true, y_pred):
            numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
            denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
            dice = 1 - (numerator + denominator) / (denominator + 1)
            return reduction_func(dice)
    return tf.function(loss)


def tversky_loss(beta=0.5, **kwargs):
    """Like Dice, but adds weight to false positives vs false negatives

    Note
    ----
    Lower beta results in a false negatives getting a higher weight relative to false positives
    """
    def loss(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
        result = 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)
        return tf.reduce_mean(result)
    return tf.function(loss)


def mixed_IOU_BCE_loss(beta=0.5, label_smoothing=0.1, reduction='auto', **kwargs):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=label_smoothing, reduction=reduction)
    iou = IOU_loss()

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        iou_loss = iou(y_true, y_pred)
        bce_loss = bce(y_true, y_pred)
        return iou_loss * beta + bce_loss * (1 - beta)

    return tf.function(loss)


def focal_loss(beta=0.25, gamma=2, reduction='sum_over_batch_size', **kwargs):
    """https://arxiv.org/abs/1708.02002 - code adapted from https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/"""
    reduction_func = get_reduction(reduction)

    def focal_loss_with_logits(logits, targets, beta, gamma, y_pred):
        weight_a = beta * (1 - y_pred) ** gamma * targets
        weight_b = (1 - beta) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = focal_loss_with_logits(logits=logits, targets=y_true, beta=beta, gamma=gamma, y_pred=y_pred)

        return reduction_func(loss)

    return tf.function(loss)


def binary_crossentropy(from_logits=False, label_smoothing=0, reduction='auto', **kwargs):
    return tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing, reduction=reduction)


def categorical_crossentropy(from_logits=False, label_smoothing=0, reduction='auto', **kwargs):
    return tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing, reduction=reduction)


def weighted_crossentropy(pos_weight=1, label_smoothing=0, **kwargs):
    smooth_val = label_smoothing * 2

    def weighted_bce(y_true, y_pred):
        smoothed_y = y_true - (y_true - 0.5) * smooth_val
        loss = tf.nn.weighted_cross_entropy_with_logits(smoothed_y, y_pred, pos_weight)
        return tf.reduce_mean(loss)
    return weighted_bce


def get_combined_losses(*losses, divide_by_total_weight=False, **loss_kwargs):
    """Get a loss function by combining other loss functions.

    Parameters
    ----------
    losses : tuple(str or func or numeric)
        The losses (and optional weights) to combine
    divide_by_total_weight : bool
        Whether to return the final loss as a weighted average
    loss_kwargs : dict
        Any other keywords for the loss functions

    NOTE
    ----
    losses should be each loss function followed by the optional weight as an int or float. If an int or float doesn't follow a loss function, then it will have a weight of 1
    For example: get_combined_losses('bce', 2, 'dice', 3) -> bce loss * 2 + dice loss * 3
    Alternatively: get_combined_losses('bce', 'dice', 3) -> bce loss * 1 + dice loss * 3
    """
    loss_funcs = []
    total_weight = 0
    i = 0
    while i < len(losses):
        loss = losses[i]
        weight = losses[i + 1] if i + 1 < len(losses) else 1
        if not isinstance(weight, (int, float)):
            i -= 1
            weight = 1
        total_weight += weight
        if isinstance(loss, str):
            loss = get_loss_func(loss)(**loss_kwargs)
        loss_funcs.append([loss, weight])
        i += 2

    def loss_func(y_true, y_pred):
        total_loss = 0
        for func, weight in loss_funcs:
            total_loss += func(y_true, y_pred) * weight
        if divide_by_total_weight:
            total_loss /= total_weight
        return total_loss
    return tf.function(loss_func)


def get_loss_func(loss_type, **kwargs):
    """Get a selected loss function

    Parameters
    ----------
    loss_type : str
        Type of loss function to return

    Optional Parameters
    -------------------
    label_smoothing : float
        [0, 1] value to shift labels away from 0 or 1 - currently only in bce or iou_bce
    beta : float
        [0, 1] value for IOU weight in iou_bce or background weight in focal
    gamma : float
        exponent term for focal loss - focal loss with gamma=0 is the same as bce
    as_scalar : bool
        If using dice, whether to return the loss as a single scalar (versus a per-prediction tensor)


    NOTE
    ----
    focal, dice, and tversky loss use code derived from https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

    Currently only BCE has been figured out for distributed training
    """
    losses = {
        'bce': binary_crossentropy,
        'cce': categorical_crossentropy,
        'iou': IOU_loss,
        'iou_bce': mixed_IOU_BCE_loss,
        'focal': focal_loss,
        'dice': dice_loss,
        'tversky': tversky_loss,
        'weighted_ce': weighted_crossentropy
    }
    if loss_type not in losses:
        raise NotImplementedError("Loss type `{}` does not exist".format(loss_type))

    return losses[loss_type]
