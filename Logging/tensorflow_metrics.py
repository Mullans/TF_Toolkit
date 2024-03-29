import tensorflow as tf
from .metrics import CoreMetricWrapper
epsilon = tf.keras.backend.epsilon()


def get_tf_metric(metric_name, **metric_kwargs):
    metric_lookup = {
        'loss': tf.keras.metrics.Mean,
        'balance': BalanceScore,
        'mcc': MatthewsCorrelationCoefficient,
        'dice': ScalarDiceScore,
        'bool_dice': BooleanDiceScore,
    }
    if metric_name.lower() not in metric_lookup:
        return tf.keras.metrics.get({"class_name": metric_name.title(), "config": metric_kwargs})
    return metric_lookup[metric_name.lower()](**metric_kwargs)


class TFMetricWrapper(CoreMetricWrapper):
    def __init__(self, metric, relevant_idx, logging_prefix="", as_copy=False, log_pattern='{prefix}/{name}: {result:.4f}', as_percent=False):
        if as_copy:
            metric = tf.keras.metrics.deserialize(tf.keras.metrics.serialize(metric))
        if not hasattr(relevant_idx, '__len__'):
            relevant_idx = [relevant_idx]
        super().__init__(metric, relevant_idx, logging_prefix=logging_prefix, log_pattern=log_pattern, as_percent=as_percent)

    def __call__(self, results):
        self.metric(*[results[idx] for idx in self.relevant_idx])

    def reset(self):
        self.metric.reset_states()

    def result(self):
        return self.metric.result()

    @property
    def name(self):
        return self.metric.name

    def copy(self, **kwargs):
        copy_wrapper = TFMetricWrapper(self.metric, self.relevant_idx, self.logging_prefix, as_copy=True)
        for key, value in kwargs:
            if hasattr(copy_wrapper, key):
                setattr(copy_wrapper, key, value)
        return copy_wrapper


@tf.keras.utils.register_keras_serializable(package='TF_Toolkit')
class BalanceScore(tf.keras.metrics.Metric):
    """TF Metric for binary prediction balance"""
    def __init__(self, name='prediction_balance', use_majority=False, **kwargs):
        super(BalanceScore, self).__init__(name=name, **kwargs)
        self.use_majority = use_majority
        self.positives = self.add_weight(name='pos', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pos = tf.math.count_nonzero(tf.round(y_pred), dtype=self.dtype)
        self.positives.assign_add(pos)
        self.total.assign_add(tf.cast(tf.size(y_pred), self.dtype))

    def result(self):
        ratio = tf.math.divide_no_nan(self.positives, self.total)
        if self.use_majority:
            return tf.maximum(ratio, 1 - ratio)
        return ratio

    def reset_state(self):
        self.positives.assign(0.)
        self.total.assign(0.)

    def get_config(self):
        config = {
            "use_majority": self.use_majority,
        }
        base_config = super(BalanceScore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package='TF_Toolkit')
class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
    """TF implementation of MatthewsCorrelationCoefficient"""
    def __init__(self, name='MCC', return_all=False, dtype=tf.float32):
        super(MatthewsCorrelationCoefficient, self).__init__(name=name, dtype=dtype)
        self.return_all = return_all
        self.true_positive = self.add_weight('true_positives', initializer='zeros')
        self.true_negative = self.add_weight('true_negatives', initializer='zeros')
        self.false_positive = self.add_weight('false_positives', initializer='zeros')
        self.false_negative = self.add_weight('false_negatives', initializer='zeros')

    def update_state(self, labels, predictions, sample_weight=None):
        label_true = tf.math.greater_equal(tf.reshape(tf.cast(labels, self.dtype), [-1]), 0.5)
        pred_true = tf.math.greater_equal(tf.reshape(tf.cast(predictions, self.dtype), [-1]), 0.5)

        true_positive = tf.math.count_nonzero(tf.math.logical_and(label_true, pred_true), dtype=self.dtype)
        true_negative = tf.math.count_nonzero(tf.math.logical_and(tf.logical_not(label_true), tf.logical_not(pred_true)), dtype=self.dtype)
        false_positive = tf.math.count_nonzero(label_true, dtype=self.dtype) - true_positive
        false_negative = tf.math.count_nonzero(tf.logical_not(label_true), dtype=self.dtype) - true_negative

        self.true_positive.assign_add(true_positive)
        self.true_negative.assign_add(true_negative)
        self.false_positive.assign_add(false_positive)
        self.false_negative.assign_add(false_negative)

    def result(self):
        if self.return_all:
            return [[self.true_negative, self.false_positive], [self.false_negative, self.true_positive]]
        numerator = (self.true_positive * self.true_negative) - (self.false_positive * self.false_negative)
        denominator = (self.true_positive + self.false_positive) * (self.true_positive + self.false_negative) * (self.true_negative + self.false_positive) * (self.true_negative + self.false_negative)
        denominator = tf.math.sqrt(denominator)
        return tf.math.divide_no_nan(numerator, denominator)

    def get_config(self):
        config = {'return_all': self.return_all}
        base_config = super(MatthewsCorrelationCoefficient, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_state(self):
        self.true_positive.assign(0.)
        self.true_negative.assign(0.)
        self.false_positive.assign(0.)
        self.false_negative.assign(0.)


@tf.keras.utils.register_keras_serializable(package='TF_Toolkit')
class BooleanDiceScore(tf.keras.metrics.Metric):
    """Dice score based on thresholding predictions to booleans"""
    def __init__(self, from_logits=False, name='BooleanDiceScore', dtype=tf.float32):
        super(BooleanDiceScore, self).__init__(name=name, dtype=dtype)
        self.from_logits = from_logits
        self.true_positive = self.add_weight('true_positives', initializer='zeros')
        self.false_positive = self.add_weight('false_positives', initializer='zeros')
        self.false_negative = self.add_weight('false_negatives', initializer='zeros')

    def update_state(self, labels, predictions, sample_weight=None):
        if self.from_logits:
            predictions = tf.nn.sigmoid(predictions)
        label_true = tf.math.greater_equal(tf.reshape(tf.cast(labels, self.dtype), [-1]), 0.5)
        pred_true = tf.math.greater_equal(tf.reshape(tf.cast(predictions, self.dtype), [-1]), 0.5)

        true_positive = tf.math.count_nonzero(tf.math.logical_and(label_true, pred_true), dtype=self.dtype)
        true_negative = tf.math.count_nonzero(tf.math.logical_and(tf.logical_not(label_true), tf.logical_not(pred_true)), dtype=self.dtype)
        false_positive = tf.math.count_nonzero(label_true, dtype=self.dtype) - true_positive
        false_negative = tf.math.count_nonzero(tf.logical_not(label_true), dtype=self.dtype) - true_negative

        self.true_positive.assign_add(true_positive)
        self.false_positive.assign_add(false_positive)
        self.false_negative.assign_add(false_negative)

    def result(self):
        numerator = 2 * self.true_positive
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return tf.math.divide_no_nan(numerator, denominator)

    def get_config(self):
        return super(BooleanDiceScore, self).get_config()

    def reset_state(self):
        self.true_positive.assign(0.)
        self.false_positive.assign(0.)
        self.false_negative.assign(0.)


@tf.keras.utils.register_keras_serializable(package='TF_Toolkit')
class ScalarDiceScore(tf.keras.metrics.Metric):
    """Dice score using continuous values"""
    def __init__(self, name='ScalarDiceScore', from_logits=False, dtype=tf.float32):
        super(ScalarDiceScore, self).__init__(name=name, dtype=dtype)
        self.sum_dice = self.add_weight('sum_dice', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')
        self.from_logits = from_logits
        # self.numerator = self.add_weight('numerator', initializer='zeros')
        # self.denominator = self.add_weight('denominator', initializer='zeros')

    def update_state(self, labels, predictions, sample_weight=None):
        if self.from_logits:
            predictions = tf.nn.sigmoid(predictions)
        count = tf.cast(tf.shape(labels)[0], self.dtype)
        self.count.assign_add(count)
        labels = tf.reshape(tf.cast(labels, self.dtype), [count, -1])
        predictions = tf.reshape(tf.cast(predictions, self.dtype), [count, -1])

        numerator = 2 * tf.reduce_sum(tf.multiply(labels, predictions), axis=1)
        denominator = tf.reduce_sum(tf.add(labels, predictions), axis=1) + epsilon
        dice = tf.reduce_sum(tf.divide(numerator, denominator))
        self.sum_dice.assign_add(dice)

    def result(self):
        return tf.math.divide_no_nan(self.sum_dice + epsilon, self.count + epsilon)

    def get_config(self):
        return super(ScalarDiceScore, self).get_config()

    def reset_state(self):
        self.count.assign(0.)
        self.sum_dice.assign(0.)


@tf.keras.utils.register_keras_serializable(package='TF_Toolkit')
class AverageLoss(tf.keras.metrics.Metric):
    """Average a given loss func that takes (y_true, y_pred) as input and returns a scalar value

    Note
    ----
    Averages the loss over steps (each update call), not over instances
    """
    def __init__(self, loss_func, name='AverageLoss', dtype=tf.float32):
        super(AverageLoss, self).__init__(name=name, dtype=dtype)
        self.loss_func = loss_func
        self.sum_loss = self.add_weight('sum_loss', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, labels, predictions, sample_weight=None):
        loss = tf.cast(self.loss_func(labels, predictions), self.dtype)
        self.sum_loss.assign_add(loss)
        self.count.assign_add(1.)

    def result(self):
        return tf.math.divide_no_nan(self.sum_loss, self.count)

    def get_config(self):
        return super(AverageLoss, self).get_config()

    def reset_state(self):
        self.count.assign(0.)
        self.sum_loss.assign(0.)
