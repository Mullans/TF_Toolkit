import tensorflow as tf


def get_metric(metric_name):
    metric_lookup = {
        'loss': [tf.keras.metrics.Mean, 'Loss: {:.4f}'],
        'accuracy': [tf.keras.metrics.BinaryAccuracy, 'Accuracy: {:6.2f}%'],
        'recall': [tf.keras.metrics.Recall, 'Recall: {:.4f}'],
        'precision': [tf.keras.metrics.Precision, 'Precision: {:.4f}'],
        'balance': [BalanceMetric, 'Balance: {:6.2f}%'],
        'mcc': [MatthewsCorrelationCoefficient, 'MCC: {:.4f}'],
        'dice': [ScalarDiceScore, 'Dice: {:.4f}'],
        'bool_dice': [BooleanDiceScore, 'Boolean Dice: {:.4f}']
    }
    if metric_name not in metric_lookup:
        raise NotImplementedError("A wrapper for that metric type has not been added yet")
    return metric_lookup[metric_name]


class MetricWrapper(object):
    def __init__(self, metric_name, relevant_idx, metric_type='train'):
        self.metric = None
        self.metric_func, log_pattern = get_metric(metric_name)
        self.metric_name = metric_name
        if isinstance(relevant_idx, int):
            relevant_idx = [relevant_idx]
        self.relevant_idx = relevant_idx
        self.log_pattern = metric_type.title() + ' ' + log_pattern
        self.as_percent = '%' in log_pattern
        self.metric_type = metric_type

    def compile(self):
        self.metric = self.metric_func(name=self.metric_type + '_' + self.metric_name)

    def __call__(self, results):
        self.metric(*[tf.squeeze(results[idx]) for idx in self.relevant_idx])

    def result(self):
        return self.metric.result()

    def log_string(self):
        if self.as_percent:
            return self.log_pattern.format(self.result() * 100)
        return self.log_pattern.format(self.result())

    def reset_states(self):
        self.metric.reset_states()

    def __repr__(self):
        return "['" + self.metric_name + "'] " + repr(self.metric_func)


# @tf.keras.utils.register_keras_serializable(package='Custom')
class BalanceMetric(tf.keras.metrics.Metric):
    """TF Metric for binary prediction balance"""
    def __init__(self, name='prediction_balance', use_majority=False, **kwargs):
        super(BalanceMetric, self).__init__(name=name, **kwargs)
        self.use_majority = use_majority
        self.positives = self.add_weight(name='pos', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pos = tf.math.count_nonzero(tf.round(y_pred), dtype=self.dtype)
        self.positives.assign_add(pos)
        self.total.assign_add(tf.cast(tf.size(y_pred), self.dtype))

    def result(self):
        ratio = tf.divide(self.positives, self.total)
        if self.use_majority:
            return tf.maximum(ratio, 1 - ratio)
        return ratio

    def reset_states(self):
        self.positives.assign(0.)
        self.total.assign(0.)

    def get_config(self):
        config = {
            "use_majority": self.use_majority,
        }
        base_config = super(BalanceMetric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# @tf.keras.utils.register_keras_serializable(package='Custom')
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

    def reset_states(self):
        self.true_positive.assign(0.)
        self.true_negative.assign(0.)
        self.false_positive.assign(0.)
        self.false_negative.assign(0.)


# @tf.keras.utils.register_keras_serializable(package='Custom')
class BooleanDiceScore(tf.keras.metrics.Metric):
    """Dice score based on thresholding predictions to booleans"""
    def __init__(self, name='BooleanDiceScore', dtype=tf.float32):
        super(BooleanDiceScore, self).__init__(name=name, dtype=dtype)
        self.true_positive = self.add_weight('true_positives', initializer='zeros')
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
        self.false_positive.assign_add(false_positive)
        self.false_negative.assign_add(false_negative)

    def result(self):
        numerator = 2 * self.true_positive
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return tf.math.divide_no_nan(numerator, denominator)

    def get_config(self):
        return super(BooleanDiceScore, self).get_config()

    def reset_states(self):
        self.true_positive.assign(0.)
        self.false_positive.assign(0.)
        self.false_negative.assign(0.)


# @tf.keras.utils.register_keras_serializable(package='Custom')
class ScalarDiceScore(tf.keras.metrics.Metric):
    """Dice score using continuous values"""
    def __init__(self, name='ScalarDiceScore', dtype=tf.float32):
        super(ScalarDiceScore, self).__init__(name=name, dtype=dtype)
        self.numerator = self.add_weight('numerator', initializer='zeros')
        self.denominator = self.add_weight('denominator', initializer='zeros')

    def update_state(self, labels, predictions, sample_weight=None):
        labels = tf.reshape(tf.cast(labels, self.dtype), [-1])
        predictions = tf.reshape(tf.cast(predictions, self.dtype), [-1])

        numerator = 2 * tf.reduce_sum(tf.multiply(labels, predictions))
        denominaotr = tf.reduce_sum(tf.add(labels, predictions))

        self.numerator.assign_add(numerator)
        self.denominator.assign_add(denominator)

    def result(self):
        return tf.math.divide_no_nan(self.numerator, self.denominator)

    def get_config(self):
        return super(ScalarDiceScore, self).get_config()

    def reset_states(self):
        self.numerator.assign(0.)
        self.denominator.assign(0.)
