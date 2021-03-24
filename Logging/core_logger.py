import numpy as np
from .metrics import MetricWrapper


def num_digits(x):
    if x == 0:
        return 1
    return int(np.ceil(np.log10(np.abs(x) + 1)))


class CoreLoggingHandler(object):
    def __init__(self, train_metrics=None, val_metrics=None):
        self.epochs = None
        self.digits = None
        if train_metrics is None:
            self.train_metrics = []
        else:
            for metric in train_metrics:
                if not isinstance(metric, MetricWrapper):
                    raise ValueError("Metrics used to initialize logging handlers must be MetricWrapper objects")
            self.train_metrics = train_metrics
        if val_metrics is None:
            self.val_metrics = []
        else:
            for metric in val_metrics:
                if not isinstance(metric, MetricWrapper):
                    raise ValueError("Metrics used to initialize logging handlers must be MetricWrapper objects")
            self.val_matrics = val_metrics

    def _get_log_string(self, prefix):
        if isinstance(prefix, int):
            epoch = prefix + 1  # Correct for 0 indexing
            if self.digits is None and self.epochs is not None:
                self.digits = str(num_digits(epoch)) if self.epochs is None else str(num_digits(self.epochs))
            prefix = 'Epoch {:' + self.digits + 'd}'
            prefix = prefix.format(epoch)
            if self.epochs is not None:
                prefix += '/{:' + self.digits + 'd} - '
                prefix = prefix.format(self.epochs)
        train_string = ', '.join([metric.log_string() for metric in self.train_metrics])
        val_string = ', '.join([metric.log_string() for metric in self.val_metrics])
        log_string = prefix + " " + ' || '.join([train_string, val_string])
        return log_string

    def add_metric(self, metric, relevant_idx, name=None, in_training=True, in_validation=True):
        if not in_training and not in_validation:
            raise ValueError("Metrics must be in either training or validation")
        if in_training:
            metric_name = name
            if name is not None:
                metric_name = 'Train_' + name if in_validation and 'train' not in name.lower() else name
            self.train_metrics.append(MetricWrapper(metric, relevant_idx, name=metric_name))
        if in_validation:
            metric_name = name
            if name is not None:
                metric_name = 'Val_' + name if in_training and 'val' not in name.lower() else name
            self.val_metrics.append(MetricWrapper(metric, relevant_idx, name=metric_name))

    def start(self, logdir, total_epochs=None):
        self.epochs = total_epochs

    def write(self, prefix):
        raise NotImplementedError("A derived class should implement write()")

    def train_step(self, results):
        for metric in self.train_metrics:
            metric(results)

    def val_step(self, results):
        for metric in self.val_metrics:
            metric(results)

    def interrupt(self):
        raise NotImplementedError("A derived class should implement interrupt()")

    def stop(self):
        raise NotImplementedError("A derived class should implement stop()")
