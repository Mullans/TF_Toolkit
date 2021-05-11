import abc
import importlib.util
import warnings

import numpy as np
from .metrics import CoreMetricWrapper

if importlib.util.find_spec('tensorflow') is not None:
    TF_AVAILABLE = True
    from .tensorflow_metrics import TFMetricWrapper, get_tf_metric
else:
    TF_AVAILABLE = False
if importlib.util.find_spec('ignite') is not None:
    IGNITE_AVAILABLE = True
    from .ignite_metrics import IgniteMetricWrapper, get_ignite_metric
else:
    IGNITE_AVAILABLE = False


def find_num_digits(x):
    return 1 if x == 0 else int(np.ceil(np.log10(np.abs(x) + 1)))


class CoreLoggingHandler(object):
    def __init__(self, train_metrics=None, val_metrics=None, total_epochs=None):
        train_metrics = [] if train_metrics is None else train_metrics
        val_metrics = [] if val_metrics is None else val_metrics
        self.total_epochs = total_epochs
        if self.total_epochs is not None:
            num_digits = str(find_num_digits(self.total_epochs))
            self.prefix = 'Epoch {epoch:' + num_digits + 'd}/{total_epochs:' + num_digits + 'd}'
        else:
            self.prefix = 'Epoch {epoch:3d}'
        for metric in train_metrics + val_metrics:
            if not isinstance(metric, CoreMetricWrapper):
                raise TypeError('Initializing metrics must be added as a subtype of CoreMetricWrapper objects')
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

    def get_log_string(self, epoch):
        prefix = self.prefix.format(epoch=epoch, total_epochs=self.total_epochs)
        train_string = ', '.join([metric.get_log_string() for metric in self.train_metrics])
        val_string = ', '.join([metric.get_log_string() for metric in self.val_metrics])
        log_string = ' || '.join([prefix, train_string, val_string])
        return log_string

    def add_metric(self,
                   metric,
                   relevant_idx,
                   in_training=True,
                   in_validation=True,
                   as_tf=False,
                   as_ignite=False,
                   log_pattern='{prefix}/{name}: {result:.4f}',
                   as_percent=False,
                   metric_kwargs={}):
        if isinstance(metric, str):
            if as_tf and not TF_AVAILABLE:
                as_tf = False
                warnings.warn('A Tensorflow install cannot be found. "as_tf" flag set to False.')
            if as_ignite and not IGNITE_AVAILABLE:
                as_ignite = False
                warnings.warn('An Ignite install cannot be found. "as_ignite" flag set to False.')
            if as_tf and as_ignite:
                warnings.warn('Cannot use both Tensorflow and Ignite as backends for the same metric. Defaulting to TensorFlow.')
            if as_tf:
                metric = get_tf_metric(metric, **metric_kwargs)
            elif as_ignite:
                metric = get_ignite_metric(metric)
            else:
                raise ValueError('No valid metric backend was found. Please install either Pytorch Ignite or TensorFlow.')
        if isinstance(metric, CoreMetricWrapper):
            if in_training and in_validation:
                prefix = metric.logging_prefix
                if prefix == '':
                    metric.logging_prefix = 'Train'
                self.train_metrics.append(metric)
                self.val_metrics.append(metric.copy(logging_prefix='Val' + prefix))
            elif in_training:
                if metric.logging_prefix == '':
                    metric.logging_prefix = 'Train'
                self.train_metrics.append(metric)
            elif in_validation:
                if metric.logging_prefix == '':
                    metric.logging_prefix = 'Val'
                self.val_metrics.append(metric)
        elif isinstance(metric, abc.ABCMeta) and "class 'tensorflow" in str(metric).lower():
            # a keras metric class
            if in_training:
                self.train_metrics.append(TFMetricWrapper(metric(**metric_kwargs), relevant_idx, logging_prefix='Train', log_pattern=log_pattern, as_percent=as_percent))
            if in_validation:
                self.val_metrics.append(TFMetricWrapper(metric(**metric_kwargs), relevant_idx, logging_prefix='Val', log_pattern=log_pattern, as_percent=as_percent))
        elif ('keras.metrics' in str(metric) or 'tensorflow_metrics' in str(metric)) and 'object' in str(metric):
            # a keras metric object
            if in_training:
                self.train_metrics.append(TFMetricWrapper(metric, relevant_idx, logging_prefix='Train', log_pattern=log_pattern, as_percent=as_percent))
            if in_validation:
                self.val_metrics.append(TFMetricWrapper(metric, relevant_idx, logging_prefix='Val', as_copy=in_training, log_pattern=log_pattern, as_percent=as_percent))
        elif isinstance(metric, abc.ABCMeta) and "class 'ignite" in str(metric).lower():
            if 'name' in metric_kwargs:
                metric_name = metric_kwargs['name']
                del metric_kwargs['name']
            else:
                metric_name = str(type(metric)).rsplit('.', 1)[1][:-2]
            if in_training:
                self.train_metrics.append(IgniteMetricWrapper(metric(**metric_kwargs), relevant_idx, logging_prefix='Train', log_pattern=log_pattern, as_percent=as_percent, name=metric_name))
            if in_validation:
                self.val_metrics.append(IgniteMetricWrapper(metric(**metric_kwargs), relevant_idx, logging_prefix='Val', log_pattern=log_pattern, as_percent=as_percent, name=metric_name))
            #TODO - add ignite metrics
            raise NotImplementedError("Working on it...")
            IgniteMetricWrapper()

    def start(self, logdir):
        raise NotImplementedError("A derived class should implement start()")

    def write(self, prefix, reset=False):
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
