import abc

from .metric_handler import MetricHandler
from .tensorflow_metrics import TFMetricWrapper, get_tf_metric


class TFMetricHandler(MetricHandler):
    def add_metric(self,
                   metric,
                   relevant_idx,
                   in_training=True,
                   in_validation=True,
                   log_pattern='{prefix}/{name}: {result:.4f}',
                   as_percent=False,
                   metric_kwargs={}):
        if isinstance(metric, str):
            metric = get_tf_metric(metric, **metric_kwargs)

        if isinstance(metric, TFMetricWrapper):
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
