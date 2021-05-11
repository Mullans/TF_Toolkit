import ignite
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
from .metrics import CoreMetricWrapper


def get_ignite_metric(metric_name):
    metric_lookup = {
        'mean': ignite.metrics.Average,
        'dice': TorchDice,
        'rounding_accuracy': RoundingAccuracy
    }
    if metric_name.lower() not in metric_lookup:
        for item in ignite.metrics.__all__:
            if metric_name.lower() == item.lower():
                return getattr(ignite.metrics, item)
        raise ValueError("Unknown metric name: {}".format(metric_name))
    return metric_lookup[metric_name.lower()]


class IgniteMetricWrapper(CoreMetricWrapper):
    def __init__(self, metric, relevant_idx, logging_prefix="", as_copy=False, log_pattern='{prefix}/{name}: {result:.4f}', as_percent=False, name=''):
        if as_copy:
            metric = type(metric)()
        if not hasattr(relevant_idx, '__len__'):
            relevant_idx = [relevant_idx]
        super().__init__(metric, relevant_idx, logging_prefix=logging_prefix, log_pattern=log_pattern, as_percent=as_percent)
        self.__name = name

    @property
    def name(self):
        return self.__name

    def __call__(self, results):
        result = [results[idx] for idx in self.relevant_idx]
        if len(result) == 1:
            self.metric.update(result[0])
        else:
            self.metric.update(result)

    def reset(self):
        self.metric.reset()

    def result(self):
        return self.metric.compute()

    def copy(self, **kwargs):
        copy_wrapper = IgniteMetricWrapper(self.metric, self.relevant_idx, self.logging_prefix, True, self.log_pattern, self.as_percent, self.name)
        for key, value in kwargs:
            if hasattr(copy_wrapper):
                setattr(copy_wrapper, key, value)
        return copy_wrapper


class TorchDice(ignite.metrics.Metric):
    def __init__(self, device="cuda"):
        self._truepositive = None
        self._falsepositive = None
        self._falsenegative = None
        super(TorchDice, self).__init__(device=device)

    @reinit__is_reduced
    def reset(self):
        self._truepositive = 0
        self._falsepositive = 0
        self._falsenegative = 0
        super(TorchDice, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()

        tp = torch.count_nonzero(torch.logical_and(y_pred, y))
        tn = torch.count_nonzero(torch.logical_and(torch.logical_not(y_pred), torch.logical_not(y)))
        fp = torch.count_nonzero(y) - tp
        fn = torch.count_nonzero(torch.logical_not(y)) - tn
        self._truepositive += tp.to(self._device)
        self._falsepositive += fp.to(self._device)
        self._falsenegative += fn.to(self._device)
        pass

    @sync_all_reduce("_truepositive", "_falsepositive", "_falsenegative")
    def compute(self):
        numerator = 2 * self._truepositive
        denominator = 2 * self._truepositive + self._falsepositive + self._falsenegative
        if denominator == 0:
            return 0
        return torch.divide(numerator, denominator)


class RoundingAccuracy(ignite.metrics.Metric):
    def __init__(self, threshold=0.5, device='cuda'):
        self.threshold = threshold
        self._matches = None
        self._total = None
        super(RoundingAccuracy, self).__init__(device=device)

    @reinit__is_reduced
    def reset(self):
        self._matches = torch.tensor(0, device=self._device)
        self._total = 0
        super(RoundingAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        y_pred = torch.greater(y_pred, self.threshold)
        correct = torch.eq(y_pred, y).view(-1)

        correct = torch.eq(y_pred, y).view(-1)
        self._matches += torch.sum(correct).to(self._device)
        self._total += correct.shape[0]

    @sync_all_reduce("_matches", "_total")
    def compute(self):
        if self._total == 0:
            return 0
        return torch.divide(self._matches, self._total)
