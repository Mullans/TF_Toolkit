# import ignite
from .metrics import MetricWrapper


class IgniteMetricWrapper(MetricWrapper):
    def __init__(self, metric, relevant_idx, name=None, logging_pattern="{:.4f}", as_percent=False):
        if name is None:
            name = str(repr(metric)).split(' ')[0].split('.')[-1]
        super().__init__(metric, relevant_idx, name, logging_pattern=logging_pattern, as_percent=as_percent)
        self.metric = metric

    def __call__(self, results):
        self.metric.update(([results[idx] for idx in self.relevant_idx]))

    def reset_states(self):
        self.metric.reset()

    def result(self):
        return self.metric.compute()
