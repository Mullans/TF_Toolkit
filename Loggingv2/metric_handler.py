class MetricHandler(object):
    def __init__(self):
        self.train_metrics = []
        self.val_metrics = []

    def add_metric(self,
                   metric,
                   relevant_idx,
                   in_training=True,
                   in_validation=True,
                   log_pattern='{prefix}/{name}: {result:.4f}',
                   as_percent=False,
                   metric_kwargs={}):
        raise NotImplementedError("The subclass should implement this")

    def train_step(self, results):
        for metric in self.train_metrics:
            metric(results)

    def val_step(self, results):
        for metric in self.val_metrics:
            metric(results)
    #
    # def get_train_string(self, epoch):
    #     return ', '.join([metric.get_log_string() for metric in self.train_metrics])
    #
    # def get_val_string(self, epoch):
    #     return ', '.join([metric.get_log_string() for metric in self.val_metrics])

    def get_train_metric(self, metric_name):
        for metric in self.train_metrics:
            if metric.name == metric_name:
                return metric
        raise ValueError(f"No training metrics found matching: {metric_name}")

    def get_val_metric(self, metric_name):
        for metric in self.val_metrics:
            if metric.name == metric_name:
                return metric
        raise ValueError(f"No validation metrics found matching: {metric_name}")
