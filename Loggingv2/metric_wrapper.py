class MetricWrapper(object):
    def __init__(self, metric, relevant_idx, logging_prefix="", log_pattern='{prefix}/{name}: {result:.4f}', as_percent=False):
        self.metric = metric
        self.relevant_idx = relevant_idx
        self.logging_prefix = logging_prefix
        self.log_pattern = log_pattern
        self.as_percent = as_percent

    def get_log_string(self):
        result = self.result() * 100 if self.as_percent else self.result()
        return self.log_pattern.format(prefix=self.logging_prefix, name=self.name, result=result)

    def reset(self):
        raise NotImplementedError("The subclass should implement this")

    def result(self):
        raise NotImplementedError("The subclass should implement this")

    def __call__(self, results):
        raise NotImplementedError('The subclass should implement this')

    def __repr__(self):
        return "['{name}']".format(name=self.name) + repr(self.metric)
