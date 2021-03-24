class MetricWrapper(object):
    def __init__(self, metric, relevant_idx, name, logging_pattern="{:.4f}", as_percent=False):
        self.name = name
        self.relevant_idx = relevant_idx
        self.pattern = self.name + ': ' + logging_pattern
        self.as_percent = as_percent

    def __call__(self, results):
        raise NotImplementedError("The subclass should implement this")

    def log_string(self):
        if self.as_percent:
            return self.pattern.format(self.result() * 100)
        return self.pattern.format(self.result())

    def reset_states(self):
        raise NotImplementedError("The subclass should implement this")

    def result(self):
        raise NotImplementedError("The subclass should implement this")

    def __repr__(self):
        return "['" + self.name + "'] " + repr(self.metric)
