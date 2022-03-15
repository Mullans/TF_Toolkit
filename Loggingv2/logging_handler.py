import importlib.util

if importlib.util.find_spec('tensorflow') is not None:
    from .tensorflow_metric_handler import TFMetricHandler
    from .tensorboard_log_writer import TensorboardLogWriter  # This may need to be made generic for pytorch tensorboard?
    from .text_log_writer import TextLogWriter
    TF_AVAIL = True
else:
    TF_AVAIL = False


class LoggingHandler(object):
    def __init__(self, total_epochs=-1, metric_type='tf', logging_type='tensorboard'):
        if metric_type.lower() in ['tf', 'tensorflow']:
            if not TF_AVAIL:
                raise ImportError('TensorFlow install not found. Unable to create LoggingHandler with TensorFlow metrics.')
            self.metric_handler = TFMetricHandler()
        elif metric_type.lower() == 'ignite':
            raise NotImplementedError('Pytorch Ignite metrics are not supported yet')
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        if logging_type.lower() == 'tensorboard':
            # initialize Tensorboard_Logging_Handler
            if not TF_AVAIL:
                raise ImportError('Tensorflow install not found. Unable to use TensorboardLogWriter.')
            self.log_writer = TensorboardLogWriter(total_epochs)
        elif logging_type.lower() in ['text', 'textfile']:
            # initialize textfile logging handler
            self.log_writer = TextLogWriter(total_epochs)
        else:
            raise ValueError(f"Unknown logging type: {logging_type}")

    def write(self, epoch, reset=True):
        log_string = self.log_writer.write(epoch, self.metric_handler.train_metrics, self.metric_handler.val_metrics)
        if reset:
            self.metric_handler.reset()
        return log_string

    def __getattr__(self, attr):
        if hasattr(self.metric_handler, attr):
            return getattr(self.metric_hander, attr)
        elif hasattr(self.log_writer, attr):
            return getattr(self.log_writer, attr)
        else:
            raise AttributeError(f"'LoggingHandler' has no attribute '{attr}'")
