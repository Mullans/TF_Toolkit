import os
import tensorboard as tb

from .core_logger import CoreLoggingHandler, num_digits


class TensorboardLoggingHandler(CoreLoggingHandler):

    def write(self, prefix):
        epoch = prefix
        log_string = self._get_log_string(prefix)
        if isinstance(epoch, str):
            epoch = 0
        with self.train_writer.as_default():
            for metric in self.train_metrics:
                tb.summary.scalar(metric.name, metric.result(), step=epoch)
        with self.val_writer.as_default():
            for metric in self.val_metrics:
                tb.summary.scalar(metric.name, metric.result(), step=epoch)
        for metric in self.train_metrics + self.val_metrics:
            metric.reset_states()
        return log_string

    def start(self, logdir, total_epochs=None):
        super(TensorboardLoggingHandler, self).start(logdir, total_epochs=total_epochs)
        self.train_writer = tb.summary.create_file_writer(os.path.join(logdir, 'train'))
        self.val_writer = tb.summary.create_file_writer(os.path.join(logdir, 'val'))
        if self.epochs is not None:
            self.digits = str(num_digits(self.epochs))

    def interrupt(self):
        pass

    def stop(self):
        pass
