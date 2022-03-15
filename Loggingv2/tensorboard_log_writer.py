import os
import tensorflow as tf

from .log_writer import LogWriter


class TensorboardLogWriter(LogWriter):
    def __init__(self):
        super(TensorboardLogWriter, self).__init__()
        self.train_writer = None
        self.val_writer = None

    def start(self, logdir):
        self.train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
        self.val_writer = tf.summary.create_file_writer(os.path.join(logdir, 'val'))

    def write(self, epoch, train_metrics, val_metrics):
        if isinstance(epoch, str):
            epoch = 0
        with self.train_writer.as_default():
            for metric in train_metrics:
                tf.summary.scalar(metric.name, metric.result(), step=epoch)
        with self.val_writer.as_default():
            for metric in val_metrics:
                tf.summary.scalar(metric.name, metric.result(), step=epoch)
        return self.get_log_string(epoch, train_metrics, val_metrics)

    def interrupt(self):
        pass

    def stop(self):
        pass
