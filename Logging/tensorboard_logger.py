import os
import tensorflow as tf

from .core_logger import CoreLoggingHandler, find_num_digits


class TensorboardLoggingHandler(CoreLoggingHandler):
    def write(self, epoch, reset=True):
        log_string = self.get_log_string(epoch)
        if isinstance(epoch, str):
            epoch = 0
        with self.train_writer.as_default():
            for metric in self.train_metrics:
                tf.summary.scalar(metric.name, metric.result(), step=epoch)  # TODO - make generic
        with self.val_writer.as_default():
            for metric in self.val_metrics:
                tf.summary.scalar(metric.name, metric.result(), step=epoch)  # TODO - make generic
        if reset:
            for metric in self.train_metrics + self.val_metrics:
                metric.reset()
        return log_string

    def start(self, logdir, total_epochs=None):
        # torch.utils.tensorboard.SummaryWriter
        # writer.add_scalar('Name', value, epoch)
        # writer.flush(), writer.close()
        if total_epochs is not None:
            self.total_epochs = total_epochs
            num_digits = str(find_num_digits(self.total_epochs))
            self.prefix = 'Epoch {epoch:' + num_digits + '}/{total_epochs:' + num_digits + 'd}'
            print(total_epochs)
        self.train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))  # TODO - make generic
        self.val_writer = tf.summary.create_file_writer(os.path.join(logdir, 'val'))  # TODO - make generic
        # if self.epochs is not None:
        #     self.digits = str(find_num_digits(self.epochs))

    def interrupt(self):
        pass

    def stop(self):
        pass
