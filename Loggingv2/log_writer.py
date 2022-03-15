import numpy as np


def find_num_digits(x):
    return 1 if x == 0 else int(np.ceil(np.log10(np.abs(x) + 1)))


class LogWriter(object):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        if total_epochs > 0:
            self.epoch_digits = str(find_num_digits(self.total_epochs))
            self.prefix = 'Epoch {epoch:' + self.epoch_digits + 'd}/{total_epochs:' + self.epoch_digits + 'd}'
        else:
            self.epoch_digits = 0
            self.prefix = 'Epoch {epoch:d}'

    def get_log_string(self, epoch, train_metrics, val_metrics):
        prefix = self.prefix.format(epoch=epoch, total_epochs=self.total_epochs)
        train_string = ', '.join([metric.get_log_string() for metric in train_metrics])
        val_string = ', '.join([metric.get_log_string() for metric in val_metrics])
        log_string = ' || '.join([prefix, train_string, val_string])
        return log_string

    def start(self, logdir):
        raise NotImplementedError("A derived class should implement start()")

    def write(self, epoch, train_metrics, val_metrics):
        raise NotImplementedError("A derived class should implement start()")

    def interrupt(self):
        raise NotImplementedError("A derived class should implement start()")

    def stop(self):
        raise NotImplementedError("A derived class should implement start()")
