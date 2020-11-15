import os

from .core_logger import CoreLoggingHandler, num_digits


class FileLoggingHandler(CoreLoggingHandler):

    def write(self, prefix):
        log_string = self._get_log_string(prefix)
        self.output_file.write(log_string + '\n')
        self.output_file.flush()
        for metric in self.train_metrics + self.val_metrics:
            metric.reset_states()
        return log_string

    def start(self, logdir, total_epochs=None):
        super(FileLoggingHandler, self).start(logdir, total_epochs=total_epochs)
        self.output_file = open(os.path.join(logdir, 'logfile.txt'), 'a+')
        self.output_file.write("Starting Training\n")
        if self.epochs is not None:
            self.digits = str(num_digits(self.epochs))

    def interrupt(self):
        self.output_file.write('Interrupted.\n')

    def stop(self):
        self.output_file.close()
