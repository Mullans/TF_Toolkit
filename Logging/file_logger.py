import os

from .core_logger import CoreLoggingHandler


class FileLoggingHandler(CoreLoggingHandler):

    def write(self, prefix, reset=True):
        log_string = self._get_log_string(prefix)
        self.output_file.write(log_string + '\n')
        self.output_file.flush()
        if reset:
            for metric in self.train_metrics + self.val_metrics:
                metric.reset_states()
        return log_string

    def start(self, logdir):
        self.output_file = open(os.path.join(logdir, 'logfile.txt'), 'a+')
        self.output_file.write("Starting Training\n")

    def interrupt(self):
        self.output_file.write('Interrupted.\n')
        self.output_file.flush()

    def stop(self):
        self.output_file.close()
