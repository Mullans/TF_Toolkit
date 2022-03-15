import os

from .log_writer import LogWriter


class TextLogWriter(LogWriter):
    def __init__(self):
        super(TextLogWriter, self).__init__()

    def start(self, logdir):
        self.output_file = open(os.path.join(logdir, 'logfile.txt'), 'a+')
        self.output_file.write('Starting Training\n')

    def write(self, epoch, train_metrics, val_metrics):
        log_string = self.get_log_string(epoch, train_metrics, val_metrics)
        self.output_file.write(log_string + '\n')
        self.output_file.flush()
        return log_string

    def interrupt(self):
        self.output_file.write('Interrupted\n')
        self.output_file.flush()

    def stop(self):
        self.output_file.close()
