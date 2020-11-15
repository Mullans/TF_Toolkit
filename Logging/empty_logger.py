from .core_logger import CoreLoggingHandler


class EmptyLoggingHandler(CoreLoggingHandler):
    """This is a dummy handler in case no logging is used during training"""
    def write(self, prefix):
        return ''

    def start(self, logdir, total_epochs=None):
        pass

    def interrupt(self):
        pass

    def stop(self):
        pass

    def train_step(self, results):
        pass
