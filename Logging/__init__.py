import warnings

try:  # Any code that relies on tensorflow
    import tensorflow as tf
    from .tensorboard_logger import TensorboardLoggingHandler
except ImportError:
    warnings.warn("Valid TensorFlow install not found. Some options will not be available.")


from .file_logger import FileLoggingHandler
from .empty_logger import EmptyLoggingHandler
