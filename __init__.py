import os
import warnings
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

try:  # Any code that relies on tensorflow
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    from .model_arch import get_model_func
    from .losses import get_loss_func
    from .learning_rates import get_lr_func
    from .train_functions import get_update_step
    from .core_model import CoreModel
except ImportError:
    warnings.warn("Valid TensorFlow install not found. Some options will not be available.")

try:
    from .Logging import TensorboardLoggingHandler, FileLoggingHandler
except ImportError:
    warnings.warn('Valid Tensorboard install not found. Some options will not be available.')

# Does not rely on outside module
from .stable_counter import StableCounter
