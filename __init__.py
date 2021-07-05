import os
import warnings
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

try:  # Any code that relies on tensorflow
    import tensorflow as tf
    from .augmenter import get_image_augmenter
    from .model_arch import get_model_func
    from .losses import get_loss_func
    from .learning_rates import get_lr_func
    from .train_functions import get_update_step
    from .core_model import CoreModel
    from .distributed_model import DistributedModel
    from .multi_model import MultiModel
    from .utils import enforce_4D
except ImportError:
    warnings.warn("Valid TensorFlow install not found. Some options will not be available.")

try:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
except NameError:
    pass  # TF isn't imported, handled by previous try/except
except RuntimeError as e:
    warnings.warn(e.args[0], category=RuntimeWarning)

try:
    from .Logging import TensorboardLoggingHandler, FileLoggingHandler
except ImportError:
    warnings.warn('Valid Tensorboard install not found. Some options will not be available.')

# Does not rely on outside module
from .stable_counter import StableCounter  # noqa
