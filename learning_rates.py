import tensorflow as tf


def exponential_decay_lr(learning_rate=1e-4, decay_steps=None, decay_rate=None, **kwargs):
    if decay_steps is None:
        raise ValueError('Decay steps must be set for exponential decay learning rate')
    if decay_rate is None:
        raise ValueError("Decay rate must be set for exponential decay learning rate")

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )
    return lr


def inverse_time_decay_lr(learning_rate=1e-4, decay_steps=None, decay_rate=None, **kwargs):
    if decay_steps is None:
        raise ValueError('Decay steps must be set for inverse time decay learning rate')
    if decay_rate is None:
        raise ValueError("Decay rate must be set for inverse time decay learning rate")

    lr = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )
    return lr


def cosine_decay_lr(learning_rate=1e-4, decay_steps=None, minimum_learning_rate=None, alpha=None, **kwargs):
    if decay_steps is None:
        raise ValueError('Decay steps must be set for cosine decay learning rate')
    if minimum_learning_rate is None and alpha is None:
        raise ValueError("Minimum learning rate must be set for cosine decay learning rate")
    if not (alpha is None or minimum_learning_rate is None):
        raise ValueError("Either alpha or minimum_learning_rate can be set, not both.")
    if alpha is None:
        alpha = minimum_learning_rate / learning_rate
    lr = tf.keras.experimental.CosineDecay(learning_rate, decay_steps, alpha=alpha)
    return lr


def cosine_decay_restarts_lr(learning_rate=1e-4, first_decay_steps=500, lr_time_multiplier=2.0, lr_multiplier=1.0, lr_alpha=0.0, **kwargs):
    return tf.keras.optimizers.schedules.CosineDecayRestarts(learning_rate,
                                                             first_decay_steps,
                                                             t_mul=lr_time_multiplier,
                                                             m_mul=lr_multiplier,
                                                             alpha=lr_alpha)


def basic_lr(learning_rate=1e-4, **kwargs):
    return learning_rate


def get_lr_func(lr_type):
    rate_types = {
        'base': basic_lr,
        'cosine_decay': cosine_decay_lr,
        'cosine_decay_restarts': cosine_decay_restarts_lr,
        'exponential_decay': exponential_decay_lr,
        'inverse_time_decay': inverse_time_decay_lr
    }
    if lr_type not in rate_types:
        raise NotImplementedError("That learning rate type has not been added yet")
    return rate_types[lr_type]
