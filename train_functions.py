import tensorflow as tf

# Train functions should take (model, optimizer, loss, logging_handler)
# Val functions should take (model, loss, logging_handler)


def train_step_func(model, optimizer, loss_func, logging_handler):
    def train_step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_func(y, predictions)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        logging_handler.train_step((y, predictions, loss))
        return loss
    return tf.function(train_step)


def val_step_func(model, loss_func, logging_handler):
    def val_step(inputs):
        x, y = inputs
        predictions = model(x, training=False)
        loss = loss_func(y, predictions)
        logging_handler.val_step((y, predictions, loss))
        return loss
    return tf.function(val_step)


def distributed_train_step_func(model, optimizer, loss_func, logging_handler, batch_size=32):
    """Default distributed train step for single input, single output, single loss models"""
    def train_step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_func(y, predictions)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        logging_handler.train_step((y, predictions, loss / batch_size))
        return loss
    return tf.function(train_step)


def distributed_val_step_func(model, loss_func, logging_handler, batch_size=32):
    """Default distributed validation step for single input, single output, single loss models"""
    def val_step(inputs):
        x, y = inputs
        predictions = model(x, training=False)
        loss = loss_func(y, predictions)
        logging_handler.val_step((y, predictions, loss / batch_size))
        return loss
    return tf.function(val_step)


def get_update_step(step_type='default', is_training=True):
    if is_training:
        step_funcs = {
            'default': train_step_func,
            'default_distributed': distributed_train_step_func
        }
    else:
        step_funcs = {
            'default': val_step_func,
            'default_distributed': distributed_val_step_func
        }
    if step_type in step_funcs:
        return step_funcs[step_type]
    else:
        raise NotImplementedError('Update step "{}" does not exist'.format(train_type))
