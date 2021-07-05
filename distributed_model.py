import gouda
import tensorflow as tf

from .core_model import CoreModel
from .losses import get_loss_func
from .model_arch import get_model_func


class DistributedModel(CoreModel):
    def __init__(self, strategy=tf.distribute.MirroredStrategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(strategy, '__call__'):
            self.strategy = strategy()
        else:
            # If the object has already been initialized
            self.strategy = strategy

    def compile_model(self, model_func=None, checking=False, **kwargs):
        """Compile the model for the given function.

        Parameters
        ----------
        model_func: str | func
            Either the string to lookup, the model function, or None if you want to use the function from self.model_args (the default is None)
        checking: bool
            If true, will only compile a model if there is no compiled model already
        """
        if checking and self.model is not None:
            return
        self.clear()
        if self.model_args['model_func'] is None and model_func is None:
            raise ValueError('No selected model function')
        for key in kwargs:
            self.model_args[key] = kwargs[key]

        if model_func is None:
            model_func = self.model_args['model_func']
        if isinstance(model_func, str):
            model_func = get_model_func(model_func)

        with self.strategy.scope():
            self.model = model_func(**self.model_args)

    def train(self,
              train_data,
              val_data,
              metrics=None,
              starting_epoch=1,
              lr_type=None,
              loss_type=None,
              epochs=50,
              save_every=10,
              load_args=False,
              reduce_lr_on_plateau=True,
              sample_callback=None,
              version='default',
              **kwargs):
        """NOTE: logging_handler from the CoreModel is replaced by metrics in the distributed. This model relies on the keras model.fit methods more than the custom training loop."""
        log_dir = gouda.ensure_dir(self.model_dir(version))
        args_path = log_dir('training_args.json')
        weights_dir = gouda.ensure_dir(log_dir('training_weights'))
        train_args = {'epochs': epochs, 'lr_type': lr_type, 'loss_type': loss_type}
        for key in kwargs:
            train_args[key] = kwargs[key]
        if reduce_lr_on_plateau:
            if 'plateau_factor' not in train_args:
                train_args['plateau_factor'] = 0.1
            if 'plateau_patience' not in train_args:
                train_args['plateau_patience'] = 3
        if load_args:
            if args_path.exists():
                train_args = gouda.load_json(args_path)
            else:
                raise ValueError("No training args file found at `{}`".format(args_path.abspath))
        for item in train_data.take(1):
            train_args['batch_size'] = item[0].numpy().shape[0]
        for item in val_data.take(1):
            train_args['val_batch_size'] = item[0].numpy().shape[0]

        self.compile_model(checking=True)
        if starting_epoch == 1:
            self.save_weights(weights_dir('model_weights_init.tf').abspath)

        # Set learning rate type and optimizer
        optimizer = self._setup_optimizer(train_args)

        # Set loss type
        if train_args['loss_type'] is None:
            if self.model_args['loss_type'] is None:
                raise ValueError("No loss function defined")
            train_args['loss_type'] = self.model_args['loss_type']
        if isinstance(train_args['loss_type'], str):
            loss = get_loss_func(train_args['loss_type'])(**train_args)
        else:
            loss = train_args['loss_type']
            train_args['loss_type'] = 'custom'

        # Currently, just uses the default training/validation steps

        # Save training args as json
        save_args = train_args.copy()
        for key in train_args:
            key_type = str(type(save_args[key]))
            if 'function' in key_type or 'class' in key_type:
                save_args[key] = 'custom'
        gouda.save_json(save_args, args_path)

        checkpoint_prefix = weights_dir('model_weights_e{epoch}').abspath
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir.abspath),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
        ]
        if reduce_lr_on_plateau:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                  factor=train_args['plateau_factor'],
                                                                  patience=train_args['plateau_patience']))

        with self.strategy.scope():
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        try:
            self.model.fit(train_data, validation_data=val_data, epochs=epochs, initial_epoch=starting_epoch - 1, callbacks=callbacks)
        except KeyboardInterrupt:
            print("\nInterrupting model training...")
        self.save_weights(log_dir('model_weights.tf').abspath)
