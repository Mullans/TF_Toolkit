import gouda
import json
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import warnings

# Try to import progress bar module
try:
    from tqdm import auto as tqdm
except ModuleNotFoundError:
    warnings.warn('tqdm module not found, defaulting to print statements')
    import dummy_bar as tqdm

# Setup tensorflow
# gpu_devices = tf.config.experimental.list_physical_devices("GPU")
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)
AUTOTUNE = tf.data.experimental.AUTOTUNE

from .losses import get_loss_func
from .model_arch import get_model_func
from .learning_rates import get_lr_func
from .stable_counter import StableCounter
from .train_functions import get_update_step
from .Logging import EmptyLoggingHandler


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


def clean_for_json(x):
    if isinstance(x, dict):
        return {key: clean_for_json(x[key]) for key in x}
    elif isinstance(x, (list, tuple)):
        old_type = type(x)
        new_data = [clean_for_json(item) for item in x]
        new_data = old_type(new_data)
        return new_data
    else:
        return x if is_jsonable(x) else 'CUSTOM: ' + str(x)


class CoreModel(object):
    def __init__(self,
                 model_group='default',
                 model_name='default',
                 project_dir=None,
                 load_args=False,
                 overwrite_args=False,
                 # distributed=False,
                 **kwargs):
        if project_dir is None:
            # Generally the location the code is called from
            project_dir = os.getcwd()
        K.clear_session()
        self.model_args = {
            'model_name': model_name,
            'model_group': model_group,
            'train_step': 'default',
            'val_step': 'default',
            'lr_type': 'base',
            'model_func': None
        }
        for key in kwargs:
            self.model_args[key] = kwargs[key]

        # self.is_distributed = distributed
        self.model = None

        self.results_dir = gouda.GoudaPath(os.path.join(project_dir, 'Results'))
        gouda.ensure_dir(self.results_dir)
        group_dir = self.results_dir / model_group
        gouda.ensure_dir(group_dir)
        self.model_dir = group_dir / model_name
        gouda.ensure_dir(self.model_dir)
        args_path = self.model_dir('model_args.json')
        if load_args:
            self.load_args(args_path)
        if overwrite_args or not args_path.exists():
            self.save_args()

    def load_args(self, args_path):
        """Load model arguments from a json file.

        NOTE
        ----
        Custom methods/models/etc will be loaded with a value of 'custom' and should be replaced
        """
        if os.path.exists(args_path):
            to_warn = []
            loaded_args = gouda.load_json(args_path)
            for key in loaded_args:
                self.model_args[key] = loaded_args[key]
                if isinstance(loaded_args[key], str) and loaded_args[key].startswith('CUSTOM: '):
                    to_warn.append(key)
            if len(to_warn) > 0:
                warnings.warn('Custom arguments for [{}] were found and should be replaced'.format(', '.join(to_warn)))
        else:
            raise ValueError('No file found at {}'.format(args_path))

    def save_args(self):
        """Save model arguments to a json file in the model directory

        NOTE
        ----
        Custom methods/models/etc will be saved with a value of 'custom' in the output file
        """
        save_args = self.model_args.copy()
        save_args = clean_for_json(save_args)
        gouda.save_json(save_args, self.model_dir / 'model_args.json')

    def clear(self):
        K.clear_session()

    def compile_model(self, model_func=None, checking=False, **kwargs):
        """Compile the model for the given function.

        Parameters
        ----------
        model_func: str | func
            Either the string to lookup, the model function, or None if you want to use the function from self.model_args (the default is None)
        checking: bool
            If true, will only compile a model if there is no compiled model already

        Note
        ----
        Compile here means to prepare the model/weights not compile like the keras model.compile() method.
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

        self.model = model_func(**self.model_args)

    def load_weights(self, group_name=None, model_name=None, version=None, path=None, **kwargs):
        """Load weights from an existing trained model.

        Paramters
        ---------
        group_name : None | str
            The name of the model group to load from (the default is None)
        model_name : None | str
            The name of the model to load from (the default is None)
        version : None | str
            The name of the model version to load weights from (the default is None)
        path : None | str
            The full path to the model weights to load. Path takes priority over
            version if both are present.

        Note
        ----
        Either version or path must be set to load weights. If either group_name
        or model_name are None, then model weights from the same group/name will
        be used.
        """
        # TODO Add a search through training weights (set desired epoch)
        if self.model is None:
            self.compile_model()
        if path is None and version is None:
            raise ValueError('Either version or path must be specified to load weights')
        if path is None:
            if group_name is None or model_name is None:
                path = self.model_dir / version / 'model_weights'
            else:
                path = self.results_dir / group_name / model_name / version / 'model_weights'
            found_tf = os.path.exists(path + '.tf.index')
            found_h5 = os.path.exists(path + '.h5')
            if found_tf and found_h5:
                raise ValueError("Multiple weight formats found for version `{}`. Please specify path".format(version))
            elif found_tf:
                path = path + '.tf'
            elif found_h5:
                path = path + '.h5'
            else:
                raise FileNotFoundError("No saved weights found for version `{}`".format(version))
        if isinstance(path, os.PathLike):
            path = str(path)
        if path.endswith('.index'):
            path = path.replace('.index', '')
        self.model.load_weights(path, **kwargs)

    def list_versions(self):
        return self.model_dir.children(dirs_only=True, basenames=True)

    def save_weights(self, path):
        if isinstance(path, gouda.GoudaPath):
            path = path.abspath
        if self.model is None:
            raise AttributeError('Weights cannot be saved before model is initialized')
        self.model.save_weights(path)

    def export_model(self, path):
        if isinstance(path, os.PathLike):
            path = str(path)
        tf.saved_model.save(self.model, path)

    @property
    def lr_type(self):
        if isinstance(self.model_args['lr_type'], str):
            return self.model_args['lr_type']
        else:
            return 'CUSTOM: ' + str(self.model_args['lr_type'])

    @lr_type.setter
    def lr_type(self, lr_type):
        self.model_args['lr_type'] = lr_type

    @property
    def loss_type(self):
        if 'loss_type' not in self.model_args:
            return None
        elif isinstance(self.model_args['loss_type'], str):
            return self.model_args['loss_type']
        else:
            return 'CUSTOM: ' + str(self.model_args['loss_type'])

    @loss_type.setter
    def loss_type(self, loss_type):
        self.model_args['loss_type'] = loss_type

    @property
    def train_step(self):
        if isinstance(self.model_args['train_step'], str):
            return self.model_args['train_step']
        else:
            return 'CUSTOM: ' + str(self.model_args['train_step'])

    @train_step.setter
    def train_step(self, train_step):
        self.model_args['train_step'] = train_step

    @property
    def val_step(self):
        if isinstance(self.model_args['val_step'], str):
            return self.model_args['val_step']
        else:
            return 'CUSTOM: ' + str(self.model_args['val_step'])

    @val_step.setter
    def val_step(self, val_step):
        self.model_args['val_step'] = val_step

    @property
    def layers(self):
        if self.model is None:
            return None
        return self.model.layers

    def __getattr__(self, attr):
        """Use TF model parameters if this class doesn't have it - subclass without subclassing"""
        if self.model is not None and hasattr(self.model, attr):
            return getattr(self.model, attr)
        raise AttributeError("'CoreModel' object has no attribute '{}'".format(attr))

    def num_parameters(self):
        total = 0
        for var in self.model.variables:
            total += tf.size(var)
        return total

    def num_trainable_paramters(self):
        total = 0
        for var in self.model.trainable_variables:
            total += tf.size(var)
        return total

    def plot_model(self):
        if self.model is None:
            raise ValueError("Compile the model before plotting")
        tf.keras.utils.plot_model(self.model, to_file=self.model_dir('model.png').abspath, show_shapes=True)

    def _setup_optimizer(self, train_args):
        if train_args['lr_type'] is None:
            if self.model_args['lr_type'] is None:
                train_args['lr_type'] = 'base'
            else:
                train_args['lr_type'] = self.model_args['lr_type']
        if isinstance(train_args['lr_type'], str):
            lr = get_lr_func(train_args['lr_type'])(**train_args)
        else:
            lr = train_args['lr_type']
            train_args['lr_type'] = 'CUSTOM: ' + str(train_args['lr_type'])
        return tf.keras.optimizers.Adam(learning_rate=lr)

    def train(self,
              train_data,
              val_data,
              logging_handler=None,
              starting_epoch=1,
              lr_type=None,
              loss_type=None,
              epochs=50,
              save_every=10,
              load_args=False,
              sample_callback=None,
              version='default',
              **kwargs):
        log_dir = gouda.ensure_dir(self.model_dir(version))
        args_path = log_dir('training_args.json')
        weights_dir = gouda.ensure_dir(log_dir('training_weights'))

        if logging_handler is None:
            logging_handler = EmptyLoggingHandler()

        train_args = {'epochs': epochs, 'lr_type': lr_type, 'loss_type': loss_type}
        for key in kwargs:
            train_args[key] = kwargs[key]
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
            if 'loss_type' not in self.model_args or self.model_args['loss_type'] is None:
                raise ValueError("No loss function defined. Use keyword 'loss_type' to define loss in model or training arguments.")
            train_args['loss_type'] = self.model_args['loss_type']
        if isinstance(train_args['loss_type'], str):
            loss = get_loss_func(train_args['loss_type'])(**train_args)
        else:
            loss = train_args['loss_type']
            train_args['loss_type'] = str(loss)

        # Set training step
        train_args['train_step'] = self.model_args['train_step']
        if isinstance(train_args['train_step'], str):
            train_step = get_update_step(train_args['train_step'], is_training=True)
        else:
            train_step = train_args['train_step']
            train_args['train_step'] = 'custom__train_func'

        # Set validation step
        train_args['val_step'] = self.model_args['val_step']
        if isinstance(train_args['val_step'], str):
            val_step = get_update_step(train_args['val_step'], is_training=False)
        else:
            val_step = train_args['val_step']
            train_args['val_step'] = 'custom_val_func'

        # Save training args as json
        save_args = clean_for_json(train_args.copy())
        gouda.save_json(save_args, args_path)

        # Start loggers
        logging_handler.start(log_dir, total_epochs=epochs)
        train_counter = StableCounter()
        if 'train_steps' in train_args:
            train_counter.set(train_args['train_steps'])
        val_counter = StableCounter()
        if 'val_steps' in train_args:
            val_counter.set(train_args['val_steps'])

        train_step = train_step(self.model, optimizer, loss, logging_handler)
        val_step = val_step(self.model, loss, logging_handler)

        epoch_pbar = tqdm.tqdm(total=epochs, unit=' epochs', initial=starting_epoch - 1)
        val_pbar = tqdm.tqdm(total=val_counter(), unit=' val samples', leave=False)
        try:
            for item in val_data:
                val_step(item)
                # logging_handler.val_step(val_step(item))
                batch_size = item[0].shape[0]
                val_counter += batch_size
                val_pbar.update(batch_size)
            log_string = logging_handler.write('Pretrained')
            epoch_pbar.write(log_string)
            val_counter.stop()
        except KeyboardInterrupt:
            if 'val_steps' not in train_args:
                val_counter.reset()
            logging_handler.write("Skipping pre-training validation.")
        epoch_pbar.update(1)
        val_pbar.close()

        try:
            epoch_digits = str(gouda.num_digits(epochs))
            for epoch in range(starting_epoch, epochs):
                train_pbar = tqdm.tqdm(total=train_counter(), unit=' samples', leave=False)
                for item in train_data:
                    batch_size = item[0].shape[0]
                    train_counter += batch_size
                    train_pbar.update(batch_size)
                    train_step(item)
                train_pbar.close()
                if sample_callback is not None:
                    sample_callback(self.model, epoch)
                val_pbar = tqdm.tqdm(total=val_counter(), leave=False)
                for item in val_data:
                    batch_size = item[0].shape[0]
                    val_counter += batch_size
                    val_pbar.update(batch_size)
                    val_step(item)
                val_pbar.close()
                train_counter.stop()
                val_counter.stop()
                log_string = logging_handler.write(epoch)
                epoch_pbar.write(log_string)
                epoch_pbar.update(1)
                if (epoch + 1) % 10 == 0:
                    weight_string = 'model_weights_e{:0' + epoch_digits + 'd}.tf'
                    self.save_weights(weights_dir(weight_string.format(epoch)).abspath)

        except KeyboardInterrupt:
            print("Interrupting model training...")
            logging_handler.interrupt()
        epoch_pbar.close()
        self.save_weights(log_dir('model_weights.tf').abspath)
        logging_handler.stop()

    # def train_distributed(self,
    #                       train_data,
    #                       val_data,
    #                       logging_handler,
    #                       starting_epoch=1,
    #                       lr_type=None,
    #                       loss_type=None,
    #                       epochs=50,
    #                       save_every=10,
    #                       load_args=False,
    #                       version='default',
    #                       **kwargs):
    #     """https://www.tensorflow.org/tutorials/distribute/custom_training#training_loop"""
    #     raise NotImplementedError("This is still under construction. Need to figure out how loss works.")
    #
    #     log_dir = gouda.ensure_dir(self.model_dir(version))
    #     args_path = log_dir('distributed_training_args.json')
    #     weights_dir = gouda.ensure_dir(log_dir('distributed_training_weights'))
    #
    #     train_args = {'epochs': epochs, 'lr_type': lr_type, 'loss_type': loss_type}
    #     for key in kwargs:
    #         train_args[key] = kwargs[key]
    #     if load_args:
    #         if args_path.exists():
    #             train_args = gouda.load_json(args_path)
    #         else:
    #             raise ValueError("No training args file found at `{}`".format(args_path.abspath))
    #     for item in train_data.take(1):
    #         train_args['batch_size'] = item[0].numpy().shape[0]
    #     for item in val_data.take(1):
    #         train_args['val_batch_size'] = item[0].numpy().shape[0]
    #
    #     lr = get_lr_func(train_args['lr_type'])(**train_args)
    #
    #     strategy = tf.distribute.MirroredStrategy()
    #     with strategy.scope():
    #         self.compile_model()
    #         optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #         logging_handler.start(log_dir, total_epochs=epochs)
    #         # loss = get_loss_func(train_args['loss_type'])(reduction='none', **train_args)
    #         # loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    #
    #         # def distributed_loss(y_true, y_pred):
    #         #     # example_loss = loss(y_true, y_pred)
    #         #     # return tf.nn.compute_average_loss(example_loss, global_batch_size=train_args['batch_size'])
    #         #     example_loss = loss_object(y_true, y_pred)
    #         #     # example_loss = tf.keras.losses.binary_crossentropy(y_true, tf.expand_dims(y_pred, 1))
    #         #     return tf.nn.compute_average_loss(example_loss, global_batch_size=train_args['batch_size'])
    #         loss_object = tf.keras.losses.BinaryCrossentropy(
    #             from_logits=True,
    #             reduction='none')
    #
    #         def distributed_loss(labels, predictions):
    #             per_example_loss = loss_object(labels, predictions)
    #             return tf.nn.compute_average_loss(per_example_loss, global_batch_size=32)
    #
    #     if starting_epoch == 1:
    #         self.model.save_weights(weights_dir('model_weights_init.tf').abspath)
    #
    #     train_counter = StableCounter()
    #     if 'train_steps' in train_args:
    #         train_counter.set(train_args['train_steps'])
    #     val_counter = StableCounter()
    #     if 'val_steps' in train_args:
    #         val_counter.set(train_args['val_steps'])
    #
    #     train_step = self.train_step(self.model, optimizer, distributed_loss, logging_handler, batch_size=train_args['batch_size'])
    #     val_step = self.val_step(self.model, distributed_loss, logging_handler, batch_size=train_args['batch_size'])
    #
    #     @tf.function
    #     def distributed_train_step(inputs):
    #         strategy.experimental_run_v2(train_step, args=(inputs, ))
    #
    #     @tf.function
    #     def distributed_val_step(inputs):
    #         strategy.experimental_run_v2(val_step, args=(inputs, ))
    #
    #     train_data = strategy.experimental_distribute_dataset(train_data)
    #     val_data = strategy.experimental_distribute_dataset(val_data)
    #
    #     epoch_pbar = tqdm.tqdm(total=epochs, unit=' epochs', initial=starting_epoch - 1)
    #     val_pbar = tqdm.tqdm(total=val_counter(), unit=' val samples', leave=False)
    #     try:
    #         for item in val_data:
    #             distributed_val_step(item)
    #             batch_size = train_args['val_batch_size']
    #             val_counter += batch_size
    #             val_pbar.update(batch_size)
    #         log_string = logging_handler.write('Pretrained')
    #         epoch_pbar.write(log_string)
    #         val_counter.stop()
    #     except KeyboardInterrupt:
    #         if 'val_steps' not in train_args:
    #             val_counter.reset()
    #         logging_handler.write("Skipping pre-training validation.")
    #     epoch_pbar.update(1)
    #     val_pbar.close()
    #
    #     try:
    #         epoch_digits = str(gouda.num_digits(epochs))
    #         for epoch in range(starting_epoch, epochs):
    #             train_pbar = tqdm.tqdm(total=train_counter(), unit=' samples', leave=False)
    #             for item in train_data:
    #                 batch_size = train_args['batch_size']
    #                 train_counter += batch_size
    #                 train_pbar.update(batch_size)
    #                 distributed_train_step(item)
    #             train_pbar.close()
    #             val_pbar = tqdm.tqdm(total=val_counter(), leave=False)
    #             for item in val_data:
    #                 batch_size = train_args['val_batch_size']
    #                 val_counter += batch_size
    #                 val_pbar.update(batch_size)
    #                 distributed_val_step(item)
    #             val_pbar.close()
    #             train_counter.stop()
    #             val_counter.stop()
    #             log_string = logging_handler.write(epoch)
    #             epoch_pbar.write(log_string)
    #             epoch_pbar.update(1)
    #             if (epoch + 1) % 10 == 0:
    #                 weight_string = 'model_weights_e{:0' + epoch_digits + 'd}.tf'
    #                 self.model.save_weights(weights_dir(weight_string.format(epoch)).abspath)
    #
    #     except KeyboardInterrupt:
    #         logging_handler.interrupt()
    #     self.model.save_weights(log_dir('model_weights.tf').abspath)
    #     logging_handler.stop()
    #     epoch_pbar.close()

    def __call__(self, x, *args, training=False, **kwargs):
        return self.model(x, *args, training=training, **kwargs)
