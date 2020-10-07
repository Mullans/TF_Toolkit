import gouda
import os
import sys
import tensorflow as tf
import tensorflow.keras.backend as K

# Try to import progress bar module
try:
    from tqdm import auto as tqdm
except ModuleNotFoundError:
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

# assumes this is in a child directory (usually ./scripts) of the main project
# PROJECT_DIR = os.path.realpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))

#
# def train_step_func(model, optimizer, loss_func, logging_handler):
#     def train_step(inputs):
#         x, y = inputs
#         with tf.GradientTape() as tape:
#             predictions = model(x, training=True)
#             loss = loss_func(y, predictions)
#         grad = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grad, model.trainable_variables))
#         logging_handler.train_step((y, predictions, loss))
#         return loss
#     return tf.function(train_step)
#
#
# def val_step_func(model, loss_func, logging_handler):
#     def val_step(inputs):
#         x, y = inputs
#         predictions = model(x, training=False)
#         loss = loss_func(y, predictions)
#         logging_handler.val_step((y, predictions, loss))
#         return loss
#     return tf.function(val_step)
#
#
# def distributed_train_step_func(model, optimizer, loss_func, logging_handler, batch_size=32):
#     """Default distributed train step for single input, single output, single loss models"""
#     def train_step(inputs):
#         x, y = inputs
#         with tf.GradientTape() as tape:
#             predictions = model(x, training=True)
#             loss = loss_func(y, predictions)
#         grad = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grad, model.trainable_variables))
#         logging_handler.train_step((y, predictions, loss / batch_size))
#         return loss
#     return tf.function(train_step)
#
#
# def distributed_val_step_func(model, loss_func, logging_handler, batch_size=32):
#     """Default distributed validation step for single input, single output, single loss models"""
#     def val_step(inputs):
#         x, y = inputs
#         predictions = model(x, training=False)
#         loss = loss_func(y, predictions)
#         logging_handler.val_step((y, predictions, loss / batch_size))
#         return loss
#     return tf.function(val_step)


class CoreModel(object):
    def __init__(self,
                 model_group='default',
                 model_name='default',
                 project_dir=None,
                 load_args=False,
                 distributed=False,
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
            'loss_type': None,
            'lr_type': 'base'
        }
        for key in kwargs:
            self.model_args[key] = kwargs[key]

        self.is_distributed = distributed
        self.model = None

        self.results_dir = gouda.GoudaPath(os.path.join(project_dir, 'Results'))
        gouda.ensure_dir(self.results_dir)
        group_dir = self.results_dir / model_group
        gouda.ensure_dir(group_dir)
        self.model_dir = group_dir / model_name
        gouda.ensure_dir(self.model_dir)
        args_path = self.model_dir('model_args.json')
        if load_args:
            if args_path.exists():
                loaded_args = gouda.load_json(args_path)
                for key in loaded_args:
                    self.model_args[key] = loaded_args[key]
            else:
                raise ValueError('No file found at {}'.format(args_path.abspath))
        if not args_path.exists():
            self.save_args()

    def save_args(self):
        save_args = self.model_args.copy()
        for key in save_args:
            if 'function' in str(type(save_args[key])) or 'CategoricalCrossentropy' in str(type(save_args[key])):
                save_args['key'] = 'custom'

        gouda.save_json(save_args, self.model_dir / 'model_args.json')

    def clear(self):
        K.clear_session()

    def compile_model(self, model_func=None, **kwargs):
        if self.model_args['model_func'] is None and model_func is None:
            raise ValueError('No selected model function')
        for key in kwargs:
            self.model_args[key] = kwargs[key]

        if model_func is None:
            model_func = self.model_args['model_func']
        if isinstance(model_func, str):
            model_func = get_model_func(model_func)

        self.model = model_func(**self.model_args)

    def load_weights(self, group_name=None, model_name=None, version=None, path=None):
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
                raise ValueError("No saved weights found for version `{}`".format(version))
        if isinstance(path, gouda.GoudaPath):
            path = path.abspath
        self.model.load_weights(path)

    @property
    def lr_type(self):
        if isinstance(self.model_args['lr_type'], str):
            return self.model_args['lr_type']
        else:
            return 'custom'

    @lr_type.setter
    def lr_type(self, lr_type):
        self.model_args['lr_type'] = lr_type

    @property
    def loss_type(self):
        if isinstance(self.model_args['loss_type'], str):
            return self.model_args['loss_type']
        else:
            return 'custom'

    @loss_type.setter
    def loss_type(self, loss_type):
        self.model_args['loss_type'] = loss_type

    @property
    def train_step(self):
        if isinstance(self.model_args['train_step'], str):
            return self.model_args['train_step']
        else:
            return 'custom'

    @train_step.setter
    def train_step(self, train_step):
        self.model_args['train_step'] = train_step

    @property
    def val_step(self):
        if isinstance(self.model_args['val_step'], str):
            return self.model_args['val_step']
        else:
            return 'custom'

    @val_step.setter
    def val_step(self, val_step):
        self.model_args['val_step'] = val_step

    def summary(self, **kwargs):
        return self.model.summary(**kwargs)

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

    def train(self,
              train_data,
              val_data,
              logging_handler,
              starting_epoch=1,
              lr_type=None,
              loss_type=None,
              epochs=50,
              save_every=10,
              load_args=False,
              version='default',
              **kwargs):
        log_dir = gouda.ensure_dir(self.model_dir(version))
        args_path = log_dir('training_args.json')
        weights_dir = gouda.ensure_dir(log_dir('training_weights'))

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

        if self.model is None:
            self.compile_model()
        if starting_epoch == 1:
            self.model.save_weights(weights_dir('model_weights_init.tf').abspath)

        # Set learning rate type
        if train_args['lr_type'] is None:
            if self.model_args['lr_type'] is None:
                train_args['lr_type'] = 'base'
            else:
                train_args['lr_type'] = self.model_args['lr_type']
        if isinstance(train_args['lr_type'], str):
            lr = get_lr_func(train_args['lr_type'])(**train_args)
        else:
            lr = train_args['lr_type']
            train_args['lr_type'] = custom
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

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

        # Set training step
        train_args['train_step'] = self.model_args['train_step']
        if isinstance(train_args['train_step'], str):
            train_step = get_update_step(train_args['train_step'], is_training=True)
        else:
            train_step = train_args['train_step']
            train_args['train_step'] = 'custom'

        # Set validation step
        train_args['val_step'] = self.model_args['val_step']
        if isinstance(train_args['val_step'], str):
            val_step = get_update_step(train_args['val_step'], is_training=False)
        else:
            val_step = train_args['val_step']
            train_args['val_step'] = 'custom'

        # Save training args as json
        save_args = train_args.copy()
        for key in save_args:
            if 'function' in str(type(save_args[key])):
                save_args['key'] = 'custom'
        gouda.save_json(train_args, args_path)

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
                    self.model.save_weights(weights_dir(weight_string.format(epoch)).abspath)

        except KeyboardInterrupt:
            print("Interrupting model training...")
            logging_handler.interrupt()
        epoch_pbar.close()
        self.model.save_weights(log_dir('model_weights.tf').abspath)
        logging_handler.stop()

    def train_distributed(self,
                          train_data,
                          val_data,
                          logging_handler,
                          starting_epoch=1,
                          lr_type=None,
                          loss_type=None,
                          epochs=50,
                          save_every=10,
                          load_args=False,
                          version='default',
                          **kwargs):
        """https://www.tensorflow.org/tutorials/distribute/custom_training#training_loop"""
        raise NotImplementedError("This is still under construction. Need to figure out how loss works.")

        log_dir = gouda.ensure_dir(self.model_dir(version))
        args_path = log_dir('distributed_training_args.json')
        weights_dir = gouda.ensure_dir(log_dir('distributed_training_weights'))

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

        lr = get_lr_func(train_args['lr_type'])(**train_args)

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.compile_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            logging_handler.start(log_dir, total_epochs=epochs)
            # loss = get_loss_func(train_args['loss_type'])(reduction='none', **train_args)
            # loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

            # def distributed_loss(y_true, y_pred):
            #     # example_loss = loss(y_true, y_pred)
            #     # return tf.nn.compute_average_loss(example_loss, global_batch_size=train_args['batch_size'])
            #     example_loss = loss_object(y_true, y_pred)
            #     # example_loss = tf.keras.losses.binary_crossentropy(y_true, tf.expand_dims(y_pred, 1))
            #     return tf.nn.compute_average_loss(example_loss, global_batch_size=train_args['batch_size'])
            loss_object = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                reduction='none')

            def distributed_loss(labels, predictions):
                per_example_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=32)

        if starting_epoch == 1:
            self.model.save_weights(weights_dir('model_weights_init.tf').abspath)

        train_counter = StableCounter()
        if 'train_steps' in train_args:
            train_counter.set(train_args['train_steps'])
        val_counter = StableCounter()
        if 'val_steps' in train_args:
            val_counter.set(train_args['val_steps'])

        train_step = self.train_step(self.model, optimizer, distributed_loss, logging_handler, batch_size=train_args['batch_size'])
        val_step = self.val_step(self.model, distributed_loss, logging_handler, batch_size=train_args['batch_size'])

        @tf.function
        def distributed_train_step(inputs):
            strategy.experimental_run_v2(train_step, args=(inputs, ))

        @tf.function
        def distributed_val_step(inputs):
            strategy.experimental_run_v2(val_step, args=(inputs, ))

        train_data = strategy.experimental_distribute_dataset(train_data)
        val_data = strategy.experimental_distribute_dataset(val_data)

        epoch_pbar = tqdm.tqdm(total=epochs, unit=' epochs', initial=starting_epoch - 1)
        val_pbar = tqdm.tqdm(total=val_counter(), unit=' val samples', leave=False)
        try:
            for item in val_data:
                distributed_val_step(item)
                batch_size = train_args['val_batch_size']
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
                    batch_size = train_args['batch_size']
                    train_counter += batch_size
                    train_pbar.update(batch_size)
                    distributed_train_step(item)
                train_pbar.close()
                val_pbar = tqdm.tqdm(total=val_counter(), leave=False)
                for item in val_data:
                    batch_size = train_args['val_batch_size']
                    val_counter += batch_size
                    val_pbar.update(batch_size)
                    distributed_val_step(item)
                val_pbar.close()
                train_counter.stop()
                val_counter.stop()
                log_string = logging_handler.write(epoch)
                epoch_pbar.write(log_string)
                epoch_pbar.update(1)
                if (epoch + 1) % 10 == 0:
                    weight_string = 'model_weights_e{:0' + epoch_digits + 'd}.tf'
                    self.model.save_weights(weights_dir(weight_string.format(epoch)).abspath)

        except KeyboardInterrupt:
            logging_handler.interrupt()
        self.model.save_weights(log_dir('model_weights.tf').abspath)
        logging_handler.stop()
        epoch_pbar.close()

    def __call__(self, x, *args, training=False, **kwargs):
        return self.model(x, *args, training=training, **kwargs)
