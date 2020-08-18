# Uses tensor_env

import gouda
import tensorflow as tf  # tensorflow 2.0 or greater
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
import tensorflow.keras.backend as K
import tqdm

from .constants import PROJECT_DIR  # path to lungseg
from .constants import RESULTS_DIR  # path to saved model weights/results
from .general_utils import StableCounter
from .tf_models import lookup_model


class BaseModel(object):
    def __init__(self,
                 model_name='default',
                 model_group='default',
                 model_type='template',
                 filter_scale=0,
                 num_outputs=2,
                 input_shape=[512, 512, 1],
                 load_args=False,
                 **kwargs
                 ):
        """Initialize a model for the network

        Parameters
        ----------
        model_name : str
            The name of the model to use - should define the model level parameters
        model_group : str
            The group of models to use - should define the model structure or data paradigm
        filter_scale : int
            The scaling factor to use for the model layers (scales by powers of 2)
        input_shape : tuple of ints
            The shape of data to be passed to the model (not including batch size)
        load_args : bool
            Whether to use pre-existing arguments for the given model group+name
        """
        if input_shape is None and load_args is False:
            raise ValueError("Input shape cannot be None for model object")
        self.model_dir = gouda.GoudaPath(gouda.ensure_dir(RESULTS_DIR, model_group, model_name))
        if load_args:
            if self.model_dir('model_args.json').exists():
                self.model_args = gouda.load_json(self.model_dir('model_args.json'))
            else:
                raise ValueError("Cannot find model args for model {}/{}".format(model_group, model_name))
        else:
            self.model_args = {
                'model_name': model_name,
                'model_group': model_group,
                'model_type': model_type,
                'filter_scale': filter_scale,
                'input_shape': input_shape,
            }
            for key in kwargs:
                self.model_args[key] = kwargs[key]
            gouda.save_json(self.model_args, self.model_dir('model_args.json'))
        K.clear_session()
        self.model = lookup_model(model_type)(**self.model_args)

    def load_weights(self, version=None, path=None):
        if path is None:
            path = self.model_dir / version / 'model_weights.h5'
        path = gouda.GoudaPath(path)
        if path.exists():
            self.model.load_weights(path)
        else:
            raise ValueError("No weights found at: {}".format(path.abspath))

    def __call__(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def get_model(self):
        return self.model

    def list_versions(self):
        return sorted(self.model_dir.children(dirs_only=True, basenames=True))

    def train(self,
              train_data,
              val_data,
              starting_epoch=1,
              epochs=200,
              save_every=-1,
              version_name='default',
              load_args=False,
              **kwargs
              ):
        """Train the model

        Parameters
        ----------
        train_data : tf.data.Dataset
            The data to train on
        val_data : tf.data.Dataset
            The data to validate on
        starting_epoch : int
            The epoch to start on - can be set greater than 1 to continue previous training
        epochs : int
            The epoch to end on - if starting epoch is greater than 1, the model will still only train until it reaches this total epoch count
        save_every : int
            The number of epochs between each set of model weights to save
        version_name : str
            The name of the model to train - version name should group training/hyperparameters
        load_args : bool
            Whether to use pre-existing parameters for the given model group+name+version
        """
        version_dir = gouda.ensure_dir(self.model_dir / version_name)
        weights_dir = gouda.ensure_dir(version_dir / 'training_weights')
        if load_args:
            if version_dir('train_args.json').exists():
                train_args = gouda.load_json(version_dir('train_args.json'))
            else:
                raise ValueError("No existing args found for {}/{}/{}".format(self.model_args['model_group'], self.model_args['model_name'], version_name))
        else:
            defaults = {
                'learning_rate': 1e-4,
                'lr_decay_steps': None,
                'lr_decay_rate': None,
                'label_smoothing': 0.05
            }
            train_args = kwargs
            for item in defaults:
                if item not in train_args:
                    train_args[item] = defaults[item]

        for x, y in train_data.take(1):
            batch_size = y.numpy().shape[0]
        for x, y in val_data.take(1):
            val_batch_size = y.numpy().shape[0]

        train_args['batch_size'] = batch_size
        train_args['val_batch_size'] = val_batch_size

        gouda.save_json(train_args, version_dir / 'train_args.json')

        if train_args['lr_decay_rate'] is not None and train_args['lr_decay_steps'] is not None:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=train_args['learning_rate'],
                decay_steps=train_args['lr_decay_steps'],
                decay_rate=train_args['lr_decay_rate']
            )
        else:
            lr = train_args['learning_rate']
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=train_args['label_smoothing'])

        # set up tensorboard

        train_writer = tf.summary.create_file_writer(version_dir / 'train')
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_acc = tf.keras.metrics.BinaryAccuracy('train_accuracy')

        val_writer = tf.summary.create_file_writer(version_dir / 'val')
        val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        val_acc = tf.keras.metrics.BinaryAccuracy('val_accuracy')

        # set up train/val steps
        def train_step(model, optimizer, x, y):
            with tf.GradientTape() as tape:
                predicted = model(x, training=True)
                loss = loss_func(y, predicted)
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

            train_loss(loss)
            train_acc(y, predicted)

        def val_step(model, x, y):
            predicted = model(x, training=False)
            loss = loss_func(y, predicted)
            val_loss(loss)
            val_acc(y, predicted)

        # training loop
        epoch_pbar = tqdm.tqdm(total=epochs, unit=' epochs', initial=starting_epoch)

        # Baseline Validation
        val_steps = StableCounter()
        val_batch_pbar = tqdm.tqdm(total=None, unit=' val samples', leave=False)
        for image, label in val_data:
            val_step(self.model, image, label)
            val_batch_pbar.update(val_batch_size)
            val_steps += val_batch_size
        val_steps.stop()
        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=0)
            tf.summary.scalar('accuracy', val_acc.result(), step=0)
        log_string = 'Untrained: Val Loss: {:.4f}, Val Accuracy: {:6.2f}'.format(val_loss.result(), val_acc.result() * 100)
        val_batch_pbar.close()
        val_loss.reset_states()
        val_acc.reset_states()
        epoch_pbar.write(log_string)

        train_steps = StableCounter()
        self.model.save_weights(weights_dir('initial_weights.h5').abspath)
        try:
            for epoch in range(starting_epoch, epochs):
                train_batch_pbar = tqdm.tqdm(total=train_steps(), unit=' samples', leave=False)
                for image, label in train_data:
                    train_step(self.model, opt, image, label)
                    train_batch_pbar.update(batch_size)
                    train_steps += batch_size
                train_steps.stop()
                with train_writer.as_default():
                    if not isinstance(lr, float):
                        tf.summary.scalar('lr', lr(opt.iterations), step=epoch)
                    tf.summary.scalar('loss', train_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', train_acc.result(), step=epoch)
                train_batch_pbar.close()
                log_string = 'Epoch {:04d}, Loss: {:.4f}, Accuracy: {:6.2f}'.format(epoch, train_loss.result(), train_acc.result() * 100)

                val_batch_pbar = tqdm.tqdm(total=val_steps())
                for image, label in val_data:
                    val_step(self.model, image, label)
                    val_batch_pbar.update(val_batch_size)
                with val_writer.as_default():
                    tf.summary.scalar('loss', val_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', val_acc.result(), step=epoch)
                val_batch_pbar.close()
                log_string += ' || Val Loss: {:.4f}, Val Accuracy: {:6.2f}'.format(val_loss.result(), val_acc.result() * 100)
                if (epoch + 1) % save_every == 0 and save_every != -1:
                    self.model.save_weights(weights_dir('model_weights_e{:03d}.h5'.format(epoch)).abspath)

                epoch_pbar.write(log_string)
                train_loss.reset_states()
                train_acc.reset_states()
                val_loss.reset_states()
                val_acc.reset_states()
                epoch_pbar.update(1)
        except KeyboardInterrupt:
            epoch_pbar.write('Stopping model with keypress...')
        epoch_pbar.close()
        self.model.save_weights(version_dir('model_weights.h5').abspath)
