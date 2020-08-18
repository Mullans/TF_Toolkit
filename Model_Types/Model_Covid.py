import gouda
import os
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import tqdm

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
AUTOTUNE = tf.data.experimental.AUTOTUNE

from .architecture import get_model
from .metrics import BalanceMetric, MatthewsCorrelationCoefficient
from .losses import get_loss_func
from .parameters import project_path
from .utils import StableCounter


class BinaryModel(object):
    def __init__(self,
                 model_name='default',
                 model_group='default',
                 model_type=None,
                 filter_scale=0,
                 input_shape=None,  # TBD
                 load_args=False,
                 **kwargs
                 ):
        self.model_args = {
            'model_name': model_name,
            'model_group': model_group,
            'model_type': model_type,
            'filter_scale': filter_scale,
            'input_shape': input_shape
        }
        for key in kwargs:
            self.model_args[key] = kwargs[key]

        self.model_dir = gouda.ensure_dir(project_path('results', model_group, model_name))

        args_path = self.model_dir('model_args.json')
        if load_args:
            if args_path.exists():
                self.model_args = gouda.load_json(args_path)
                self.model_args['model_name'] = model_name
                self.model_args['model_group'] = model_group
            else:
                raise ValueError("Cannot load model args. No file found at: {}".format(args_path.path))
        if not args_path.exists():
            # Only models without pre-existing model args will save in order to prevent overwriting
            gouda.save_json(self.model_args, args_path)

        self.compile()

    def save_args(self):
        """Manually save the model args - this will overwrite any existing args"""
        args_path = self.model_dir('model_args.json')
        gouda.save_json(self.model_args, args_path)

    def compile(self):
        """Clear the current graph and initialize a new model"""
        K.clear_session()
        model_func = get_model(self.model_args['model_type'])
        self.model = model_func(**self.model_args)

    def load_weights(self, model_version=None, weights_path=None):
        """Load model weights from either a version of the current group/model or from a file path"""
        if weights_path is None:
            if model_version is None:
                raise ValueError("model_version must be specified if weights_path is not used")
            weights_path = project_path('results',
                                        self.model_args['model_group'],
                                        self.model_args['model_name'],
                                        model_version,
                                        'model_weights.h5')
            if not weights_path.exists():
                raise ValueError("No file found at {}".format(weights_path.abspath))
        weights_path = gouda.GoudaPath(weights_path)
        self.model.load_weights(weights_path.abspath)

    def train(
        self,
        train_data,
        val_data,
        num_train_samples=None,
        num_val_samples=None,
        starting_epoch=1,
        epochs=50,
        model_version='default',
        load_args=False,
        plot_model=True,
        learning_rate=1e-4,
        label_smoothing=0.1,
        loss_func='mixed',
        save_every=10,
        **kwargs
    ):
        # Set up directories
        log_dir = gouda.ensure_dir(self.model_dir(model_version))
        args_path = log_dir('training_args.json')
        weights_dir = gouda.ensure_dir(log_dir('training_weights'))

        # Set up training args
        train_args = {
            'learning_rate': learning_rate,
            'label_smoothing': label_smoothing,
            'loss_function': loss_func,
            'lr_exp_decay_rate': None,  # Multiplier for exponential decay (ie 0.2 means lr_2 = 0.2 * lr_1)
            'lr_exp_decay_steps': None,  # Steps between exponential lr decay
            'lr_cosine_decay': False,  # Whether to use cosine lr decay
            'lr_cosine_decay_steps': epochs,  # The number of steps to reach the minimum lr
            'lr_cosine_decay_min_lr': 0.0  # The minimum lr when using steps < epochs
        }
        for key in kwargs:
            train_args[key] = kwargs[key]
        if load_args:
            if args_path.exists():
                train_args = gouda.load_json(args_path.abspath)
            else:
                raise ValueError("Cannot load training args. No file found at: {}".format(args_path.path))

        for x, y in train_data.take(1):
            train_args['train_batch_size'] = y.numpy().shape[0]
        for x, y in val_data.take(1):
            train_args['val_batch_size'] = y.numpy().shape[0]

        gouda.save_json(train_args, args_path)

        # Save initial weights and model structure
        if self.model is None:
            self.compile()
        self.model.save_weights(weights_dir('model_weights_init.h5').abspath)
        if plot_model:
            tf.keras.utils.plot_model(self.model, to_file=log_dir('model.png').abspath, show_shapes=True)

        # Set up loss
        if train_args['lr_exp_decay_rate'] is not None and train_args['lr_exp_decay_steps'] is not None:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=train_args['learning_rate'],
                decay_steps=train_args['lr_exp_decay_steps'],
                decay_rate=train_args['lr_exp_decay_rate']
            )
        elif train_args['lr_cosine_decay']:
            alpha = train_args['lr_cosine_decay_min_lr'] / train_args['learning_rate']
            lr = tf.keras.experimental.CosineDecay(train_args['learning_rate'], train_args['lr_cosine_decay_steps'], alpha=alpha)
        else:
            lr = train_args['learning_rate']
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_func = get_loss_func(train_args['loss_function'], **train_args)
        # if train_args['loss_function'] == 'bce':
        #     loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=train_args['label_smoothing'])
        # elif train_args['loss_function'] == 'iou':
        #     loss_func = IOU_loss
        # elif train_args['loss_function'] == 'mixed':
        #     loss_func = mixed_IOU_BCE_loss(train_args['loss_alpha'], train_args['label_smoothing'])
        # elif
        # else:
        #     raise NotImplementedError("Loss function `{}` hasn't been added yet".format(train_args['loss_function']))

        # Set up logging
        train_writer = tf.summary.create_file_writer(log_dir('train').path)
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_acc = tf.keras.metrics.BinaryAccuracy('train_accuracy')
        train_bal = BalanceMetric('train_balance')
        train_string = "Loss: {:.4f}, Accuracy: {:6.2f}, Balance: {:.2f}"

        val_writer = tf.summary.create_file_writer(log_dir('val').path)
        val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        val_acc = tf.keras.metrics.BinaryAccuracy('val_accuracy')
        val_bal = BalanceMetric('val_balance')
        val_string = " || Val Loss: {:.4f}, Val Accuracy: {:6.2f}, Val Balance: {:.2f}"

        # Define train/val steps
        def train_step(model, optimizer, x, y):
            with tf.GradientTape() as tape:
                predicted = model(x, training=True)
                loss = loss_func(y, predicted)
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

            train_loss(loss)
            train_acc(y, predicted)
            train_bal(y, predicted)

        def val_step(model, x, y):
            predicted = model(x, training=False)
            loss = loss_func(y, predicted)
            val_loss(loss)
            val_acc(y, predicted)
            val_bal(y, predicted)

        train_step = tf.function(train_step)
        val_step = tf.function(val_step)

        train_steps = StableCounter()
        if num_train_samples is not None:
            train_steps.set(num_train_samples)
        val_steps = StableCounter()
        if num_val_samples is not None:
            val_steps.set(num_val_samples)

        # Training loop
        epoch_pbar = tqdm.tqdm(total=epochs, unit=' epochs', initial=starting_epoch)
        val_batch_pbar = tqdm.tqdm(total=val_steps(), unit=' val samples', leave=False)
        for image, label in val_data:
            val_step(self.model, image, label)
            val_batch_pbar.update(train_args['val_batch_size'])
            val_steps += train_args['val_batch_size']
        val_steps.stop()
        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=0)
            tf.summary.scalar('accuracy', val_acc.result(), step=0)
            tf.summary.scalar('balance', val_bal.result(), step=0)
        epoch_pbar.write('Pretrained - Val Loss: {:.4f}, Val Accuracy: {:6.2f}, Val Balance: {:.2f}'.format(val_loss.result(), 100 * val_acc.result(), val_bal.result()))
        val_batch_pbar.close()
        val_loss.reset_states()
        val_acc.reset_states()
        val_bal.reset_states()

        try:
            for epoch in range(starting_epoch, epochs):
                log_string = 'Epoch {:3d} - '.format(epoch)
                # Training loop
                train_batch_pbar = tqdm.tqdm(total=train_steps(), unit=' samples', leave=False)
                for image, label in train_data:
                    train_step(self.model, opt, image, label)
                    train_batch_pbar.update(train_args['train_batch_size'])
                    train_steps += train_args['train_batch_size']
                with train_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', train_acc.result(), step=epoch)
                    tf.summary.scalar('balance', train_bal.result(), step=epoch)
                log_string += train_string.format(train_loss.result(), train_acc.result() * 100, train_bal.result())
                train_batch_pbar.close()

                # Validation Loop
                val_batch_pbar = tqdm.tqdm(total=val_steps(), unit=' val samples', leave=False)
                for image, label in val_data:
                    val_step(self.model, image, label)
                    val_batch_pbar.update(train_args['val_batch_size'])
                with val_writer.as_default():
                    tf.summary.scalar('loss', val_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', val_acc.result(), step=epoch)
                    tf.summary.scalar('balance', val_bal.result(), step=epoch)
                log_string += val_string.format(val_loss.result(), val_acc.result() * 100, val_bal.result())
                val_batch_pbar.close()

                if (epoch + 1) % 10 == 0:
                    self.model.save_weights(weights_dir('model_weights_e{:03d}.h5'.format(epoch)).path)

                epoch_pbar.write(log_string)
                train_loss.reset_states()
                train_acc.reset_states()
                train_bal.reset_states()
                train_steps.stop()
                val_loss.reset_states()
                val_acc.reset_states()
                val_bal.reset_states()
                epoch_pbar.update(1)
        except KeyboardInterrupt:
            print("Interrupting training...")
        self.model.save_weights(log_dir('model_weights.h5').path)
        epoch_pbar.close()

    def __call__(self, x, *args, **kwargs):
        return self.model(x, *args, training=False, **kwargs)
