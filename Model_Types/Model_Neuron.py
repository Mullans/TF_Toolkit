import gouda
import os
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import tqdm


from . import RESULTS_DIR
from .tf_utils import BalanceMetric, MatthewsCorrelationCoefficient
from .utils import StableCounter
from .model_defs import get_model_func


class NeuronModel(object):
    def __init__(self,
                 model_name='default',
                 model_group='default',
                 model_type='multires',
                 filter_scale=0,
                 out_layers=1,
                 out_classes=2,
                 input_shape=[1024, 1360, 1],
                 patch_out=False,
                 load_args=False,
                 **kwargs
                 ):
        K.clear_session()
        self.loaded = False

        self.model_dir = gouda.GoudaPath(gouda.ensure_dir(RESULTS_DIR, model_group, model_name))
        args_path = self.model_dir / 'model_args.json'
        if load_args:
            if not args_path.exists():
                raise ValueError("No model arguments found at path: {}".format(args_path.abspath))
            self.model_args = gouda.load_json(args_path)
        else:
            self.model_args = {
                'model_name': model_name,
                'model_group': model_group,
                'model_type': model_type,
                'filter_scale': filter_scale,
                'out_layers': out_layers,
                'out_classes': out_classes,
                'input_shape': input_shape,
                'patch_out': patch_out
            }
            for key in kwargs:
                self.model_args[key] = kwargs[key]
            gouda.save_json(self.model_args, args_path)

        model_func = get_model_func(self.model_args['model_type'])
        self.model = model_func(**self.model_args)

    def __call__(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def predict(self, x):
        return self.model(x, training=False)

    def get_model(self):
        return self.model

    def load_weights(self, weights_path=None, version=None):
        if weights_path is None and version is None:
            raise ValueError("Either weights_path or version must be specified to load weights")
        if weights_path is None:
            weights_path = (self.model_dir / version / 'model_weights.h5').abspath
        if not os.path.exists(weights_path):
            raise ValueError("No model weights found at: {}".format(weights_path))
        self.model.load_weights(weights_path)

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
        train_args = kwargs
        for x, y in train_data.take(1):
            batch_size = y.numpy().shape[0]
        for x, y in val_data.take(1):
            val_batch_size = y.numpy().shape[0]

        version_dir = self.model_dir / version_name
        args_path = version_dir / 'training_args.json'
        if load_args:
            if not args_path.exists():
                raise ValueError("No training arguments found at path: {}".format(args_path.abspath))
            train_args = gouda.load_json(args_path)
        else:
            defaults = {
                'learning_rate': 1e-4,
                'lr_decay_rate': None,
                'lr_decay_steps': None,
                'label_smoothing': 0.05,
            }
            train_args['version_name': version_name]
            for key in defaults:
                if key not in train_args:
                    train_args[key] = defaults[key]
            train_args['batch_size'] = batch_size
            gouda.save_json(train_args, args_path)

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

        train_writer = tf.summary.create_file_writer(version_dir / 'train')
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_bal = BalanceMetric('train_balance')
        train_acc = [tf.keras.metrics.BinaryAccuracy('train_accuracy_{}'.format(i)) for i in range(self.model_args['out_classes'])]
        train_mcc = [MatthewsCorrelationCoefficient(name='train_mcc_{}'.format(i)) for i in range(self.model_args['out_classes'])]

        val_writer = tf.summary.create_file_writer(version_dir / 'val')
        val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        val_bal = BalanceMetric('val_balance')
        val_acc = [tf.keras.metrics.BinaryAccuracy('val_accuracy_{}'.format(i)) for i in range(self.model_args['out_classes'])]
        val_mcc = [MatthewsCorrelationCoefficient(name='val_mcc_{}'.format(i)) for i in range(self.model_args['out_classes'])]

        def train_step(model, optimizer, x, y, num_classes):
            with tf.GradientTape() as tape:
                predicted = model(x, training=True)
                loss = loss_func(y, predicted)
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

            train_loss(loss)
            train_bal(y, predicted)
            y_split = tf.split(y, num_classes, axis=-1)
            pred_split = tf.split(predicted, num_classes, axis=-1)
            for acc, mcc, y, pred in zip(train_acc, train_mcc, y_split, pred_split):
                acc(y, pred)
                mcc(y, pred)

        def val_step(model, x, y, num_classes):
            predicted = model(x, training=False)
            loss = loss_func(y, predicted)
            val_loss(loss)
            val_bal(y, predicted)
            y_split = tf.split(y, num_classes, axis=-1)
            pred_split = tf.split(predicted, num_classes, axis=-1)
            for acc, mcc, y, pred in zip(val_acc, val_mcc, y_split, pred_split):
                acc(y, pred)
                mcc(y, pred)

        train_step = tf.function(train_step)
        val_step = tf.function(val_step)

        epoch_pbar = tqdm.tqdm(total=spochs, unit=' epochs', initial=starting_epoch)

        val_steps = StableCounter()
        val_batch_pbar = tqdm.tqdm(total=val_steps(), unit=' val samples', leave=False)
        for image, label in val_data:
            val_step(self.model, image, label, self.model_args['out_classes'])
            val_batch_pbar.update(val_batch_size)
            val_steps += val_batch_size
        val_steps.stop()
        val_batch_pbar.close()

        logstring = 'Untrained: '
        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=0)
            logstring += 'Val Loss: {:.4f}'.format(val_loss.result())
            tf.summary.scalar('balance', val_bal.result(), step=0)
            logstring += ', Val Balance: {:.4f}'.format(val_bal.result())
            accuracies = []
            for i, acc in enumerate(val_acc):
                tf.summary.scalar('accuracy_{}'.format(i), acc.result(), step=0)
                accuracies.append('{:.2f}'.format(acc.result() * 100))
            logstring += ", Val Accuracy " + '/'.join(accuracies)
            mccs = []
            for i, mcc in enumerate(val_mcc):
                tf.summary.scalar('mcc_{}'.format(i), mcc.result(), step=0)
                mccs.append("{:.4f}".format(mcc.result()))
            logstring += ", Val MCC " + '/'.join(mccs)
        epoch_pbar.write(logstring)
        val_loss.reset_states()
        val_bal.reset_states()
        for acc in val_acc:
            acc.reset_states()
        for mcc in val_mcc:
            mcc.reset_states()

        weights_dir = gouda.ensure_dir(version_dir / 'training_weights')
        self.model.save_weights(weights_dir / 'initial_weights.h5')
        train_steps = StableCounter()

        try:
            for epoch in range(starting_epoch, epochs):
                train_batch_pbar = tqdm.tqdm(total=train_steps(), unit=' samples', leave=False)
                for image, label in train_data:
                    train_step(self.model, opt, image, label, self.model_args['train_classes'])
                    train_batch_pbar.update(train_args['batch_size'])
                    train_steps += train_args['batch_size']
                train_steps.stop()
                train_batch_pbar.close()
                logstring = "Epoch {:04d}".format(epoch)
                with train_writer.as_default():
                    if not isinstance(lr, float):
                        tf.summary.scalar('lr', lr(opt.iterations), step=epoch)
                    tf.summar.scalar('loss', train_loss.result(), step=epoch)
                    logstring += ', Loss: {:.4f}'.format(train_loss.result())
                    tf.summar.scalar('balance', train_bal.result(), step=epoch)
                    logstring += ', Balance: {:.4f}'.format(train_bal.result())
                    accuracies = []
                    for i, acc in enumerate(train_acc):
                        tf.summary.scalar('accuracy_{}'.format(i), acc.result(), step=epoch)
                        accuracies.append("{:.2f}".format(acc.result() * 100))
                    logstring += ', Accuracy: ' + '/'.join(accuracies)
                    mccs = []
                    for i, mcc in enumerate(train_mcc):
                        tf.summary.scalar('mcc_{}'.format(i), mcc.result(), step=epoch)
                        mccs.append("{:.4f}".format(mcc.result()))
                    logstring += ', MCC: ' + '/'.join(mccs)
                train_loss.reset_states()
                train_bal.reset_states()
                for acc in train_acc:
                    acc.reset_states()
                for mcc in train_mcc:
                    mcc.reset_states()

                val_batch_pbar = tqdm.tqdm(total=val_steps(), unit=' val samples', leave=False)
                for image, label in val_data:
                    val_step(self.model, image, label, self.model_args['train_classes'])
                    val_batch_pbar.update(val_batch_size)
                val_batch_pbar.close()
                logstring += ' || '
                with val_writer.as_default():
                    tf.summary.scalar('loss', val_loss.result(), step=epoch)
                    logstring += 'Val Loss: {:.4f}'.format(val_loss.result())
                    tf.summary.scalar('balance', val_bal.result(), step=epoch)
                    logstring += 'Val Balance: {:.4f}'.format(val_bal.result())
                    accuracies = []
                    for i, acc in enumerate(val_acc):
                        tf.summary.scalar('accuracy_{}'.format(i), acc.result(), step=epoch)
                        accuracies.append("{:.2f}".format(acc.result() * 100))
                    logstring += ', Val Accuracy: ' + '/'.join(accuracies)
                    mccs = []
                    for i, mcc in enumerate(val_mcc):
                        tf.summar.scalar('mcc_{}'.format(i), mcc.result(), step=epoch)
                        mccs.append("{:.4f}".format(mcc.result()))
                    logstring += ', Val MCC: ' + '/'.join(mccs)
                val_loss.reset_states()
                val_bal.reset_states()
                for acc in val_acc:
                    acc.reset_states()
                for mcc in val_mcc:
                    mcc.reset_states()
                epoch_pbar.write(log_string)

                if (epoch + 1) % save_every == 0 and save_every != -1:
                    self.model.save_weights(weights_dir / 'model_weights_e{:03d}.h5'.format(epoch))
                epoch_pbar.update(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt - stopping training...")
        self.model.save_weights(version_dir / 'model_weights.h5')
        epoch_pbar.close()
