import os

import tensorflow as tf
import warnings

from .core_model import CoreModel
from .model_arch import get_model_func


class MultiModel(CoreModel):
    def __init__(self, *args, **kwargs):
        """A SubClass of CoreModel to allow multiple models trained in parallel"""
        super(MultiModel, self).__init__(*args, **kwargs)
        self.models = {}

    @property
    def model(self):
        return self.models

    @model.setter
    def model(self, value):
        if value is not None:
            warnings.warn("Cannot set parameter 'model' in MultiModel object")

    def add_model(self, model_func):
        """Add a model function to the MultiModel object

        NOTE
        ----
        This will clear any currently compiled models
        """
        self.clear()
        self.model_args['model_func'].append(model_func)

    def compile_model(self, model_func=[], checking=False, **kwargs):
        """Compile the models for the given functions.

        Parameters
        ----------
        model_func: list of str | func
            Either the strings to lookup, the model functions, or an empty list if you want to use the functions from self.model_args (the default is None)
        checking: bool
            If true, will only compile models if there are no compiled models already

        NOTE
        ----
        When compiling models, if any keys in self.model_args have the form 'key_[0-9]', then that key will only be passed to the ith model in the model_func list. If a specific and generic key both exist, the specific key will take precedence.
        """
        if checking:
            if len(self.models) != 0:
                return
        self.clear()
        if self.model_args['model_func'] is None and len(model_func) == 0:
            raise ValueError("No selected model functions")

        for key in kwargs:
            self.model_args[key] = kwargs[key]
        if len(model_func) == 0:
            model_func = self.model_args['model_func']

        for i, func in enumerate(model_func):
            if isinstance(func, str):
                func = get_model_func(func)
            specific_args = {}
            for key in self.model_args:
                if key[-1].isnumeric():
                    sub_key, index = key.rsplit('_', 1)
                    if int(index) == i:
                        specific_args[sub_key] = self.model_args[key]
                elif key not in self.model_args:
                    specific_args[key] = self.model_args[key]

            model = func(**specific_args)
            self.models[model.name] = model

    def load_weights(self, name, group_name=None, model_name=None, version=None, path=None):
        """Load weights from an existing trained model.

        Paramters
        ---------
        name: str
            The name of the specific model to load
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
        if name not in self.models:
            self.compile_model()
        if path is None and version is None:
            raise ValueError('Either version or path must be specified to load weights')
        if path is None:
            if group_name is None or model_name is None:
                path = self.model_dir / version / 'model_weights'
            else:
                path = self.results_dir / group_name / model_name / version / 'model_weights'
            path = path + '_' + name
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
        if isinstance(path, os.PathLike):
            path = str(path)
        self.models[name].load_weights(path)

    def save_weights(self, path):
        if isinstance(path, os.PathLike):
            path = str(path)
        for model_key in self.models:
            path_items = path.rsplit('.', 1)
            out_path = path_items[0] + '_' + model_key
            if len(path_items) == 2:
                out_path += '.' + path_items[1]
            self.models[model_key].save_weights(out_path)

    def export_model(self, path):
        if isinstance(path, os.PathLike):
            path = str(path)
        for model_key in self.models:
            path_items = path.rsplit('.', 1)
            out_path = path_items[0] + '_' + model_key
            if len(path_items) == 2:
                out_path += '.' + path_items[1]
            tf.saved_model.save(self.models[model_key], out_path)

    def summary(self, name, **kwargs):
        return self.models[name].summary(**kwargs)

    def num_parameters(self, name=None):
        """Get the total number of parameters for the models.

        Parameters
        ----------
        name: str
            An optional argument to specify a single model (the default is None)
        """
        total = 0
        if name is None:
            for key in self.models:
                for var in self.models[key].variables:
                    total += tf.size(var)
        else:
            for var in self.models[name].variables:
                total += tf.size(var)
        return total

    def num_trainable_paramters(self, name=None):
        """Get the total number of trainable parameters for the models.

        Parameters
        ----------
        name: str
            An optional argument to specify a single model (the default is None)
        """
        total = 0
        if name is None:
            for key in self.models:
                for var in self.models[key].trainable_variables:
                    total += tf.size(var)
        else:
            for var in self.models[name].trainable_variables:
                total += tf.size(var)
        return total

    def plot_model(self, name):
        if name not in self.models:
            raise ValueError("Compile the model before plotting")
        tf.keras.utils.plot_model(self.models[name], to_file=self.model_dir('model.png').abspath, show_shapes=True)

    def _setup_optimizer(self, train_args):
        copy_args = train_args.copy()  # TODO: make this less hacky
        return {
            'auto': super()._setup_optimizer(copy_args),
            'seg': super()._setup_optimizer(train_args)
        }

    def __call__(self, x, name, *args, training=False, **kwargs):
        return self.models[name](x, *args, training=training, **kwargs)
