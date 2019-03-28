import six
import tensorflow as tf

from . import backend as K
from .legacy import interfaces

from .utils.generic_utils import deserialize_keras_object



def deserialize(config, custom_objects=None):
    """Inverse of the `serialize` function.

    # Arguments
        config: Optimizer configuration dictionary.
        custom_objects: Optional dictionary mapping
            names (strings) to custom objects
            (classes and functions)
            to be considered during deserialization.

    # Returns
        A Keras Optimizer instance.
    """
    all_classes = {
        'adadelta': Adadelta,
        'tfoptimizer': TFOptimizer,
    }
    # Make deserialization case-insensitive for built-in optimizers.
    if config['class_name'].lower() in all_classes:
        config['class_name'] = config['class_name'].lower()
    return deserialize_keras_object(config,
                                    module_objects=all_classes,
                                    custom_objects=custom_objects,
                                    printable_module_name='optimizer')


class Optimizer(object):
    """Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.

    All Keras optimizers support the following keyword arguments:

        clipnorm: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        clipvalue: float >= 0. Gradients will be clipped
            when their absolute value exceeds this value.
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if None in grads:
            raise ValueError('An operation has `None` for gradient. '
                             'Please make sure that all of your ops have a '
                             'gradient defined (i.e. are differentiable). '
                             'Common ops without gradient: '
                             'K.argmax, K.round, K.eval.')
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def set_weights(self, weights):
        """Sets the weights of the optimizer, from Numpy arrays.

        Should only be called after computing the gradients
        (otherwise the optimizer has no weights).

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the optimizer (i.e. it should match the
                output of `get_weights`).

        # Raises
            ValueError: in case of incompatible weight shapes.
        """
        params = self.weights
        if len(params) != len(weights):
            raise ValueError('Length of the specified weight list (' +
                             str(len(weights)) +
                             ') does not match the number of weights ' +
                             'of the optimizer (' + str(len(params)) + ')')
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Optimizer weight shape ' +
                                 str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        K.batch_set_value(weight_value_tuples)

    def get_weights(self):
        """Returns the current value of the weights of the optimizer.

        # Returns
            A list of numpy arrays.
        """
        return K.batch_get_value(self.weights)

    def get_config(self):
        config = {}
        if hasattr(self, 'clipnorm'):
            config['clipnorm'] = self.clipnorm
        if hasattr(self, 'clipvalue'):
            config['clipvalue'] = self.clipvalue
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Adadelta(Optimizer):
    """Adadelta optimizer.

    Adadelta is a more robust extension of Adagrad
    that adapts learning rates based on a moving window of gradient updates,
    instead of accumulating all past gradients. This way, Adadelta continues
    learning even when many updates have been done. Compared to Adagrad, in the
    original version of Adadelta you don't have to set an initial learning
    rate. In this version, initial learning rate and decay factor can
    be set, as in most other Keras optimizers.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Initial learning rate, defaults to 1.
            It is recommended to leave it at the default value.
        rho: float >= 0. Adadelta decay factor, corresponding to fraction of
            gradient to keep at each time step.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Initial learning rate decay.

    # References
        - [Adadelta - an adaptive learning rate method]
          (https://arxiv.org/abs/1212.5701)
    """

    def __init__(self, lr=1.0, rho=0.95, epsilon=None, decay=0.,
                 **kwargs):
        super(Adadelta, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.rho = rho
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        delta_accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators + delta_accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)
            new_p = p - lr * update

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
            self.updates.append(K.update(d_a, new_d_a))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': self.rho,
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(Adadelta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TFOptimizer(Optimizer):
    """Wrapper class for native TensorFlow optimizers.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.optimizer.compute_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        opt_update = self.optimizer.apply_gradients(
            grads, global_step=self.iterations)
        self.updates.append(opt_update)
        return self.updates

    @property
    def weights(self):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    def from_config(self, config):
        raise NotImplementedError


def get(identifier):
    """Retrieves a Keras Optimizer instance.

    # Arguments
        identifier: Optimizer identifier, one of
            - String: name of an optimizer
            - Dictionary: configuration dictionary.
            - Keras Optimizer instance (it will be returned unchanged).
            - TensorFlow Optimizer instance
                (it will be wrapped as a Keras Optimizer).

    # Returns
        A Keras Optimizer instance.

    # Raises
        ValueError: If `identifier` cannot be interpreted.
    """
    if K.backend() == 'tensorflow':
        # Wrap TF optimizer instances
        if isinstance(identifier, tf.train.Optimizer):
            return TFOptimizer(identifier)
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    if isinstance(identifier, Optimizer):
        return identifier
    else:
        raise ValueError('Could not interpret optimizer identifier: ' +
                         str(identifier))
