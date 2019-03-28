import warnings

import six

from . import backend as K
from .engine import Layer
from .utils.generic_utils import deserialize_keras_object


def softmax(x, axis=-1):
    """Softmax activation function.

    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the softmax normalization is applied.

    # Returns
        Tensor, output of softmax transformation.

    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 1:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
    elif ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)

def linear(x):
    """Linear (i.e. identity) activation function.
    """
    return x


def relu(x, alpha=0., max_value=None, threshold=0.):
    """Rectified Linear Unit.

    With default values, it returns element-wise `max(x, 0)`.

    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.

    # Arguments
        x: Input tensor.
        alpha: float. Slope of the negative part. Defaults to zero.
        max_value: float. Saturation threshold.
        threshold: float. Threshold value for thresholded activation.

    # Returns
        A tensor.
    """
    return K.relu(x, alpha=alpha, max_value=max_value, threshold=threshold)



def deserialize(name, custom_objects=None):
    return deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='activation function')


def get(identifier):
    """Get the `identifier` activation function.

    # Arguments
        identifier: None or str, name of the function.

    # Returns
        The activation function, `linear` if `identifier` is None.

    # Raises
        ValueError if unknown identifier
    """
    if identifier is None:
        return linear
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, Layer):
            warnings.warn(
                'Do not pass a layer instance (such as {identifier}) as the '
                'activation argument of another layer. Instead, advanced '
                'activation layers should be used just like any other '
                'layer in a model.'.format(
                    identifier=identifier.__class__.__name__))
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'activation function identifier:', identifier)
