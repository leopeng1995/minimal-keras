import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.core.protobuf import config_pb2

from .common import epsilon
from .common import floatx
from .common import normalize_data_format

from ..utils.generic_utils import has_arg

py_all = all
py_any = any

# INTERNAL UTILS

# This is the default internal TF session used by Keras.
# It can be set manually via `set_session(sess)`.
_SESSION = None

name_scope = tf.name_scope

# This dictionary holds a mapping {graph: learning_phase}.
# A learning phase is a bool tensor used to run Keras models in
# either train mode (learning_phase == 1) or test mode (learning_phase == 0).
_GRAPH_LEARNING_PHASES = {}

# This dictionary holds a mapping {graph: UID_DICT}.
# each UID_DICT is a dictionary mapping name prefixes to a current index,
# used for generating graph-specific string UIDs
# for various names (e.g. layer names).
_GRAPH_UID_DICTS = {}


# This boolean flag can be set to True to leave variable initialization
# up to the user.
# Change its value via `manual_variable_initialization(value)`.
_MANUAL_VAR_INIT = False


def get_uid(prefix=''):
    """Get the uid for the default graph.

    # Arguments
        prefix: An optional prefix of the graph.

    # Returns
        A unique identifier for the graph.
    """
    global _GRAPH_UID_DICTS
    graph = tf.get_default_graph()
    if graph not in _GRAPH_UID_DICTS:
        _GRAPH_UID_DICTS[graph] = defaultdict(int)
    _GRAPH_UID_DICTS[graph][prefix] += 1
    return _GRAPH_UID_DICTS[graph][prefix]


# VARIABLE MANIPULATION

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


def constant(value, dtype=None, shape=None, name=None):
    """Creates a constant tensor.

    # Arguments
        value: A constant value (or list)
        dtype: The type of the elements of the resulting tensor.
        shape: Optional dimensions of resulting tensor.
        name: Optional name for the tensor.

    # Returns
        A Constant Tensor.
    """
    if dtype is None:
        dtype = floatx()
    return tf.constant(value, dtype=dtype, shape=shape, name=name)


def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value.

    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: tensor (`int` or `bool`).
        then_expression: either a tensor, or a callable that returns a tensor.
        else_expression: either a tensor, or a callable that returns a tensor.

    # Returns
        The selected tensor.

    # Raises
        ValueError: If rank of `condition` is greater than rank of expressions.
    """
    if condition.dtype != tf.bool:
        condition = tf.cast(condition, 'bool')
    cond_ndim = ndim(condition)
    if not cond_ndim:
        if not callable(then_expression):
            def then_expression_fn():
                return then_expression
        else:
            then_expression_fn = then_expression
        if not callable(else_expression):
            def else_expression_fn():
                return else_expression
        else:
            else_expression_fn = else_expression
        x = tf.cond(condition,
                    then_expression_fn,
                    else_expression_fn)
    else:
        # tf.where needs its condition tensor
        # to be the same shape as its two
        # result tensors
        if callable(then_expression):
            then_expression = then_expression()
        if callable(else_expression):
            else_expression = else_expression()
        expr_ndim = ndim(then_expression)
        if cond_ndim > expr_ndim:
            raise ValueError('Rank of `condition` should be less than or'
                             ' equal to rank of `then_expression` and '
                             '`else_expression`. ndim(condition)=' +
                             str(cond_ndim) + ', ndim(then_expression)'
                             '=' + str(expr_ndim))
        if cond_ndim > 1:
            ndim_diff = expr_ndim - cond_ndim
            cond_shape = tf.concat([tf.shape(condition), [1] * ndim_diff], axis=0)
            condition = tf.reshape(condition, cond_shape)
            expr_shape = tf.shape(then_expression)
            shape_diff = expr_shape - cond_shape
            tile_shape = tf.where(shape_diff > 0, expr_shape, tf.ones_like(expr_shape))
            condition = tf.tile(condition, tile_shape)
        x = tf.where(condition, then_expression, else_expression)
    return x


def dropout(x, level, noise_shape=None, seed=None):
    """Sets entries in `x` to zero at random, while scaling the entire tensor.

    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
        noise_shape: shape for randomly generated keep/drop flags,
            must be broadcastable to the shape of `x`
        seed: random seed to ensure determinism.

    # Returns
        A tensor.
    """
    retain_prob = 1. - level
    if seed is None:
        seed = np.random.randint(10e6)
    # the dummy 1. works around a TF bug
    # (float32_ref vs. float32 incompatibility)
    return tf.nn.dropout(x * 1., retain_prob, noise_shape, seed=seed)


# NN OPERATIONS

def relu(x, alpha=0., max_value=None, threshold=0.):
    """Rectified linear unit.

    With default values, it returns element-wise `max(x, 0)`.

    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.

    # Arguments
        x: A tensor or variable.
        alpha: A scalar, slope of negative section (default=`0.`).
        max_value: float. Saturation threshold.
        threshold: float. Threshold value for thresholded activation.

    # Returns
        A tensor.
    """

    if alpha != 0.:
        if max_value is None and threshold == 0.:
            return tf.nn.leaky_relu(x, alpha=alpha)

        if threshold != 0.:
            negative_part = tf.nn.relu(-x + threshold)
        else:
            negative_part = tf.nn.relu(-x)

    clip_max = max_value is not None

    if threshold != 0:
        # computes x for x > threshold else 0
        x = x * tf.cast(tf.greater(x, threshold), floatx())
    elif max_value == 6:
        # if no threshold, then can use nn.relu6 native TF op for performance
        x = tf.nn.relu6(x)
        clip_max = False
    else:
        x = tf.nn.relu(x)

    if clip_max:
        max_value = _to_tensor(max_value, x.dtype.base_dtype)
        zero = _to_tensor(0., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, max_value)

    if alpha != 0:
        alpha = _to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part
    return x


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def prod(x, axis=None, keepdims=False):
    """Multiplies the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the product. If `None` (default), computes
            the product over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the product of elements of `x`.
    """
    return tf.reduce_prod(x, axis, keepdims)


def shape(x):
    """Returns the symbolic shape of a tensor or variable.

    # Arguments
        x: A tensor or variable.

    # Returns
        A symbolic shape (which is itself a tensor).

    # Examples
    ```python
        # TensorFlow example
        >>> from keras import backend as K
        >>> tf_session = K.get_session()
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> inputs = keras.backend.placeholder(shape=(2, 4, 5))
        >>> K.shape(kvar)
        <tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
        >>> K.shape(inputs)
        <tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
        # To get integer shape (Instead, you can use K.int_shape(x))
        >>> K.shape(kvar).eval(session=tf_session)
        array([2, 2], dtype=int32)
        >>> K.shape(inputs).eval(session=tf_session)
        array([2, 4, 5], dtype=int32)
    ```
    """
    return tf.shape(x)


def batch_flatten(x):
    """Turn a nD tensor into a 2D tensor with same 0th dimension.

    In other words, it flattens each data samples of a batch.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    x = tf.reshape(x, tf.stack([-1, prod(shape(x)[1:])]))
    return x


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiates a placeholder tensor and returns it.

    # Arguments
        shape: Shape of the placeholder
            (integer tuple, may include `None` entries).
        ndim: Number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: Placeholder type.
        sparse: Boolean, whether the placeholder should have a sparse type.
        name: Optional name string for the placeholder.

    # Returns
        Tensor instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input_ph = K.placeholder(shape=(2, 4, 5))
        >>> input_ph._keras_shape
        (2, 4, 5)
        >>> input_ph
        <tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
    ```
    """
    if dtype is None:
        dtype = floatx()
    if not shape:
        if ndim:
            shape = tuple([None for _ in range(ndim)])
    if sparse:
        x = tf.sparse_placeholder(dtype, shape=shape, name=name)
    else:
        x = tf.placeholder(dtype, shape=shape, name=name)
    x._keras_shape = shape
    x._uses_learning_phase = False
    return x


def is_tensor(x):
    return isinstance(x, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(x)


def is_keras_tensor(x):
    """Returns whether `x` is a Keras tensor.

    A "Keras tensor" is a tensor that was returned by a Keras layer,
    (`Layer` class) or by `Input`.

    # Arguments
        x: A candidate tensor.

    # Returns
        A boolean: Whether the argument is a Keras tensor.

    # Raises
        ValueError: In case `x` is not a symbolic tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> from keras.layers import Input, Dense
        >>> np_var = numpy.array([1, 2])
        >>> K.is_keras_tensor(np_var) # A numpy array is not a symbolic tensor.
        ValueError
        >>> k_var = tf.placeholder('float32', shape=(1,1))
        >>> K.is_keras_tensor(k_var) # A variable indirectly created outside of keras is not a Keras tensor.
        False
        >>> keras_var = K.variable(np_var)
        >>> K.is_keras_tensor(keras_var)  # A variable created with the keras backend is not a Keras tensor.
        False
        >>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
        >>> K.is_keras_tensor(keras_placeholder)  # A placeholder is not a Keras tensor.
        False
        >>> keras_input = Input([10])
        >>> K.is_keras_tensor(keras_input) # An Input is a Keras tensor.
        True
        >>> keras_layer_output = Dense(10)(keras_input)
        >>> K.is_keras_tensor(keras_layer_output) # Any Keras layer output is a Keras tensor.
        True
    ```
    """
    if not is_tensor(x):
        raise ValueError('Unexpectedly found an instance of type `' +
                         str(type(x)) + '`. '
                                        'Expected a symbolic tensor instance.')
    return hasattr(x, '_keras_history')


def int_shape(x):
    """Returns the shape of tensor or variable as a tuple of int or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> K.int_shape(inputs)
        (2, 4, 5)
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.int_shape(kvar)
        (2, 2)
    ```
    """
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        return tuple(x.get_shape().as_list())
    except ValueError:
        return None


def variable(value, dtype=None, name=None, constraint=None):
    """Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.
        constraint: Optional projection function to be
            applied to the variable after an optimizer update.

    # Returns
        A variable instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
        >>> K.dtype(kvar)
        'float64'
        >>> print(kvar)
        example_var
        >>> K.eval(kvar)
        array([[ 1.,  2.],
               [ 3.,  4.]])
    ```
    """
    if dtype is None:
        dtype = floatx()
    if hasattr(value, 'tocoo'):
        sparse_coo = value.tocoo()
        indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                  np.expand_dims(sparse_coo.col, 1)), 1)
        v = tf.SparseTensor(indices=indices,
                            values=sparse_coo.data,
                            dense_shape=sparse_coo.shape)
        v._keras_shape = sparse_coo.shape
        v._uses_learning_phase = False
        return v
    v = tf.Variable(value, dtype=tf.as_dtype(dtype), name=name)
    if isinstance(value, np.ndarray):
        v._keras_shape = value.shape
    elif hasattr(value, 'get_shape'):
        v._keras_shape = int_shape(value)
    v._uses_learning_phase = False
    # TODO: move to Variable constructor when supported in public release.
    try:
        v.constraint = constraint
    except AttributeError:
        v._constraint = constraint
    return v


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Returns a tensor with uniform distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        minval: A float, lower boundary of the uniform distribution
            to draw samples.
        maxval: A float, upper boundary of the uniform distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random_uniform(shape, minval=minval, maxval=maxval,
                             dtype=dtype, seed=seed)


# DEVICE MANIPULATION AND PROBING

class _TfDeviceCaptureOp(object):
    """Class for capturing the TF device scope."""

    def __init__(self):
        self.device = None

    def _set_device(self, device):
        """This method captures TF's explicit device scope setting."""
        self.device = device


def _get_current_tf_device():
    """Return explicit device of current context, otherwise returns `None`.

    # Returns
        If the current device scope is explicitly set, it returns a string with
        the device (`CPU` or `GPU`). If the scope is not explicitly set, it will
        return `None`.
    """
    g = tf.get_default_graph()
    op = _TfDeviceCaptureOp()
    g._apply_device_functions(op)
    return op.device


def _is_current_explicit_device(device_type):
    """Check if the current device is explicitly set on the device type specified.

    # Arguments
        device_type: A string containing `GPU` or `CPU` (case-insensitive).

    # Returns
        A boolean indicating if the current device scope is explicitly set on the device type.

    # Raises
        ValueError: If the `device_type` string indicates an unsupported device.
    """
    device_type = device_type.upper()
    if device_type not in ['CPU', 'GPU']:
        raise ValueError('`device_type` should be either "CPU" or "GPU".')
    device = _get_current_tf_device()
    return (device is not None and device.device_type == device_type.upper())


def get_session():
    """Returns the TF session to be used by the backend.

    If a default TensorFlow session is available, we will return it.

    Else, we will return the global Keras session.

    If no global Keras session exists at this point:
    we will create a new global session.

    Note that you can manually set the global session
    via `K.set_session(sess)`.

    # Returns
        A TensorFlow session.
    """
    global _SESSION

    default_session = tf.get_default_session()

    if default_session is not None:
        session = default_session
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                num_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                        allow_soft_placement=True)
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if not _MANUAL_VAR_INIT:
        with session.graph.as_default():
            variables = tf.global_variables()
            candidate_vars = []
            for v in variables:
                if not getattr(v, '_keras_initialized', False):
                    candidate_vars.append(v)
            if candidate_vars:
                # This step is expensive, so we only run it on variables
                # not already marked as initialized.
                is_initialized = session.run(
                    [tf.is_variable_initialized(v) for v in candidate_vars])
                uninitialized_vars = []
                for flag, v in zip(is_initialized, candidate_vars):
                    if not flag:
                        uninitialized_vars.append(v)
                    v._keras_initialized = True
                if uninitialized_vars:
                    session.run(tf.variables_initializer(uninitialized_vars))
    # hack for list_devices() function.
    # list_devices() function is not available under tensorflow r1.3.
    if not hasattr(session, 'list_devices'):
        session.list_devices = lambda: device_lib.list_local_devices()
    return session


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    global _LOCAL_DEVICES
    if _LOCAL_DEVICES is None:
        _LOCAL_DEVICES = get_session().list_devices()
    return [x.name for x in _LOCAL_DEVICES if x.device_type == 'GPU']


def _has_nchw_support():
    """Check whether the current scope supports NCHW ops.

    TensorFlow does not support NCHW on CPU. Therefore we check if we are not explicitly put on
    CPU, and have GPUs available. In this case there will be soft-placing on the GPU device.

    # Returns
        bool: if the current scope device placement would support nchw
    """
    explicitly_on_cpu = _is_current_explicit_device('CPU')
    gpus_available = len(_get_available_gpus()) > 0
    return (not explicitly_on_cpu and gpus_available)


def dtype(x):
    """Returns the dtype of a Keras tensor or variable, as a string.

    # Arguments
        x: Tensor or variable.

    # Returns
        String, dtype of `x`.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> K.dtype(K.placeholder(shape=(2,4,5)))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
        'float64'
        # Keras variable
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
        >>> K.dtype(kvar)
        'float32_ref'
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.dtype(kvar)
        'float32_ref'
    ```
    """
    return x.dtype.base_dtype.name


def _preprocess_conv2d_input(x, data_format, force_transpose=False):
    """Transpose and cast the input before the conv2d.

    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
        force_transpose: boolean, whether force to transpose input from NCHW to NHWC
                        if the `data_format` is `"channels_first"`.

    # Returns
        A tensor.
    """
    # tensorflow doesn't support float64 for conv layer before 1.8.0
    if (dtype(x) == 'float64' and
                StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.8.0')):
        x = tf.cast(x, 'float32')
    tf_data_format = 'NHWC'
    if data_format == 'channels_first':
        if not _has_nchw_support() or force_transpose:
            x = tf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        else:
            tf_data_format = 'NCHW'
    return x, tf_data_format


def _preprocess_padding(padding):
    """Convert keras' padding to tensorflow's padding.

    # Arguments
        padding: string, `"same"` or `"valid"`.

    # Returns
        a string, `"SAME"` or `"VALID"`.

    # Raises
        ValueError: if `padding` is invalid.
    """
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid padding: ' + str(padding))
    return padding


def conv2d(x, kernel, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """2D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.

    # Returns
        A tensor, result of 2D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)

    padding = _preprocess_padding(padding)
    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format)

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


def bias_add(x, bias, data_format=None):
    """Adds a bias vector to a tensor.

    # Arguments
        x: Tensor or variable.
        bias: Bias tensor to add.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        Output tensor.

    # Raises
        ValueError: In one of the two cases below:
                    1. invalid `data_format` argument.
                    2. invalid bias shape.
                       the bias should be either a vector or
                       a tensor with ndim(x) - 1 dimension
    """
    data_format = normalize_data_format(data_format)
    bias_shape = int_shape(bias)
    if len(bias_shape) != 1 and len(bias_shape) != ndim(x) - 1:
        raise ValueError('Unexpected bias dimensions %d, expect to be 1 or %d dimensions'
                         % (len(bias_shape), ndim(x)))
    if ndim(x) == 5:
        if len(bias_shape) == 1:
            new_shape = (1, 1, 1, 1, bias_shape[0])
        else:
            new_shape = (1,) + bias_shape
        new_shape = transpose_shape(new_shape, data_format, spatial_axes=(1, 2, 3))
        x += reshape(bias, new_shape)
    elif ndim(x) == 4:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                if _has_nchw_support():
                    x = tf.nn.bias_add(x, bias,
                                       data_format='NCHW')
                else:
                    x += reshape(bias, (1, bias_shape[0], 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x = tf.nn.bias_add(x, bias,
                                   data_format='NHWC')
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif ndim(x) == 3:
        if len(bias_shape) == 1:
            new_shape = (1, 1, bias_shape[0])
        else:
            new_shape = (1,) + bias_shape
        new_shape = transpose_shape(new_shape, data_format, spatial_axes=(1,))
        x += reshape(bias, new_shape)
    else:
        x = tf.nn.bias_add(x, bias)
    return x


def pool2d(x, pool_size, strides=(1, 1),
           padding='valid', data_format=None,
           pool_mode='max'):
    """2D Pooling.

    # Arguments
        x: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        pool_mode: string, `"max"` or `"avg"`.

    # Returns
        A tensor, result of 2D pooling.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """
    data_format = normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)
    else:
        strides = (1, 1) + strides
        pool_size = (1, 1) + pool_size

    if pool_mode == 'max':
        x = tf.nn.max_pool(x, pool_size, strides,
                           padding=padding,
                           data_format=tf_data_format)
    elif pool_mode == 'avg':
        x = tf.nn.avg_pool(x, pool_size, strides,
                           padding=padding,
                           data_format=tf_data_format)
    else:
        raise ValueError('Invalid pool_mode: ' + str(pool_mode))

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


def learning_phase():
    """Returns the learning phase flag.

    The learning phase flag is a bool tensor (0 = test, 1 = train)
    to be passed as input to any Keras function
    that uses a different behavior at train time and test time.

    # Returns
        Learning phase (scalar integer tensor or Python integer).
    """
    graph = tf.get_default_graph()
    if graph not in _GRAPH_LEARNING_PHASES:
        phase = tf.placeholder_with_default(False,
                                            shape=(),
                                            name='keras_learning_phase')
        _GRAPH_LEARNING_PHASES[graph] = phase
    return _GRAPH_LEARNING_PHASES[graph]


def in_train_phase(x, alt, training=None):
    """Selects `x` in train phase, and `alt` otherwise.

    Note that `alt` should have the *same shape* as `x`.

    # Arguments
        x: What to return in train phase
            (tensor or callable that returns a tensor).
        alt: What to return otherwise
            (tensor or callable that returns a tensor).
        training: Optional scalar tensor
            (or Python boolean, or Python integer)
            specifying the learning phase.

    # Returns
        Either `x` or `alt` based on the `training` flag.
        the `training` flag defaults to `K.learning_phase()`.
    """
    if training is None:
        training = learning_phase()
        uses_learning_phase = True
    else:
        uses_learning_phase = False

    if training is 1 or training is True:
        if callable(x):
            return x()
        else:
            return x

    elif training is 0 or training is False:
        if callable(alt):
            return alt()
        else:
            return alt

    # else: assume learning phase is a placeholder tensor.
    x = switch(training, x, alt)
    if uses_learning_phase:
        x._uses_learning_phase = True
    return x


def softmax(x, axis=-1):
    """Softmax of a tensor.

    # Arguments
        x: A tensor or variable.
        axis: The dimension softmax would be performed on.
            The default is -1 which indicates the last dimension.

    # Returns
        A tensor.
    """
    return tf.nn.softmax(x, axis=axis)


def is_sparse(tensor):
    """Returns whether a tensor is a sparse tensor.

    # Arguments
        tensor: A tensor instance.

    # Returns
        A boolean.

    # Example
    ```python
        >>> from keras import backend as K
        >>> a = K.placeholder((2, 2), sparse=False)
        >>> print(K.is_sparse(a))
        False
        >>> b = K.placeholder((2, 2), sparse=True)
        >>> print(K.is_sparse(b))
        True
    ```
    """
    return isinstance(tensor, tf.SparseTensor)


def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.

    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.

    # Examples
    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(2, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
    ```

    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(32, 28, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
    ```

    ```python
        # Theano-like behavior example
        >>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
        >>> y = K.ones((4, 3, 5))
        >>> xy = K.dot(x, y)
        >>> K.int_shape(xy)
        (2, 4, 5)
    ```
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(int_shape(x), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(int_shape(y), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if is_sparse(x):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out


def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer or list of integers in [-rank(x), rank(x)),
            the axes to compute the mean. If `None` (default), computes
            the mean over all dimensions.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keepdims` is `True`,
            the reduced dimensions are retained with length 1.

    # Returns
        A tensor with the mean of elements of `x`.
    """
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    return tf.reduce_mean(x, axis, keepdims)


def cast(x, dtype):
    """Casts a tensor to a different dtype and returns it.

    You can cast a Keras variable but it still returns a Keras tensor.

    # Arguments
        x: Keras tensor (or variable).
        dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

    # Returns
        Keras tensor with dtype `dtype`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder((2, 3), dtype='float32')
        >>> input
        <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
        # It doesn't work in-place as below.
        >>> K.cast(input, dtype='float16')
        <tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
        >>> input
        <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
        # you need to assign it.
        >>> input = K.cast(input, dtype='float16')
        >>> input
        <tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
    ```
    """
    return tf.cast(x, dtype)


def equal(x, y):
    """Element-wise equality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return tf.equal(x, y)


def not_equal(x, y):
    """Element-wise inequality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return tf.not_equal(x, y)


def argmax(x, axis=-1):
    """Returns the index of the maximum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    """
    return tf.argmax(x, axis)


def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)


def flatten(x):
    """Flatten a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor, reshaped into 1-D
    """
    return tf.reshape(x, [-1])


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy with integer targets.

    # Arguments
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # If the channels are not in the last axis, move them to be there:
    if axis != -1 and axis != output_dimensions[-1]:
        permutation = output_dimensions[:axis] + output_dimensions[axis + 1:]
        permutation += [axis]
        output = tf.transpose(output, perm=permutation)

    # Note: tf.nn.sparse_softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output)

    output_shape = output.get_shape()
    targets = cast(flatten(target), 'int64')
    logits = tf.reshape(output, [-1, int(output_shape[-1])])
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets,
        logits=logits)
    if len(output_shape) >= 3:
        # if our output includes timestep dimension
        # or spatial dimensions we need to reshape
        return tf.reshape(res, tf.shape(output)[:-1])
    else:
        return res


def square(x):
    """Element-wise square.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return tf.square(x)


def gradients(loss, variables):
    """Returns the gradients of `loss` w.r.t. `variables`.

    # Arguments
        loss: Scalar tensor to minimize.
        variables: List of variables.

    # Returns
        A gradients tensor.
    """
    return tf.gradients(loss, variables, colocate_gradients_with_ops=True)


def zeros(shape, dtype=None, name=None):
    """Instantiates an all-zeros variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable
        dtype: String, data type of returned Keras variable
        name: String, name of returned Keras variable

    # Returns
        A variable (including Keras metadata), filled with `0.0`.
        Note that if `shape` was symbolic, we cannot return a variable,
        and will return a dynamically-shaped tensor instead.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.zeros((3,4))
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    tf_dtype = tf.as_dtype(dtype)
    v = tf.zeros(shape=shape, dtype=tf_dtype, name=name)
    if py_all(v.get_shape().as_list()):
        return variable(v, dtype=dtype, name=name)
    return v


# UPDATES OPS


def update(x, new_x):
    """Update the value of `x` to `new_x`.

    # Arguments
        x: A `Variable`.
        new_x: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return tf.assign(x, new_x)


def update_add(x, increment):
    """Update the value of `x` by adding `increment`.

    # Arguments
        x: A `Variable`.
        increment: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return tf.assign_add(x, increment)


def sqrt(x):
    """Element-wise square root.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    zero = _to_tensor(0., x.dtype.base_dtype)
    inf = _to_tensor(np.inf, x.dtype.base_dtype)
    x = tf.clip_by_value(x, zero, inf)
    return tf.sqrt(x)


# GRAPH MANIPULATION

class Function(object):
    """Runs a computation graph.

    It's possible to pass arguments to `tf.Session.run()` via `session_kwargs`.
    In particular additional operations via `fetches` argument and additional
    tensor substitutions via `feed_dict` arguments. Note that given
    substitutions are merged with substitutions from `inputs`. Even though
    `feed_dict` is passed once in the constructor (called in `model.compile()`)
    we can modify the values in the dictionary. Through this feed_dict we can
    provide additional substitutions besides Keras inputs.

    # Arguments
        inputs: Feed placeholders to the computation graph.
        outputs: Output tensors to fetch.
        updates: Additional update ops to be run at function call.
        name: a name to help users identify what this function does.
        session_kwargs: arguments to `tf.Session.run()`:
            `fetches`, `feed_dict`,
            `options`, `run_metadata`
    """

    def __init__(self, inputs, outputs,
                 updates=None,
                 name=None,
                 **session_kwargs):
        updates = updates or []
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` to a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(outputs, (list, tuple)):
            raise TypeError('`outputs` of a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(updates, (list, tuple)):
            raise TypeError('`updates` in a TensorFlow backend function '
                            'should be a list or tuple.')
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with tf.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if isinstance(update, tuple):
                    p, new_p = update
                    updates_ops.append(tf.assign(p, new_p))
                else:
                    # assumed already an op
                    updates_ops.append(update)
            self.updates_op = tf.group(*updates_ops)
        self.name = name
        # additional tensor substitutions
        self.feed_dict = session_kwargs.pop('feed_dict', {})
        # additional operations
        self.fetches = session_kwargs.pop('fetches', [])
        if not isinstance(self.fetches, list):
            self.fetches = [self.fetches]
        # The main use case of `fetches` being passed to a model is the ability
        # to run custom updates
        # (since the outputs of fetches are never returned).
        # This requires us to wrap fetches in `identity` ops.
        self.fetches = [tf.identity(x) for x in self.fetches]
        # self.session_kwargs is used for _legacy_call
        self.session_kwargs = session_kwargs.copy()
        self.run_options = session_kwargs.pop('options', None)
        self.run_metadata = session_kwargs.pop('run_metadata', None)
        if session_kwargs:
            raise ValueError('Some keys in session_kwargs are not '
                             'supported at this '
                             'time: %s', session_kwargs.keys())
        self._callable_fn = None
        self._feed_arrays = None
        self._feed_symbols = None
        self._symbol_vals = None
        self._session = None

    def _make_callable(self, feed_arrays, feed_symbols, symbol_vals, session):
        """Generates a callable that runs the graph.

        # Arguments
            feed_arrays: List of input tensors to be fed
                Numpy arrays at runtime.
            feed_symbols: List of input tensors to be fed
                symbolic tensors at runtime.
            symbol_vals: List of symbolic tensors to be fed to `feed_symbols`.
            session: Session to use to generate the callable.

        # Returns
            Function that runs the graph according to the above options.
        """
        # Prepare callable options.
        callable_opts = config_pb2.CallableOptions()
        # Handle external-data feed.
        for x in feed_arrays:
            callable_opts.feed.append(x.name)
        if self.feed_dict:
            for key in sorted(self.feed_dict.keys()):
                callable_opts.feed.append(key.name)
        # Handle symbolic feed.
        for x, y in zip(feed_symbols, symbol_vals):
            connection = callable_opts.tensor_connection.add()
            if x.dtype != y.dtype:
                y = tf.cast(y, dtype=x.dtype)
            from_tensor = tf_ops._as_graph_element(y)
            if from_tensor is None:
                from_tensor = y
            connection.from_tensor = from_tensor.name  # Data tensor
            connection.to_tensor = x.name  # Placeholder
        # Handle fetches.
        for x in self.outputs + self.fetches:
            callable_opts.fetch.append(x.name)
        # Handle updates.
        callable_opts.target.append(self.updates_op.name)
        # Handle run_options.
        if self.run_options:
            callable_opts.run_options.CopyFrom(self.run_options)
        # Create callable.
        callable_fn = session._make_callable_from_options(callable_opts)
        # Cache parameters corresponding to the generated callable, so that
        # we can detect future mismatches and refresh the callable.
        self._callable_fn = callable_fn
        self._feed_arrays = feed_arrays
        self._feed_symbols = feed_symbols
        self._symbol_vals = symbol_vals
        self._session = session

    def _call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` should be a list or tuple.')

        session = get_session()
        feed_arrays = []
        array_vals = []
        feed_symbols = []
        symbol_vals = []
        for tensor, value in zip(self.inputs, inputs):
            if value is None:
                continue
            if is_tensor(value):
                # Case: feeding symbolic tensor.
                feed_symbols.append(tensor)
                symbol_vals.append(value)
            else:
                feed_arrays.append(tensor)
                # We need to do array conversion and type casting
                # at this level, since
                # `callable_fn` only supports exact matches.
                array_vals.append(
                    np.asarray(value,
                               dtype=tf.as_dtype(tensor.dtype).as_numpy_dtype))
        if self.feed_dict:
            for key in sorted(self.feed_dict.keys()):
                array_vals.append(
                    np.asarray(self.feed_dict[key],
                               dtype=tf.as_dtype(key.dtype).as_numpy_dtype))

        # Refresh callable if anything has changed.
        if (self._callable_fn is None or
                feed_arrays != self._feed_arrays or
                symbol_vals != self._symbol_vals or
                feed_symbols != self._feed_symbols or
                session != self._session):
            self._make_callable(feed_arrays,
                                feed_symbols,
                                symbol_vals,
                                session)
        if self.run_metadata:
            fetched = self._callable_fn(*array_vals, run_metadata=self.run_metadata)
        else:
            fetched = self._callable_fn(*array_vals)
        return fetched[:len(self.outputs)]

    def _legacy_call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` should be a list or tuple.')
        feed_dict = self.feed_dict.copy()
        for tensor, value in zip(self.inputs, inputs):
            if is_sparse(tensor):
                sparse_coo = value.tocoo()
                indices = np.concatenate(
                    (np.expand_dims(sparse_coo.row, 1),
                     np.expand_dims(sparse_coo.col, 1)), 1)
                value = (indices, sparse_coo.data, sparse_coo.shape)
            feed_dict[tensor] = value
        fetches = self.outputs + [self.updates_op] + self.fetches
        session = get_session()
        updated = session.run(fetches=fetches, feed_dict=feed_dict,
                              **self.session_kwargs)
        return updated[:len(self.outputs)]

    def __call__(self, inputs):
        if hasattr(get_session(), '_make_callable_from_options'):
            if py_any(is_sparse(x) for x in self.inputs):
                if py_any(is_tensor(x) for x in inputs):
                    raise ValueError(
                        'Feeding from symbolic tensors is not '
                        'supported with sparse inputs.')
                return self._legacy_call(inputs)

            # callable generated by Session._make_callable_from_options accepts
            # `run_metadata` keyword argument since TF 1.10
            if (self.run_metadata and
                    StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.10.0')):
                if py_any(is_tensor(x) for x in inputs):
                    raise ValueError(
                        'In order to feed symbolic tensors to a Keras model and set '
                        '`run_metadata`, you need tensorflow 1.10 or higher.')
                return self._legacy_call(inputs)

            return self._call(inputs)
        else:
            if py_any(is_tensor(x) for x in inputs):
                raise ValueError(
                    'In order to feed symbolic tensors to a Keras model '
                    'in TensorFlow, you need tensorflow 1.8 or higher.')
            return self._legacy_call(inputs)


def function(inputs, outputs, updates=None, **kwargs):
    """Instantiates a Keras function.

    # Arguments
        inputs: List of placeholder tensors.
        outputs: List of output tensors.
        updates: List of update ops.
        **kwargs: Passed to `tf.Session.run`.

    # Returns
        Output values as Numpy arrays.

    # Raises
        ValueError: if invalid kwargs are passed in.
    """
    if kwargs:
        for key in kwargs:
            if not (has_arg(tf.Session.run, key, True) or has_arg(Function.__init__, key, True)):
                msg = 'Invalid argument "%s" passed to K.function with TensorFlow backend' % key
                raise ValueError(msg)
    return Function(inputs, outputs, updates=updates, **kwargs)
