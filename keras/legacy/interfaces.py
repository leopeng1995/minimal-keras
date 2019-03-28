import functools
import warnings

import numpy as np
import six


def raise_duplicate_arg_error(old_arg, new_arg):
    raise TypeError('For the `' + new_arg + '` argument, '
                                            'the layer received both '
                                            'the legacy keyword argument '
                                            '`' + old_arg + '` and the Keras 2 keyword argument '
                                                            '`' + new_arg + '`. Stick to the latter!')


def generate_legacy_interface(allowed_positional_args=None,
                              conversions=None,
                              preprocessor=None,
                              value_conversions=None,
                              object_type='class'):
    if allowed_positional_args is None:
        check_positional_args = False
    else:
        check_positional_args = True
    allowed_positional_args = allowed_positional_args or []
    conversions = conversions or []
    value_conversions = value_conversions or []

    def legacy_support(func):
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            if object_type == 'class':
                object_name = args[0].__class__.__name__
            else:
                object_name = func.__name__
            if preprocessor:
                args, kwargs, converted = preprocessor(args, kwargs)
            else:
                converted = []
            if check_positional_args:
                if len(args) > len(allowed_positional_args) + 1:
                    raise TypeError('`' + object_name +
                                    '` can accept only ' +
                                    str(len(allowed_positional_args)) +
                                    ' positional arguments ' +
                                    str(tuple(allowed_positional_args)) +
                                    ', but you passed the following '
                                    'positional arguments: ' +
                                    str(list(args[1:])))
            for key in value_conversions:
                if key in kwargs:
                    old_value = kwargs[key]
                    if old_value in value_conversions[key]:
                        kwargs[key] = value_conversions[key][old_value]
            for old_name, new_name in conversions:
                if old_name in kwargs:
                    value = kwargs.pop(old_name)
                    if new_name in kwargs:
                        raise_duplicate_arg_error(old_name, new_name)
                    kwargs[new_name] = value
                    converted.append((new_name, old_name))
            if converted:
                signature = '`' + object_name + '('
                for i, value in enumerate(args[1:]):
                    if isinstance(value, six.string_types):
                        signature += '"' + value + '"'
                    else:
                        if isinstance(value, np.ndarray):
                            str_val = 'array'
                        else:
                            str_val = str(value)
                        if len(str_val) > 10:
                            str_val = str_val[:10] + '...'
                        signature += str_val
                    if i < len(args[1:]) - 1 or kwargs:
                        signature += ', '
                for i, (name, value) in enumerate(kwargs.items()):
                    signature += name + '='
                    if isinstance(value, six.string_types):
                        signature += '"' + value + '"'
                    else:
                        if isinstance(value, np.ndarray):
                            str_val = 'array'
                        else:
                            str_val = str(value)
                        if len(str_val) > 10:
                            str_val = str_val[:10] + '...'
                        signature += str_val
                    if i < len(kwargs) - 1:
                        signature += ', '
                signature += ')`'
                warnings.warn('Update your `' + object_name + '` call to the ' +
                              'Keras 2 API: ' + signature, stacklevel=2)
            return func(*args, **kwargs)

        wrapper._original_function = func
        return wrapper

    return legacy_support


def get_updates_arg_preprocessing(args, kwargs):
    # Old interface: (params, constraints, loss)
    # New interface: (loss, params)
    if len(args) > 4:
        raise TypeError('`get_update` call received more arguments '
                        'than expected.')
    elif len(args) == 4:
        # Assuming old interface.
        opt, params, _, loss = args
        kwargs['loss'] = loss
        kwargs['params'] = params
        return [opt], kwargs, []
    elif len(args) == 3:
        if isinstance(args[1], (list, tuple)):
            assert isinstance(args[2], dict)
            assert 'loss' in kwargs
            opt, params, _ = args
            kwargs['params'] = params
            return [opt], kwargs, []
    return args, kwargs, []


legacy_get_updates_support = generate_legacy_interface(
    allowed_positional_args=None,
    conversions=[],
    preprocessor=get_updates_arg_preprocessing)


def add_weight_args_preprocessing(args, kwargs):
    if len(args) > 1:
        if isinstance(args[1], (tuple, list)):
            kwargs['shape'] = args[1]
            args = (args[0],) + args[2:]
            if len(args) > 1:
                if isinstance(args[1], six.string_types):
                    kwargs['name'] = args[1]
                    args = (args[0],) + args[2:]
    return args, kwargs, []


legacy_add_weight_support = generate_legacy_interface(
    allowed_positional_args=['name', 'shape'],
    preprocessor=add_weight_args_preprocessing)

legacy_input_support = generate_legacy_interface(
    allowed_positional_args=None,
    conversions=[('input_dtype', 'dtype')])

legacy_model_constructor_support = generate_legacy_interface(
    allowed_positional_args=None,
    conversions=[('input', 'inputs'),
                 ('output', 'outputs')])

generate_legacy_method_interface = functools.partial(generate_legacy_interface,
                                                     object_type='method')


# Model methods

def generator_methods_args_preprocessor(args, kwargs):
    converted = []
    if len(args) < 3:
        if 'samples_per_epoch' in kwargs:
            samples_per_epoch = kwargs.pop('samples_per_epoch')
            if len(args) > 1:
                generator = args[1]
            else:
                generator = kwargs['generator']
            if hasattr(generator, 'batch_size'):
                kwargs['steps_per_epoch'] = samples_per_epoch // generator.batch_size
            else:
                kwargs['steps_per_epoch'] = samples_per_epoch
            converted.append(('samples_per_epoch', 'steps_per_epoch'))

    keras1_args = {'samples_per_epoch', 'val_samples',
                   'nb_epoch', 'nb_val_samples', 'nb_worker'}
    if keras1_args.intersection(kwargs.keys()):
        warnings.warn('The semantics of the Keras 2 argument '
                      '`steps_per_epoch` is not the same as the '
                      'Keras 1 argument `samples_per_epoch`. '
                      '`steps_per_epoch` is the number of batches '
                      'to draw from the generator at each epoch. '
                      'Basically steps_per_epoch = samples_per_epoch/batch_size. '
                      'Similarly `nb_val_samples`->`validation_steps` and '
                      '`val_samples`->`steps` arguments have changed. '
                      'Update your method calls accordingly.', stacklevel=3)

    return args, kwargs, converted


legacy_generator_methods_support = generate_legacy_method_interface(
    allowed_positional_args=['generator', 'steps_per_epoch', 'epochs'],
    conversions=[('samples_per_epoch', 'steps_per_epoch'),
                 ('val_samples', 'steps'),
                 ('nb_epoch', 'epochs'),
                 ('nb_val_samples', 'validation_steps'),
                 ('nb_worker', 'workers'),
                 ('pickle_safe', 'use_multiprocessing'),
                 ('max_q_size', 'max_queue_size')],
    preprocessor=generator_methods_args_preprocessor)

legacy_dropout_support = generate_legacy_interface(
    allowed_positional_args=['rate', 'noise_shape', 'seed'],
    conversions=[('p', 'rate')])

legacy_dense_support = generate_legacy_interface(
    allowed_positional_args=['units'],
    conversions=[('output_dim', 'units'),
                 ('init', 'kernel_initializer'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('W_constraint', 'kernel_constraint'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')])


def conv2d_args_preprocessor(args, kwargs):
    converted = []
    if len(args) > 4:
        raise TypeError('Layer can receive at most 3 positional arguments.')
    elif len(args) == 4:
        if isinstance(args[2], int) and isinstance(args[3], int):
            new_keywords = ['padding', 'strides', 'data_format']
            for kwd in new_keywords:
                if kwd in kwargs:
                    raise ValueError(
                        'It seems that you are using the Keras 2 '
                        'and you are passing both `kernel_size` and `strides` '
                        'as integer positional arguments. For safety reasons, '
                        'this is disallowed. Pass `strides` '
                        'as a keyword argument instead.')
            kernel_size = (args[2], args[3])
            args = [args[0], args[1], kernel_size]
            converted.append(('kernel_size', 'nb_row/nb_col'))
    elif len(args) == 3 and isinstance(args[2], int):
        if 'nb_col' in kwargs:
            kernel_size = (args[2], kwargs.pop('nb_col'))
            args = [args[0], args[1], kernel_size]
            converted.append(('kernel_size', 'nb_row/nb_col'))
    elif len(args) == 2:
        if 'nb_row' in kwargs and 'nb_col' in kwargs:
            kernel_size = (kwargs.pop('nb_row'), kwargs.pop('nb_col'))
            args = [args[0], args[1], kernel_size]
            converted.append(('kernel_size', 'nb_row/nb_col'))
    elif len(args) == 1:
        if 'nb_row' in kwargs and 'nb_col' in kwargs:
            kernel_size = (kwargs.pop('nb_row'), kwargs.pop('nb_col'))
            kwargs['kernel_size'] = kernel_size
            converted.append(('kernel_size', 'nb_row/nb_col'))
    return args, kwargs, converted


legacy_conv2d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('init', 'kernel_initializer'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('W_constraint', 'kernel_constraint'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=conv2d_args_preprocessor)


legacy_pooling2d_support = generate_legacy_interface(
    allowed_positional_args=['pool_size', 'strides', 'padding'],
    conversions=[('border_mode', 'padding'),
                 ('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})
