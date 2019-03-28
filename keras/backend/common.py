# the type of float to use throughout the session.
_FLOATX = 'float32'
_EPSILON = 1e-7
_IMAGE_DATA_FORMAT = 'channels_last'


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```
    """
    return _EPSILON


def image_data_format():
    """Returns the default image data format convention ('channels_first' or 'channels_last').

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> keras.backend.image_data_format()
        'channels_first'
    ```
    """
    return _IMAGE_DATA_FORMAT


def floatx():
    """Returns the default float type, as a string.
    (e.g. 'float16', 'float32', 'float64').

    # Returns
        String, the current default float type.

    # Example
    ```python
        >>> keras.backend.floatx()
        'float32'
    ```
    """
    return _FLOATX


def normalize_data_format(value):
    """Checks that the value correspond to a valid data format.

    # Arguments
        value: String or None. `'channels_first'` or `'channels_last'`.

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.normalize_data_format(None)
        'channels_first'
        >>> K.normalize_data_format('channels_last')
        'channels_last'
    ```

    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    """
    if value is None:
        value = image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format
