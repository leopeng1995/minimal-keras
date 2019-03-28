import sys

from .common import image_data_format

# Default backend: TensorFlow.
_BACKEND = 'tensorflow'

sys.stderr.write('Using TensorFlow backend.\n')
from .tensorflow_backend import *


def backend():
    """Publicly accessible method
    for determining the current backend.

    # Returns
        String, the name of the backend Keras is currently using.

    # Example
    ```python
        >>> keras.backend.backend()
        'tensorflow'
    ```
    """
    return _BACKEND
