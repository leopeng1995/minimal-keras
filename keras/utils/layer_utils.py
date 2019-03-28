import numpy as np


def count_params(weights):
    """Count the total number of scalars composing the weights.

    # Arguments
        weights: An iterable containing the weights on which to compute params

    # Returns
        The total number of scalars composing the weights
    """
    return int(np.sum([K.count_params(p) for p in set(weights)]))


def get_source_inputs(tensor, layer=None, node_index=None):
    """Returns the list of input tensors necessary to compute `tensor`.

    Output will always be a list of tensors
    (potentially with 1 element).

    # Arguments
        tensor: The tensor to start from.
        layer: Origin layer of the tensor. Will be
            determined via tensor._keras_history if not provided.
        node_index: Origin node index of the tensor.

    # Returns
        List of input tensors.
    """
    if not hasattr(tensor, '_keras_history'):
        return tensor

    if layer is None or node_index:
        layer, node_index, _ = tensor._keras_history
    if not layer._inbound_nodes:
        return [tensor]
    else:
        node = layer._inbound_nodes[node_index]
        if not node.inbound_layers:
            # Reached an Input layer, stop recursion.
            return node.input_tensors
        else:
            source_tensors = []
            for i in range(len(node.inbound_layers)):
                x = node.input_tensors[i]
                layer = node.inbound_layers[i]
                node_index = node.node_indices[i]
                previous_sources = get_source_inputs(x,
                                                     layer,
                                                     node_index)
                # Avoid input redundancy.
                for x in previous_sources:
                    if x not in source_tensors:
                        source_tensors.append(x)
            return source_tensors

