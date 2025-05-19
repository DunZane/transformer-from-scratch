# mypy: allow-untyped-defs
"""This file contains utilities for initializing neural network parameters."""
import math

import torch
from torch import Tensor, nn
from typing import Optional as _Optional


def _calculate_fan_in_and_fan_out(tensor):
    r"""To get tensor's fan_in and fan_out

    :param tensor:
    :return:
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer 2 dimensions.")

    num_input_fmaps = tensor.size(1)  # use in Linear: [out, in] = [64, 128]
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not support by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s  # use in conv2D: [out, in, kH, kW]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _no_grad_uniform_(tensor, a, b, generator=None):
    with torch.no_grad():
        return torch.uniform_(tensor, a, b)


def _no_grad_fill_(tensor, val):
    with torch.no_grad():
        return
    pass


def constant_(tensor: Tensor, val: float) -> Tensor:
    r"""Fill the input tensor with the value :math:`\text{val}`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        val: the value to fill tensor with

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            constant_, (tensor), tensor=tensor, val=val
        )
    return _no_grad_fill_(tensor, val)


def xavier_uniform_(
        tensor: Tensor,
        gain: float = 1.0,
        generator: _Optional[torch.Generator] = None) -> None:
    r"""Fill the input `Tensor` with values using a Xavier uniform distribution

    The method is described in `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).
    The resulting tensor will have values samples from
    :math:`\mathcal{U}(-a,a)` where

    .. math:
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as the Glorot initialization

    Args:
        tensor: an n-dimensional `tensor.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from the distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))

    Note:
        Be aware the ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape=[fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.xavier_uniform_(w.T, ...)``.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std

    return _no_grad_uniform_(tensor, -a, a, generator)
