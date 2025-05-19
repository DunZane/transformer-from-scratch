# mypy: allow-untyped-defs

__all__ = [
    "MultiheadAttention"
]

from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from deepcraft.init import xavier_uniform_, constant_


class MultiheadAttention(nn.Module):
    r"""Parallel attention mechanisms for capturing diverse contextual patterns.

    .. refer note::
        - pytorch official: https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/activation.py
        - annotated transformer: http://nlp.seas.harvard.edu/annotated-transformer/

    Method described in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^0

    where :math:`\text{head}_i = \text{Attention}(QW_i^Q,KW_i^K,VW_i^V)`.

    Args：
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on the ``attn_output_weights``. Default: ``0.0``(no dropout).
        bias: If specified, add bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    """

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout=0.0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            device=None,
            dtype=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim}, num_heads={num_heads} instead."
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * self.num_heads == embed_dim
        ), "head_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(
                torch.empty(embed_dim, embed_dim), **factory_kwargs
            )
            self.k_proj_weight = Parameter(
                torch.empty(embed_dim, self.kdim), **factory_kwargs
            )
            self.v_proj_weight = Parameter(
                torch.empty(embed_dim, self.vdim), **factory_kwargs
            )

            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty(3 * embed_dim, embed_dim), **factory_kwargs
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim), **factory_kwargs)
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(  # like nn.linear, but non-dynamically quantizable
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs))  # (batch_size, seq_len, embed_dim)
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        r""" Init paras and bias before model train.
        """
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj_bias, 0.0)
        if self.bias_k is not None:
            constant_(self.bias_k, 0.0)
        if self.bias_v is not None:
            constant_(self.bias_v,0.0)



