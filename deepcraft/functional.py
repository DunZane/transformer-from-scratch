"""Functional interface."""
import warnings
from typing import Optional, TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int


def _canonical_mask(
        mask: Optional[Tensor],
        mask_name: str,
        other_type: Optional[DType],
        other_name: str,
        target_type: DType,
        check_other: bool = True,
) -> Optional[Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )
        if check_other and other_type is not None:
            if other_type != other_type:
                warnings.warn(
                    f"Support for mismathed {mask_name} and {other_name}"
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(
                mask, float("-inf")
            )
    return mask
