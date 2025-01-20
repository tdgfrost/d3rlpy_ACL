from typing import cast, Optional

import torch
import torch.nn.functional as F
from torch import nn
import zuko
from torch._prims_common import DeviceLikeType
from torch.nn.modules.module import T

from ...types import TorchObservation
from .encoders import Encoder

__all__ = ["NSFFunction", "compute_nsf_function_error"]


class NSFFunction(nn.Module):  # type: ignore
    _encoder: Encoder

    def __init__(self, encoder: Encoder):
        super().__init__()
        self._encoder = encoder

    def forward(self, x: TorchObservation) -> torch.Tensor:
        return cast(torch.Tensor, self._encoder(x))

    def to(
        self: T, device: Optional[DeviceLikeType]
    ) -> T:
        self._encoder = self._encoder.to(device)
        return self._encoder

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))


def compute_nsf_function_error(
    nsf_function: NSFFunction,
    observations: TorchObservation,
    target: torch.Tensor,
) -> torch.Tensor:
    return None
