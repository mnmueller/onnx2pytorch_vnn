import torch.nn.functional as F

from ..operations.base import Operator


# TODO @Robin Changed here
class Pad(Operator):
    def __init__(self, mode="constant", padding=None, constant=0.0):
        self.mode = mode
        self.padding = padding
        self.constant = constant
        super().__init__()

    def forward(self, input, pads=None, value=0):
        if self.padding is not None:
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        out = F.pad(input, list(pads), mode=self.mode, value=self.constant)
        return out

    def extra_repr(self) -> str:
        return "mode={}, padding={}".format(self.mode, self.padding)
