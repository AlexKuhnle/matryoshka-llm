import torch
from typing import Callable

from .mlinear import MLinear


class MGLU(torch.nn.Module):

    def __init__(
        self,
        input_sizes: int,
        output_sizes: int,
        activation_module: Callable[[], torch.nn.Module],
        bias: bool,
    ):
        super().__init__()

        self.linear = MLinear(input_sizes, output_sizes, bias=bias)
        self.gate = MLinear(input_sizes, output_sizes, bias=bias)
        self.activation = activation_module()

    def forward(self, x):
        return self.activation(self.linear(x)) * self.gate(x)
