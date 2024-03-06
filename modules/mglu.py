import torch
from typing import Callable, Sequence

from .glu import GLU
from .mlinear import MLinear


class MGLU(torch.nn.Module):

    @classmethod
    def get_non_matryoshka_module(cls):
        return GLU

    def __init__(
        self,
        input_sizes: Sequence[int],
        output_sizes: Sequence[int],
        activation_module: Callable[[], torch.nn.Module],
        bias: bool,
    ):
        super().__init__()

        self.linear = MLinear(input_sizes, output_sizes, bias=bias)
        self.gate = MLinear(input_sizes, output_sizes, bias=bias)
        self.activation = activation_module()

    def init_nested_module(self, index, module):
        self.linear.init_nested_module(index, module.linear)
        self.gate.init_nested_module(index, module.gate)

    def forward(self, x):
        return self.activation(self.linear(x)) * self.gate(x)
