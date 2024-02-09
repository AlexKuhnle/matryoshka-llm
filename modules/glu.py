import torch
from typing import Callable


class GLU(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_module: Callable[[], torch.nn.Module],
    ):
        super().__init__()

        self.linear = torch.nn.Linear(input_size, output_size)
        self.gate = torch.nn.Linear(input_size, output_size)
        self.activation = activation_module()

    def forward(self, x):
        return self.activation(self.linear(x)) * self.gate(x)
