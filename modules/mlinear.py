import math
import torch
from typing import Sequence


class MLinear(torch.nn.Module):

    def __init__(self, input_sizes: Sequence[int], output_sizes: Sequence[int], bias: bool):
        super().__init__()

        assert len(input_sizes) == len(output_sizes)
        assert all(input_sizes[n] < input_sizes[n + 1] for n in range(len(input_sizes) - 1))
        assert all(output_sizes[n] < output_sizes[n + 1] for n in range(len(output_sizes) - 1))
        self.input_sizes = input_sizes

        self.weight_blocks = torch.nn.ParameterList()
        prev_output_size = 0
        for input_size, output_size in zip(input_sizes, output_sizes):
            block = torch.nn.Parameter(torch.empty(input_size, output_size - prev_output_size))
            torch.nn.init.kaiming_uniform_(block, a=math.sqrt(5))
            self.weight_blocks.append(block)
            prev_output_size = output_size

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_sizes[-1]))
        else:
            self.bias = None

    def weight(self):
        return torch.cat([
            torch.nn.functional.pad(block, (0, 0, 0, self.input_sizes[-1] - input_size))  # !!!!!!!!!!!!!!!!!!!!!!!!! move to constructor and register_buffer?
            for block, input_size in zip(self.weight_blocks, self.input_sizes)
        ], dim=1)

    def forward(self, x):
        x = x @ self.weight()
        if self.bias is not None:
            x += self.bias
        return x
