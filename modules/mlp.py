import torch
from typing import Callable, Sequence

from .glu import GLU


class MLP(torch.nn.Sequential):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int],
        activation_module: Callable[[], torch.nn.Module],
        glu: bool,
        bias: bool,
        dropout: float,
        final_dropout: bool,
    ):
        self.hidden_sizes = list(hidden_sizes)
        self.activation_module = activation_module
        self.is_glu = glu

        actual_hidden_sizes = self.hidden_sizes
        if self.is_glu:
            if len(self.hidden_sizes) == 1:
                actual_hidden_sizes = [int(self.hidden_sizes[0] * 2 / 3)]
            else:
                raise NotImplementedError

        layers = list()
        prev_size = input_size
        for hidden_size in actual_hidden_sizes:
            if self.is_glu:
                layers.append(GLU(
                    prev_size,
                    hidden_size,
                    activation_module=activation_module,
                    bias=bias
                ))
            else:
                layers.append(torch.nn.Linear(prev_size, hidden_size, bias=bias))
                layers.append(activation_module())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(torch.nn.Linear(prev_size, output_size, bias=bias))
        if dropout > 0.0 and final_dropout:
            layers.append(torch.nn.Dropout(dropout))

        super().__init__(*layers)
