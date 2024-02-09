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
        glu: bool = False,
        dropout: float = 0.0,
        final_dropout: bool = True,
    ):
        if glu:
            if len(hidden_sizes) == 1:
                hidden_sizes = [int(hidden_sizes[0] * 2 / 3)]
            else:
                raise NotImplementedError

        layers = list()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            if glu:
                layers.append(GLU(prev_size, hidden_size, activation_module))
            else:
                layers.append(torch.nn.Linear(prev_size, hidden_size))
                layers.append(activation_module())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(torch.nn.Linear(prev_size, output_size))
        if dropout > 0.0 and final_dropout:
            layers.append(torch.nn.Dropout(dropout))

        super().__init__(*layers)



        input_size * hidden_size
        input_size * hidden_size + hidden_size * output_size
