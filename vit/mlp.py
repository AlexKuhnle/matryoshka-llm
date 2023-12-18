import torch
from typing import Callable, Sequence


class MLP(torch.nn.Sequential):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int],
        activation_module: Callable[[], torch.nn.Module],
        dropout: float = 0.0,
        final_dropout: bool = True,
    ):
        layers = list()

        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(activation_module())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(torch.nn.Linear(prev_size, output_size))
        if dropout > 0.0 and final_dropout:
            layers.append(torch.nn.Dropout(dropout))

        super().__init__(*layers)
