import torch
from typing import Callable, Sequence

from .mglu import MGLU
from .mlinear import MLinear


class MMLP(torch.nn.Sequential):

    def __init__(
        self,
        input_sizes: Sequence[int],
        output_sizes: Sequence[int],
        hidden_sizes: Sequence[Sequence[int]],
        activation_module: Callable[[], torch.nn.Module],
        glu: bool,
        bias: bool,
        dropout: float,
        final_dropout: bool,
    ):
        if glu:
            if len(hidden_sizes) == 1:
                hidden_sizes = [[int(hidden_size * 2 / 3) for hidden_size in hidden_sizes[0]]]
            else:
                raise NotImplementedError

        layers = list()
        prev_sizes = input_sizes
        for _hidden_sizes in hidden_sizes:
            if glu:
                layers.append(MGLU(
                    prev_sizes,
                    _hidden_sizes,
                    activation_module=activation_module,
                    bias=bias
                ))
            else:
                layers.append(MLinear(prev_sizes, _hidden_sizes, bias=bias))
                layers.append(activation_module())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            prev_sizes = _hidden_sizes

        layers.append(MLinear(prev_sizes, output_sizes, bias=bias))
        if dropout > 0.0 and final_dropout:
            layers.append(torch.nn.Dropout(dropout))

        super().__init__(*layers)
