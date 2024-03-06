import torch
from typing import Callable, Sequence

from .mglu import MGLU
from .mlinear import MLinear
from .mlp import MLP


class MMLP(torch.nn.Sequential):

    @classmethod
    def get_non_matryoshka_module(cls):
        return MLP

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
        self.hidden_sizes = list(hidden_sizes)
        self.activation_module = activation_module
        self.is_glu = glu

        actual_hidden_sizes = self.hidden_sizes
        if self.is_glu:
            if len(self.hidden_sizes) == 1:
                actual_hidden_sizes = [[int(size * 2 / 3) for size in hidden_sizes[0]]]
            else:
                raise NotImplementedError

        layers = list()
        prev_sizes = input_sizes
        for hidden_sizes in actual_hidden_sizes:
            if self.is_glu:
                layers.append(MGLU(
                    prev_sizes,
                    hidden_sizes,
                    activation_module=activation_module,
                    bias=bias
                ))
            else:
                layers.append(MLinear(prev_sizes, hidden_sizes, bias=bias))
                layers.append(activation_module())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            prev_sizes = hidden_sizes

        layers.append(MLinear(prev_sizes, output_sizes, bias=bias))
        if dropout > 0.0 and final_dropout:
            layers.append(torch.nn.Dropout(dropout))

        super().__init__(*layers)

    def init_nested_module(self, index, module):
        for source, target in zip(self, module):
            if isinstance(source, (MLinear, MGLU)):
                source.init_nested_module(index, target)
