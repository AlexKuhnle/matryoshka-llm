import torch
from typing import Callable, Sequence, Tuple


class CNN(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        image_size: Tuple[int, int],
        conv_sizes: Sequence[int] = [8, 16, 32],
        conv_activation_module: Callable[[], torch.nn.Module] = torch.nn.ReLU,
        mlp_sizes: Sequence[int] = [64],
        mlp_activation_module: Callable[[], torch.nn.Module] = torch.nn.Tanh,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = list()
        prev_size = input_size
        height = image_size[0]
        width = image_size[1]
        for conv_size in conv_sizes:
            layers.append(
                torch.nn.Conv2d(prev_size, conv_size, kernel_size=3, stride=2, padding=1)
            )
            width = width // 2 + width % 2
            height = height // 2 + height % 2
            layers.append(torch.nn.LayerNorm((conv_size, height, width)))
            layers.append(conv_activation_module())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            prev_size = conv_size
        self.convs = torch.nn.Sequential(*layers)

        self.flatten = torch.nn.Flatten(1, -1)

        layers = list()
        prev_size = height * width * prev_size
        for mlp_size in mlp_sizes:
            layers.append(torch.nn.Linear(prev_size, mlp_size))
            layers.append(mlp_activation_module())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            prev_size = mlp_size
        layers.append(torch.nn.Linear(prev_size, output_size))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x
