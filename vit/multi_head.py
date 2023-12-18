import torch
from typing import Callable


class MultiHead(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_heads: int,
        head_size: int,
        head_module: Callable[..., torch.nn.Module],
        head_kwargs: dict,
        concat_dim: int = -1,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert isinstance(concat_dim, int) and concat_dim != 0
        self.concat_dim = concat_dim

        assert isinstance(num_heads, int) and num_heads >= 1
        self.heads = torch.nn.ModuleList([
            head_module(input_size, head_size, **head_kwargs)
            for _ in range(num_heads)
        ])

        self.output_proj = torch.nn.Linear(num_heads * head_size, output_size)

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        xs = [head(x) for head in self.heads]
        x = torch.cat(xs, dim=self.concat_dim)
        x = self.output_proj(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
