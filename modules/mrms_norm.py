import torch
from typing import Sequence


class MRMSNorm(torch.nn.Module):

    def __init__(self, sizes: Sequence[int], dim: int = -1):
        super().__init__()

        self.sizes = list(sizes)
        self.scale = torch.nn.Parameter(torch.ones(self.sizes[-1]))
        self.dim = dim

    def forward(self, x):
        xs = list()
        prev_size = 0
        for size in self.sizes:
            rrms = torch.rsqrt((x[..., :size] * x[..., :size]).mean(self.dim, keepdim=True) + 1e-6)
            xs.append(x[..., prev_size: size] * rrms)
            prev_size = size
        x = torch.cat(xs, dim=-1)
        return x * self.scale
