import torch
from typing import Sequence

from .rms_norm import RMSNorm


class MRMSNorm(torch.nn.Module):

    @classmethod
    def get_non_matryoshka_module(cls):
        return RMSNorm

    def __init__(self, sizes: Sequence[int], dim: int = -1):
        super().__init__()

        assert all(sizes[n] < sizes[n + 1] for n in range(len(sizes) - 1))
        self.sizes = list(sizes)

        self.scale = torch.nn.Parameter(torch.ones(self.sizes[-1]))
        self.dim = dim

    def init_nested_module(self, index, module):
        assert 0 <= index < len(self.sizes)
        module.scale.copy_(self.scale[:self.sizes[index]])

    def forward(self, x):
        xs = list()
        prev_size = 0
        for size in self.sizes:
            rrms = torch.rsqrt((x[..., :size] * x[..., :size]).mean(self.dim, keepdim=True) + 1e-6)
            xs.append(x[..., prev_size: size] * rrms)
            prev_size = size
        x = torch.cat(xs, dim=-1)
        return x * self.scale
