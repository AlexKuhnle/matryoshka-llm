import torch


class RMSNorm(torch.nn.Module):

    def __init__(self, size: int, dim: int = -1):
        super().__init__()

        self.scale = torch.nn.Parameter(torch.ones(size))
        self.dim = dim

    def forward(self, x):
        rrms = torch.rsqrt((x * x).mean(self.dim, keepdim=True) + 1e-6)
        return x * rrms * self.scale
