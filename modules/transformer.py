import torch
from typing import Callable, Optional, Sequence

from .mhsa import MHSA
from .mlp import MLP


class Transformer(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        normalization_module: Callable[[int], torch.nn.Module],
        mhsa_num_heads: int,
        mhsa_head_size: int,
        mlp_hidden_sizes: Sequence[int],
        mlp_activation_module: Callable[[], torch.nn.Module],
        mhsa_kv_groups: Optional[int] = None,
        mhsa_qk_size: Optional[int] = None,
        mhsa_torch_sdpa: bool = True,
        mlp_glu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.mhsa_norm = normalization_module(input_size)

        self.mhsa = MHSA(
            input_size,
            input_size,
            num_heads=mhsa_num_heads,
            kv_groups=mhsa_kv_groups,
            head_size=mhsa_head_size,
            qk_size=mhsa_qk_size,
            torch_sdpa=mhsa_torch_sdpa,
            dropout=dropout,
        )

        self.mlp_norm = normalization_module(input_size)
        self.mlp = MLP(
            input_size,
            input_size,
            hidden_sizes=mlp_hidden_sizes,
            activation_module=mlp_activation_module,
            glu=mlp_glu,
            dropout=dropout,
        )

    def forward(self, x, mask=None, fn_apply_pos=None):
        res = self.mhsa_norm(x)
        res = self.mhsa(res, mask=mask, fn_apply_pos=fn_apply_pos)
        x = x + res
        res = self.mlp_norm(x)
        res = self.mlp(res)
        x = x + res
        return x
