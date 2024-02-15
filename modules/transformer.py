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
        mhsa_kv_groups: Optional[int],
        mhsa_head_size: int,
        mhsa_qk_size: Optional[int],
        mhsa_torch_sdpa: bool,
        mhsa_flash_sdpa: bool,
        mlp_hidden_sizes: Sequence[int],
        mlp_activation_module: Callable[[], torch.nn.Module],
        mlp_glu: bool,
        bias: bool,
        dropout: float,
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
            flash_sdpa=mhsa_flash_sdpa,
            bias=bias,
            dropout=dropout,
        )

        self.mlp_norm = normalization_module(input_size)
        self.mlp = MLP(
            input_size,
            input_size,
            hidden_sizes=mlp_hidden_sizes,
            activation_module=mlp_activation_module,
            glu=mlp_glu,
            bias=bias,
            dropout=dropout,
            final_dropout=True,
        )

    def forward(self, x, mask=None, fn_apply_pos=None):
        res = self.mhsa_norm(x)
        res = self.mhsa(res, mask=mask, fn_apply_pos=fn_apply_pos)
        x = x + res
        res = self.mlp_norm(x)
        res = self.mlp(res)
        x = x + res
        return x
