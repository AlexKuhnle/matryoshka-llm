import torch
from typing import Callable, Optional, Sequence, Tuple, Union

from .mmhsa import MMHSA
from .mmlp import MMLP


class MTransformer(torch.nn.Module):

    def __init__(
        self,
        input_sizes: Sequence[int],
        normalization_module: Callable[[Sequence[int]], torch.nn.Module],
        mhsa_num_heads: int,
        mhsa_kv_groups: Optional[int],
        mhsa_head_sizes: Optional[Sequence[int]],
        mhsa_qk_sizes: Optional[Sequence[int]],
        mhsa_torch_sdpa: bool,
        mhsa_flash_sdpa: bool,
        mlp_hidden_sizes: Sequence[Sequence[int]],
        mlp_activation_module: Callable[[], torch.nn.Module],
        mlp_glu: bool,
        bias: bool,
        dropout: float,
    ):
        super().__init__()

        self.mhsa_norm = normalization_module(input_sizes)

        self.mhsa = MMHSA(
            input_sizes,
            input_sizes,
            num_heads=mhsa_num_heads,
            kv_groups=mhsa_kv_groups,
            head_sizes=mhsa_head_sizes,
            qk_sizes=mhsa_qk_sizes,
            torch_sdpa=mhsa_torch_sdpa,
            flash_sdpa=mhsa_flash_sdpa,
            bias=bias,
            dropout=dropout,
        )

        self.mlp_norm = normalization_module(input_sizes)
        self.mlp = MMLP(
            input_sizes,
            input_sizes,
            hidden_sizes=mlp_hidden_sizes,
            activation_module=mlp_activation_module,
            glu=mlp_glu,
            bias=bias,
            dropout=dropout,
            final_dropout=True,
        )

    def forward(
        self,
        x,
        mask: Optional[Union[bool, torch.Tensor]] = None,
        fn_apply_pos: Optional[Callable[[torch.Tensor, Optional[int]], torch.Tensor]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        res = self.mhsa_norm(x)
        if kv_cache is None:
            res = self.mhsa(res, mask=mask, fn_apply_pos=fn_apply_pos)
        else:
            res, kv_cache = self.mhsa(res, mask=mask, fn_apply_pos=fn_apply_pos, kv_cache=kv_cache)
        x = x + res
        res = self.mlp_norm(x)
        res = self.mlp(res)
        x = x + res
        if kv_cache is None:
            return x
        else:
            return x, kv_cache
