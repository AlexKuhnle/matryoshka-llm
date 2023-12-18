import torch
from typing import Callable, Optional, Sequence

from .mlp import MLP
from .multi_head import MultiHead
from .self_attention import SelfAttention


class TransformerEncoder(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        mha_num_heads: int,
        mha_head_size: int,
        mlp_hidden_sizes: Sequence[int],
        mlp_activation_module: Callable[[], torch.nn.Module],
        mha_query_key_size: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.mha_layernorm = torch.nn.LayerNorm(input_size)
        self.mha = MultiHead(
            input_size,
            input_size,
            num_heads=mha_num_heads,
            head_size=mha_head_size,
            head_module=SelfAttention,
            head_kwargs=dict(query_key_size=mha_query_key_size),
            concat_dim=-1,
            dropout=dropout,
        )

        self.mlp_layernorm = torch.nn.LayerNorm(input_size)
        self.mlp = MLP(
            input_size,
            input_size,
            hidden_sizes=mlp_hidden_sizes,
            activation_module=mlp_activation_module,
            dropout=dropout,
        )

    def forward(self, x):
        res = self.mha_layernorm(x)
        res = self.mha(res)
        x = x + res
        res = self.mlp_layernorm(x)
        res = self.mlp(res)
        x = x + res
        return x
