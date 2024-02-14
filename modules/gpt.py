import torch
from typing import Callable, Optional, Sequence

from .mlp import MLP
from .position import init_position_scheme
from .transformer import Transformer


class GPT(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_trafos: int,
        trafo_size: int,
        position_scheme: str,
        position_per_layer: bool,
        normalization_module: Callable[[int], torch.nn.Module],
        mhsa_num_heads: int,
        mhsa_kv_groups: Optional[int],
        mhsa_head_size: int,
        mhsa_qk_size: Optional[int],
        mhsa_torch_sdpa: bool,
        mlp_hidden_sizes: Sequence[int],
        mlp_activation_module: Callable[[], torch.nn.Module],
        mlp_glu: bool,
        dropout: float,
    ):
        super().__init__()

        self.context_length = context_length
        self.mhsa_torch_sdpa = mhsa_torch_sdpa

        self.embedding = torch.nn.Embedding(vocab_size, trafo_size)

        self.fn_apply_pos, self.pos_embeddings = init_position_scheme(
            scheme=position_scheme,
            context_length=self.context_length,
            trafo_size=(mhsa_head_size if position_per_layer else trafo_size),
        )
        self.pos_per_layer = position_per_layer

        if dropout > 0.0:
            self.input_dropout = torch.nn.Dropout(dropout)
        else:
            self.input_dropout = None

        self.trafos = torch.nn.ModuleList([
            Transformer(
                trafo_size,
                normalization_module=normalization_module,
                mhsa_num_heads=mhsa_num_heads,
                mhsa_kv_groups=mhsa_kv_groups,
                mhsa_head_size=mhsa_head_size,
                mhsa_qk_size=mhsa_qk_size,
                mhsa_torch_sdpa=mhsa_torch_sdpa,
                mlp_hidden_sizes=mlp_hidden_sizes,
                mlp_activation_module=mlp_activation_module,
                mlp_glu=mlp_glu,
                dropout=dropout,
            ) for _ in range(num_trafos)
        ])

        self.final_norm = normalization_module(trafo_size)

        self.prediction = torch.nn.Linear(trafo_size, vocab_size)

    def forward(self, x):
        batch_size, context_length = x.size()
        assert context_length <= self.context_length
        if self.mhsa_torch_sdpa:
            mask = True
        else:
            mask = torch.ones(context_length, context_length, dtype=torch.bool, device=x.device)
            mask = mask.tril()
            mask = torch.tile(mask, (batch_size, 1, 1))

        x = self.embedding(x)
        if not self.pos_per_layer:
            x = self.fn_apply_pos(x)
        if self.input_dropout is not None:
            x = self.input_dropout(x)
        for trafo in self.trafos:
            if self.pos_per_layer:
                x = trafo(x, mask=mask, fn_apply_pos=self.fn_apply_pos)
            else:
                x = trafo(x, mask=mask)
        x = self.final_norm(x)
        x = self.prediction(x)
        return x
