import torch
from typing import Callable, Optional, Sequence, Tuple

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
        embedding_norm: bool,
        position_scheme: str,
        position_per_layer: bool,
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

        self.context_length = context_length
        self.requires_mask_tensor = (not mhsa_torch_sdpa and not mhsa_flash_sdpa)

        self.embedding = torch.nn.Embedding(vocab_size, trafo_size)

        if embedding_norm:
            self.embedding_norm = normalization_module(trafo_size)
        else:
            self.embedding_norm = None

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
                mhsa_flash_sdpa=mhsa_flash_sdpa,
                mlp_hidden_sizes=mlp_hidden_sizes,
                mlp_activation_module=mlp_activation_module,
                mlp_glu=mlp_glu,
                bias=bias,
                dropout=dropout,
            ) for _ in range(num_trafos)
        ])

        self.final_norm = normalization_module(trafo_size)

        self.prediction = torch.nn.Linear(trafo_size, vocab_size, bias=bias)

    def empty_kv_cache(self, batch_size):
        return [(
            torch.empty(size=(batch_size, trafo.mhsa.num_heads, 0, trafo.mhsa.qk_size)).cuda(),
            torch.empty(size=(batch_size, trafo.mhsa.num_heads, 0, trafo.mhsa.head_size)).cuda(),
        ) for trafo in self.trafos]
        # return [(list(), list()) for _ in range(len(self.trafos))]

    def forward(
        self,
        x,
        kv_cache: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        if self.requires_mask_tensor or kv_cache is not None:
            q_length = kv_length = x.size(1)
            if kv_cache is not None:
                kv_length += kv_cache[0][0].size(2)
                # kv_length = q_length + len(kv_cache[0][0])
            mask = torch.ones(q_length, kv_length, dtype=torch.bool, device=x.device)
            mask = mask.tril(diagonal=(kv_length - q_length))
            mask = mask.expand((1, 1, *mask.size()))
        else:
            mask = True

        x = self.embedding(x)
        if self.embedding_norm is not None:
            x = self.embedding_norm(x)
        if not self.pos_per_layer:
            x = self.fn_apply_pos(x)
        if self.input_dropout is not None:
            x = self.input_dropout(x)

        for n, trafo in enumerate(self.trafos):
            kwargs = dict(mask=mask)
            if self.pos_per_layer:
                kwargs["fn_apply_pos"] = self.fn_apply_pos
            if kv_cache is not None:
                kwargs["kv_cache"] = kv_cache[n]

            if kv_cache is None:
                x = trafo(x, **kwargs)
            else:
                x, kv_cache[n] = trafo(x, **kwargs)

        x = self.final_norm(x)
        x = self.prediction(x)

        if kv_cache is None:
            return x
        else:
            return x, kv_cache
