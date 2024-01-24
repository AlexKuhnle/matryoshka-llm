import torch
from typing import Callable, Optional, Sequence

from .mlp import MLP
from .transformer import Transformer


class GPT(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 1024,
        num_trafos: int = 8,
        trafo_size: int = 1024,
        trafo_mha_num_heads: int = 8,
        trafo_mha_head_size: int = 128,
        trafo_mha_query_key_size: Optional[int] = None,
        trafo_mlp_hidden_sizes: Sequence[int] = [1024],
        trafo_mlp_activation_module: Callable[[], torch.nn.Module] = torch.nn.GELU,
        mlp_hidden_sizes: Sequence[int] = [1024],
        mlp_activation_module: torch.nn.Module = torch.nn.Tanh,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.context_length = context_length

        self.embedding = torch.nn.Embedding(vocab_size, trafo_size)

        initial_value = torch.randn(1, self.context_length, trafo_size) * 0.01
        self.pos_embeddings = torch.nn.Parameter(initial_value, requires_grad=True)

        if dropout > 0.0:
            self.input_dropout = torch.nn.Dropout(dropout)
        else:
            self.input_dropout = None

        self.trafos = torch.nn.ModuleList([
            Transformer(
                trafo_size,
                mha_num_heads=trafo_mha_num_heads,
                mha_head_size=trafo_mha_head_size,
                mha_query_key_size=trafo_mha_query_key_size,
                mlp_hidden_sizes=trafo_mlp_hidden_sizes,
                mlp_activation_module=trafo_mlp_activation_module,
                dropout=dropout,
            ) for _ in range(num_trafos)
        ])

        self.trafo_layernorm = torch.nn.LayerNorm(trafo_size)

        self.mlp = MLP(
            trafo_size,
            vocab_size,
            hidden_sizes=mlp_hidden_sizes,
            activation_module=mlp_activation_module,
            dropout=dropout,
            final_dropout=False,
        )

    def forward(self, x):
        batch_size, context_length = x.size()
        assert context_length <= self.context_length
        pos_embeddings = self.pos_embeddings[:, :context_length]
        pos_embeddings = torch.tile(pos_embeddings, (batch_size, 1, 1))
        mask = torch.ones(context_length, context_length, dtype=torch.bool, device=x.device).tril()
        mask = torch.tile(mask, (batch_size, 1, 1))

        x = self.embedding(x)
        x = x + pos_embeddings
        if self.input_dropout is not None:
            x = self.input_dropout(x)

        for trafo in self.trafos:
            x = trafo(x, mask)
        x = self.trafo_layernorm(x)

        x = self.mlp(x)

        return x
