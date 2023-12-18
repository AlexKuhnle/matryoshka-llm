import torch
from typing import Callable, Optional, Sequence, Tuple

from .mlp import MLP
from .patchify import Patchify
from .transformer_encoder import TransformerEncoder


class ViT(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        image_size: Tuple[int, int],
        patch_size: int = 4,
        num_trafos: int = 2,
        trafo_size: int = 16,
        trafo_mha_num_heads: int = 4,
        trafo_mha_head_size: int = 8,
        trafo_mha_query_key_size: Optional[int] = None,
        trafo_mlp_hidden_sizes: Sequence[int] = [16],
        trafo_mlp_activation_module: Callable[[], torch.nn.Module] = torch.nn.GELU,
        mlp_hidden_sizes: Sequence[int] = [16],
        mlp_activation_module: torch.nn.Module = torch.nn.Tanh,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patchify = Patchify(patch_size=patch_size)

        self.input_proj = torch.nn.Linear(patch_size * patch_size * input_size, trafo_size)

        num_patches = (image_size[0] // patch_size + (image_size[0] % patch_size > 0)) * \
            (image_size[1] // patch_size + (image_size[1] % patch_size > 0))
        initial_value = torch.randn(1, num_patches + 1, trafo_size) * 0.01
        self.pos_embeddings = torch.nn.Parameter(initial_value, requires_grad=True)

        initial_value = torch.randn(1, 1, trafo_size) * 0.01
        self.class_embedding = torch.nn.Parameter(initial_value, requires_grad=True)

        if dropout > 0.0:
            self.input_dropout = torch.nn.Dropout(dropout)
        else:
            self.input_dropout = None

        self.trafos = torch.nn.Sequential(*(
            TransformerEncoder(
                trafo_size,
                mha_num_heads=trafo_mha_num_heads,
                mha_head_size=trafo_mha_head_size,
                mha_query_key_size=trafo_mha_query_key_size,
                mlp_hidden_sizes=trafo_mlp_hidden_sizes,
                mlp_activation_module=trafo_mlp_activation_module,
                dropout=dropout,
            ) for _ in range(num_trafos)
        ))

        self.trafo_layernorm = torch.nn.LayerNorm(trafo_size)

        self.mlp = MLP(
            trafo_size,
            output_size,
            hidden_sizes=mlp_hidden_sizes,
            activation_module=mlp_activation_module,
            dropout=dropout,
            final_dropout=False,
        )

    def forward(self, x):
        x = self.patchify(x)
        x = self.input_proj(x)

        pos_embeddings = torch.tile(self.pos_embeddings, (x.size(0), 1, 1))
        class_embedding = torch.tile(self.class_embedding, (x.size(0), 1, 1))
        x = torch.cat([class_embedding, x], dim=1) + pos_embeddings
        if self.input_dropout is not None:
            x = self.input_dropout(x)

        x = self.trafos(x)
        x = self.trafo_layernorm(x)

        x = x[:, 0]
        x = self.mlp(x)

        return x
