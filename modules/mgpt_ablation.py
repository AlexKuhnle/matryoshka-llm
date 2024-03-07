import torch
from typing import Callable, Optional, Sequence

from .position import init_position_scheme
from .transformer import Transformer


class MGPTAblation(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        prediction_sizes: Sequence[int],
        prediction_multihead: bool,
        num_trafos: int,
        trafo_size: int,
        embedding_norm: bool,
        position_scheme: str,
        position_per_layer: bool,
        normalization_module: Callable[[int], torch.nn.Module],
        mhsa_num_heads: int,
        mhsa_kv_groups: Optional[int],
        mhsa_head_size: Optional[int],
        mhsa_qk_size: Optional[int],
        mhsa_torch_sdpa: bool,
        mlp_hidden_sizes: Sequence[int],
        mlp_activation_module: Callable[[], torch.nn.Module],
        mlp_glu: bool,
        bias: bool,
        dropout: float,
    ):
        super().__init__()

        self.context_length = context_length
        self.requires_mask_tensor = (not mhsa_torch_sdpa)
        self.trafo_size = trafo_size

        self.embedding = torch.nn.Embedding(vocab_size, self.trafo_size)

        if embedding_norm:
            self.embedding_norm = normalization_module(self.trafo_size)
        else:
            self.embedding_norm = None

        if dropout > 0.0:
            self.input_dropout = torch.nn.Dropout(dropout)
        else:
            self.input_dropout = None

        self.trafos = torch.nn.ModuleList([
            Transformer(
                self.trafo_size,
                normalization_module=normalization_module,
                mhsa_num_heads=mhsa_num_heads,
                mhsa_kv_groups=mhsa_kv_groups,
                mhsa_head_size=mhsa_head_size,
                mhsa_qk_size=mhsa_qk_size,
                mhsa_torch_sdpa=mhsa_torch_sdpa,
                mlp_hidden_sizes=mlp_hidden_sizes,
                mlp_activation_module=mlp_activation_module,
                mlp_glu=mlp_glu,
                bias=bias,
                dropout=dropout,
            ) for _ in range(num_trafos)
        ])

        self.position_scheme = position_scheme
        self.fn_apply_pos, pos_embeddings = init_position_scheme(
            scheme=position_scheme,
            context_length=self.context_length,
            trafo_size=(self.trafos[0].mhsa.head_size if position_per_layer else self.trafo_size),
        )
        if isinstance(pos_embeddings, torch.nn.Parameter):
            self.pos_embeddings = pos_embeddings
        self.pos_per_layer = position_per_layer

        self.final_norm = normalization_module(self.trafo_size)

        assert all(
            prediction_sizes[n] < prediction_sizes[n + 1]
            for n in range(len(prediction_sizes) - 1)
        )
        assert prediction_sizes[-1] == trafo_size
        self.prediction_sizes = list(self.prediction_sizes)
        self.prediction_multihead = prediction_multihead
        if self.prediction_multihead:
            self.predictions = torch.nn.ModuleList([
                torch.nn.Linear(pred_size, vocab_size, bias=bias)
                for pred_size in self.prediction_sizes
            ])
        else:
            self.prediction = torch.nn.Linear(self.trafo_size, vocab_size, bias=bias)

    def forward(
        self,
        x,
    ):
        if self.requires_mask_tensor:
            q_length = x.size(1)
            mask = torch.ones(q_length, q_length, dtype=torch.bool, device=x.device)
            mask = mask.tril()
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

            x = trafo(x, **kwargs)

        x = self.final_norm(x)
        if self.prediction_multihead:
            x = [
                prediction(x[..., :pred_size])
                for prediction, pred_size in zip(self.predictions, self.prediction_sizes)
            ]
        else:
            x = self.prediction(x)
            x = [x[..., :pred_size] for pred_size in self.prediction_sizes]

        return x
