import torch
from typing import Callable, List, Optional, Sequence, Tuple

from .gpt import GPT
from .mmlp import MMLP
from .mtransformer import MTransformer
from .position import init_position_scheme


class MGPT(torch.nn.Module):

    @classmethod
    def get_non_matryoshka_module(cls):
        return GPT

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_trafos: int,
        trafo_sizes: Sequence[int],
        embedding_norm: bool,
        position_scheme: str,
        position_per_layer: bool,
        normalization_module: Callable[[Sequence[int]], torch.nn.Module],
        mhsa_num_heads: int,
        mhsa_kv_groups: Optional[int],
        mhsa_head_sizes: Optional[Sequence[int]],
        mhsa_qk_sizes: Optional[Sequence[int]],
        mhsa_torch_sdpa: bool,
        mlp_hidden_sizes: Sequence[Sequence[int]],
        mlp_activation_module: Callable[[], torch.nn.Module],
        mlp_glu: bool,
        bias: bool,
        dropout: float,
    ):
        super().__init__()

        self.context_length = context_length
        self.requires_mask_tensor = (not mhsa_torch_sdpa)
        self.trafo_sizes = list(trafo_sizes)

        self.embedding = torch.nn.Embedding(vocab_size, self.trafo_sizes[-1])

        if embedding_norm:
            self.embedding_norm = normalization_module(self.trafo_sizes)
        else:
            self.embedding_norm = None

        if dropout > 0.0:
            self.input_dropout = torch.nn.Dropout(dropout)
        else:
            self.input_dropout = None

        self.trafos = torch.nn.ModuleList([
            MTransformer(
                self.trafo_sizes,
                normalization_module=normalization_module,
                mhsa_num_heads=mhsa_num_heads,
                mhsa_kv_groups=mhsa_kv_groups,
                mhsa_head_sizes=mhsa_head_sizes,
                mhsa_qk_sizes=mhsa_qk_sizes,
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
            scheme=self.position_scheme,
            context_length=self.context_length,
            trafo_sizes=(self.trafos[0].mhsa.head_sizes if position_per_layer else self.trafo_sizes),
        )
        if isinstance(pos_embeddings, torch.nn.Parameter):
            self.pos_embeddings = pos_embeddings
        self.position_per_layer = position_per_layer

        self.final_norm = normalization_module(self.trafo_sizes)

        self.predictions = torch.nn.ModuleList([
            torch.nn.Linear(trafo_size, vocab_size, bias=bias)
            for trafo_size in self.trafo_sizes
        ])
    
    def get_nested_kwargs(self, index, force_non_matryoshka):
        kwargs = dict(
            vocab_size=self.embedding.num_embeddings,
            context_length=self.context_length,
            num_trafos=len(self.trafos),
            embedding_norm=(self.embedding_norm is not None),
            position_scheme=self.position_scheme,
            position_per_layer=self.position_per_layer,
            mhsa_num_heads=self.trafos[0].mhsa.num_heads,
            mhsa_kv_groups=self.trafos[0].mhsa.kv_groups,
            mhsa_torch_sdpa=self.trafos[0].mhsa.torch_sdpa,
            mlp_activation_module=self.trafos[0].mlp.activation_module,
            mlp_glu=self.trafos[0].mlp.is_glu,
            bias=(self.trafos[0].mhsa.query_proj.bias is not None),
            dropout=(0.0 if self.input_dropout is None else self.input_dropout.p),
        )
        if index == 0 or force_non_matryoshka:
            kwargs.update(
                trafo_size=self.trafo_sizes[index],
                normalization_module=self.final_norm.__class__.get_non_matryoshka_module(),
                mhsa_head_size=self.trafos[0].mhsa.head_sizes[index],
                mhsa_qk_size=self.trafos[0].mhsa.qk_sizes[index],
                mlp_hidden_sizes=[sizes[index] for sizes in self.trafos[0].mlp.hidden_sizes],
            )
        else:
            kwargs.update(
                trafo_sizes=self.trafo_sizes[:index + 1],
                normalization_module=self.final_norm.__class__,
                mhsa_head_sizes=self.trafos[0].mhsa.head_sizes[:index + 1],
                mhsa_qk_sizes=self.trafos[0].mhsa.qk_sizes[:index + 1],
                mlp_hidden_sizes=[sizes[:index + 1] for sizes in self.trafos[0].mlp.hidden_sizes],
            )
        return kwargs

    def init_nested_module(self, index, module):
        module.embedding.weight.copy_(self.embedding.weight[:, :self.trafo_sizes[index]])
        if self.embedding_norm is not None:
            self.embedding_norm.init_nested_module(index, module.embedding_norm)
        module.pos_embeddings.copy_(self.pos_embeddings[:, :self.trafo_sizes[index]])
        for source, target in zip(self.trafos, module.trafos):
            source.init_nested_module(index, target)
        self.final_norm.init_nested_module(index, module.final_norm)
        if hasattr(module, "predictions"):
            assert not hasattr(module, "prediction")
            for target, source in zip(module.predictions, self.predictions):
                target.weight.copy_(source.weight)
                if source.bias is not None:
                    target.bias.copy_(source.bias)
        else:
            module.prediction.weight.copy_(self.predictions[index].weight)
            if self.predictions[index].bias is not None:
                module.prediction.bias.copy_(self.predictions[index].bias)

    def empty_kv_cache(self, batch_size):
        return [(
            torch.empty(size=(batch_size, trafo.mhsa.num_heads, 0, trafo.mhsa.qk_sizes[-1])).cuda(),
            torch.empty(size=(batch_size, trafo.mhsa.num_heads, 0, trafo.mhsa.head_sizes[-1])).cuda(),
        ) for trafo in self.trafos]

    def forward(
        self,
        x,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        if self.requires_mask_tensor or kv_cache is not None:
            q_length = kv_length = x.size(1)
            if kv_cache is not None:
                kv_length += kv_cache[0][0].size(2)
                # kv_length = q_length + len(kv_cache[0][0])
            mask = torch.ones(q_length, kv_length, dtype=torch.bool, device=x.device)
            mask = mask.tril(diagonal=(kv_length - q_length))
        else:
            mask = True

        x = self.embedding(x)
        if self.embedding_norm is not None:
            x = self.embedding_norm(x)
        if not self.position_per_layer:
            x = self.fn_apply_pos(x)
        if self.input_dropout is not None:
            x = self.input_dropout(x)

        for n, trafo in enumerate(self.trafos):
            kwargs = dict(mask=mask)
            if self.position_per_layer:
                kwargs["fn_apply_pos"] = self.fn_apply_pos
            if kv_cache is not None:
                kwargs["kv_cache"] = kv_cache[n]

            if kv_cache is None:
                x = trafo(x, **kwargs)
            else:
                x, kv_cache[n] = trafo(x, **kwargs)

        x = self.final_norm(x)
        x = [prediction(x[..., :trafo_size]) for prediction, trafo_size in zip(self.predictions, self.trafo_sizes)]

        return x
