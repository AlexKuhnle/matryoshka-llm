import math
import torch
from typing import Callable, Optional, Sequence, Tuple, Union

from .mhsa import MHSA
from .mlinear import MLinear


class MMHSA(torch.nn.Module):

    @classmethod
    def get_non_matryoshka_module(cls):
        return MHSA

    def __init__(
        self,
        input_sizes: Sequence[int],
        output_sizes: Sequence[int],
        num_heads: int,
        kv_groups: Optional[int],
        head_sizes: Optional[Sequence[int]],
        qk_sizes: Optional[Sequence[int]],
        torch_sdpa: bool,
        bias: bool,
        dropout: float,
    ):
        super().__init__()

        self.input_sizes = list(input_sizes)
        self.output_sizes = list(output_sizes)
        self.num_heads = num_heads
        self.kv_groups = self.num_heads if kv_groups is None else kv_groups
        assert self.num_heads % self.kv_groups == 0
        self.kv_repeats = self.num_heads // self.kv_groups
        if head_sizes is None:
            self.head_sizes = [output_size // self.num_heads for output_size in self.output_sizes] 
        else:
            assert all(output_size % self.num_heads == 0 for output_size in self.output_sizes)
            self.head_sizes = list(head_sizes)
        self.qk_sizes = self.head_sizes if qk_sizes is None else list(qk_sizes)
        self.isqrt_qk_sizes = [1.0 / math.sqrt(qk_size) for qk_size in self.qk_sizes]

        self.query_proj = MLinear(
            self.input_sizes,
            [self.num_heads * qk_size for qk_size in self.qk_sizes],
            bias=bias,
        )
        self.key_proj = MLinear(
            self.input_sizes,
            [self.kv_groups * qk_size for qk_size in self.qk_sizes],
            bias=bias,
        )
        self.value_proj = MLinear(
            self.input_sizes,
            [self.kv_groups * head_size for head_size in self.head_sizes],
            bias=bias,
        )
        self.output_proj = MLinear(
            [self.num_heads * head_size for head_size in self.head_sizes],
            self.output_sizes,
            bias=bias,
        )

        if self.num_heads > 1:
            self.register_buffer(
                "query_reorder",
                MMHSA._compute_heads_reorder_indices(num_heads=self.num_heads, sizes=self.qk_sizes),
            )
            self.register_buffer(
                "output_reorder",
                MMHSA._compute_heads_reorder_indices(
                    num_heads=self.num_heads, sizes=self.head_sizes, reverse=True,
                ),
            )
        if self.kv_groups > 1:
            self.register_buffer(
                "key_reorder",
                MMHSA._compute_heads_reorder_indices(num_heads=self.kv_groups, sizes=self.qk_sizes),
            )
            self.register_buffer(
                "value_reorder",
                MMHSA._compute_heads_reorder_indices(num_heads=self.kv_groups, sizes=self.head_sizes),
            )

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.torch_sdpa = torch_sdpa

    @staticmethod
    def _compute_heads_reorder_indices(num_heads, sizes, reverse=False):
        heads_reorder = [list() for _ in range(num_heads)]
        offset = 0
        prev_size = 0
        for size in sizes:
            segment = size - prev_size
            indices = torch.arange(0, segment)
            for head in range(num_heads):
                heads_reorder[head].append(offset + indices)
                offset += segment
            prev_size = size
        assert offset == num_heads * sizes[-1]
        reorder = torch.cat([x for head in heads_reorder for x in head])
        if reverse:
            reverse = torch.empty(num_heads * sizes[-1], dtype=int)
            reverse[reorder.clone()] = torch.arange(num_heads * sizes[-1])
            reorder = reverse
        assert (reorder.sort()[0] == torch.arange(num_heads * sizes[-1])).all()
        return reorder

    def init_nested_module(self, index, module):
        self.query_proj.init_nested_module(index, module.query_proj)
        self.key_proj.init_nested_module(index, module.key_proj)
        self.value_proj.init_nested_module(index, module.value_proj)
        self.output_proj.init_nested_module(index, module.output_proj)

    def forward(
        self,
        x, 
        mask: Optional[Union[bool, torch.Tensor]] = None,
        fn_apply_pos: Optional[Callable[[torch.Tensor, Optional[int]], torch.Tensor]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        batch_size, q_length, _ = x.size()

        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        if self.num_heads > 1:
            query = query[..., self.query_reorder]
        if self.kv_groups > 1:
            key = key[..., self.key_reorder]
            value = value[..., self.value_reorder]

        query = query.reshape(batch_size, q_length, self.num_heads, self.qk_sizes[-1]) \
            .transpose(-3, -2)
        key = key.reshape(batch_size, q_length, self.kv_groups, self.qk_sizes[-1]) \
            .transpose(-3, -2)
        value = value.reshape(batch_size, q_length, self.kv_groups, self.head_sizes[-1]) \
            .transpose(-3, -2)

        if self.kv_repeats > 1:
            key = key.unsqueeze(-3) \
                .expand(batch_size, self.kv_groups, self.kv_repeats, q_length, self.qk_sizes[-1]) \
                .reshape(batch_size, self.num_heads, q_length, self.qk_sizes[-1])
            value = value.unsqueeze(-3) \
                .expand(batch_size, self.kv_groups, self.kv_repeats, q_length, self.head_sizes[-1]) \
                .reshape(batch_size, self.num_heads, q_length, self.head_sizes[-1])

        kv_length = q_length
        if kv_cache is not None:
            assert False
            kv_length += kv_cache[0].size(2)

        if fn_apply_pos is not None:
            query = fn_apply_pos(query, start=(kv_length - q_length))
            key = fn_apply_pos(key, start=(kv_length - q_length))

        if kv_cache is not None:
            assert False
            k_cache, v_cache = kv_cache
            key = torch.cat([k_cache, key], dim=2)
            value = torch.cat([v_cache, value], dim=2)
            kv_cache = (key, value)

        if self.torch_sdpa:
            xs = list()
            prev_head_size = 0
            is_causal = (mask is True)
            mask = (mask if isinstance(mask, torch.Tensor) else None)
            for qk_size, head_size in zip(self.qk_sizes, self.head_sizes):
                xs.append(torch.nn.functional.scaled_dot_product_attention(
                    query[..., :qk_size], key[..., :qk_size], value[..., :head_size],
                    attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=None,
                )[..., prev_head_size: head_size])
                prev_head_size = head_size
            x = torch.cat(xs, dim=-1)

        else:
            dp_blocks = list()
            prev_qk_size = 0
            prev_head_size = 0
            for qk_size, head_size, isqrt_qk_size in zip(self.qk_sizes, self.head_sizes, self.isqrt_qk_sizes):
                dp_blocks.append(query[..., prev_qk_size: qk_size] @ key[..., prev_qk_size: qk_size].transpose(-2, -1))

            attention_logits = list()
            prev_dp_block = None
            for dp_block, isqrt_qk_size in zip(dp_blocks, self.isqrt_qk_sizes):
                if prev_dp_block is not None:
                    dp_block += prev_dp_block
                attention_logits.append(dp_block * isqrt_qk_size)

            if mask is not None:
                assert not isinstance(mask, bool)
                mask = torch.where(mask, 0.0, float("-inf"))
                attention_logits = [logits + mask for logits in attention_logits]

            xs = list()
            prev_head_size = 0
            for logits, head_size in zip(attention_logits, self.head_sizes):
                attention = torch.nn.functional.softmax(logits, dim=-1)
                xs.append(attention @ value[..., prev_head_size: head_size])
                prev_head_size = head_size
            x = torch.cat(xs, dim=-1)

        x = x.transpose(-3, -2) \
            .reshape(batch_size, q_length, self.num_heads * self.head_sizes[-1])

        if self.num_heads > 1:
            x = x[..., self.output_reorder]

        x = self.output_proj(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if kv_cache is None:
            return x
        else:
            return x, kv_cache
