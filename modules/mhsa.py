import flash_attn
import math
import torch
from typing import Optional


class MHSA(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_heads: int,
        kv_groups: Optional[int],
        head_size: int,
        qk_size: Optional[int],
        torch_sdpa: bool,
        flash_sdpa: bool,
        bias: bool,
        dropout: float,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.kv_groups = self.num_heads if kv_groups is None else kv_groups
        assert self.num_heads % self.kv_groups == 0
        self.kv_repeats = self.num_heads // self.kv_groups
        self.head_size = head_size
        self.qk_size = self.head_size if qk_size is None else qk_size

        self.query_proj = torch.nn.Linear(input_size, self.num_heads * self.qk_size, bias=bias)
        self.key_proj = torch.nn.Linear(input_size, self.kv_groups * self.qk_size, bias=bias)
        self.value_proj = torch.nn.Linear(input_size, self.kv_groups * self.head_size, bias=bias)
        self.output_proj = torch.nn.Linear(self.num_heads * self.head_size, output_size, bias=bias)

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        assert not torch_sdpa or not flash_sdpa
        self.torch_sdpa = torch_sdpa
        self.flash_sdpa = flash_sdpa
        if not self.torch_sdpa and not self.flash_sdpa:
            self.isqrt_qk_size = 1.0 / math.sqrt(self.qk_size)

    def forward(self, x, mask=None, fn_apply_pos=None):
        batch_size, context_length, _ = x.size()

        query = self.query_proj(x) \
            .reshape(batch_size, context_length, self.num_heads, self.qk_size) \
            .transpose(-3, -2)
        key = self.key_proj(x) \
            .reshape(batch_size, context_length, self.kv_groups, self.qk_size) \
            .transpose(-3, -2)
        value = self.value_proj(x) \
            .reshape(batch_size, context_length, self.kv_groups, self.head_size) \
            .transpose(-3, -2)

        if self.kv_repeats > 1:
            key = key.unsqueeze(-3) \
                .expand(batch_size, self.kv_groups, self.kv_repeats, context_length, self.qk_size) \
                .reshape(batch_size, self.num_heads, context_length, self.qk_size)
            value = value.unsqueeze(-3) \
                .expand(batch_size, self.kv_groups, self.kv_repeats, context_length, self.head_size) \
                .reshape(batch_size, self.num_heads, context_length, self.head_size)

        if fn_apply_pos is not None:
            query = fn_apply_pos(query)
            key = fn_apply_pos(key)

        if self.torch_sdpa:
            is_causal = (mask is True)
            mask = (mask if isinstance(mask, torch.Tensor) else None)
            x = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=None,
            )
        elif self.flash_sdpa:
            is_causal = (mask is True)
            assert not isinstance(mask, torch.Tensor)
            x = flash_attn.flash_attn_func(
                query, key, value,
                dropout_p=0.0, softmax_scale=None, causal=is_causal, window_size=(-1, -1),
                alibi_slopes=None, deterministic=False, return_attn_probs=False,
            )
        else:
            attention_logits = (query @ key.transpose(-2, -1)) * self.isqrt_qk_size

            if self.temperature_proj is not None:
                temperature = self.temperature_proj(x)
                temperature = torch.nn.functional.softplus(temperature)
                min_temperature = 0.01
                temperature = (temperature + min_temperature) / (math.log(2.0) + min_temperature)
                temperature = 1.0 / temperature
                attention_logits *= temperature

            # self.log("attention-logits-mean", attention_logits[mask].mean())
            # self.log("attention-logits-max", attention_logits[mask].max())

            if mask is not None:
                assert not isinstance(mask, bool)
                attention_logits += torch.where(mask, 0.0, float("-inf"))

            attention = torch.nn.functional.softmax(attention_logits, dim=-1)
            x = attention @ value

        x = x.transpose(-3, -2) \
            .reshape(batch_size, context_length, self.num_heads * self.head_size)
        x = self.output_proj(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
