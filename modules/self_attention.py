import math
import torch
from typing import Optional


class SelfAttention(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        query_key_size: Optional[int] = None,
        torch_sdpa: bool = True,
    ):
        super().__init__()

        if query_key_size is None:
            query_key_size = output_size

        self.query_proj = torch.nn.Linear(input_size, query_key_size)
        self.key_proj = torch.nn.Linear(input_size, query_key_size)
        self.value_proj = torch.nn.Linear(input_size, output_size)

        self.torch_sdpa = torch_sdpa
        if not self.torch_sdpa:
            self.isqrt_query_key_size = 1.0 / math.sqrt(query_key_size)

        if False:
            assert not self.torch_sdpa
            self.temperature_proj = torch.nn.Linear(input_size, 1)
        else:
            self.temperature_proj = None

    def forward(self, x, mask=None):
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        if self.torch_sdpa:
            is_causal = (mask is True)
            mask = (mask if isinstance(mask, torch.Tensor) else None)
            x = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=None,
            )
            return x

        else:
            attention_logits = (query @ key.transpose(-2, -1)) * self.isqrt_query_key_size

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
            return x
