import math
import torch
from typing import Optional


class SelfAttention(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        query_key_size: Optional[int] = None,
    ):
        super().__init__()

        if query_key_size is None:
            query_key_size = output_size
        self.sqrt_query_key_size = math.sqrt(query_key_size)

        self.query_proj = torch.nn.Linear(input_size, query_key_size)
        self.key_proj = torch.nn.Linear(input_size, query_key_size)
        self.value_proj = torch.nn.Linear(input_size, output_size)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, mask=None):
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        attention_scores = query @ key.transpose(-2, -1)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
        attention = self.softmax(attention_scores / self.sqrt_query_key_size)
        x = attention @ value
        return x
