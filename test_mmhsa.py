import torch

from modules.mhsa import MHSA
from modules.mmhsa import MMHSA
from modules.position import init_position_scheme


if __name__ == "__main__":
    batch_size = 3
    context_length = 5
    # pos_scheme = "learned-add"
    pos_scheme = "rope"
    pos_context_length = 83
    # input_sizes = [7, 9, 11]
    # output_sizes = [13, 15, 17]
    # input_sizes = [6, 8, 10]
    # output_sizes = [12, 14, 16]
    input_sizes = [12, 24, 48]
    output_sizes = [8, 16, 32]
    num_heads = 4
    kv_groups = 2
    head_sizes = [14, 28, 56]
    qk_sizes = [10, 20, 40]
    torch_sdpa = True
    bias = False
    dropout = 0.0

    fn_apply_pos, _ = init_position_scheme(
        scheme=pos_scheme, context_length=pos_context_length, trafo_sizes=qk_sizes,
    )

    if torch_sdpa:
        mask = True
    else:
        mask = torch.ones(context_length, context_length, dtype=torch.bool)
        mask = mask.tril()

    mmhsa = MMHSA(
        input_sizes, output_sizes,
        num_heads=num_heads, kv_groups=kv_groups,
        head_sizes=head_sizes, qk_sizes=qk_sizes,
        torch_sdpa=torch_sdpa, bias=bias, dropout=dropout,
    )
    x = torch.rand(size=(batch_size, context_length, input_sizes[-1]))
    y = mmhsa(x, mask=mask, fn_apply_pos=fn_apply_pos)
    print("Matryoshka checks:")
    with torch.no_grad():
        targets = list()
        for index, (input_size, output_size, head_size, qk_size) in enumerate(zip(input_sizes, output_sizes, mmhsa.head_sizes, mmhsa.qk_sizes)):
            if index == 0:
                equivalent = MMHSA.get_non_matryoshka_module()(
                    input_size, output_size,
                    num_heads=num_heads, kv_groups=kv_groups,
                    head_size=head_size, qk_size=qk_size,
                    torch_sdpa=torch_sdpa, bias=bias, dropout=dropout,
                )
            else:
                equivalent = MMHSA(
                    input_sizes[:index + 1], output_sizes[:index + 1],
                    num_heads=num_heads, kv_groups=kv_groups,
                    head_sizes=head_sizes[:index + 1], qk_sizes=qk_sizes[:index + 1],
                    torch_sdpa=torch_sdpa, bias=bias, dropout=dropout,
                )
            mmhsa.init_nested_module(index, equivalent)
            targets.append(equivalent(x[..., :input_size], mask=mask, fn_apply_pos=fn_apply_pos))
            for target in targets:
                assert torch.allclose(y[..., :target.size(-1)], target)
            print(f"  {input_size}/{output_size}: check")
