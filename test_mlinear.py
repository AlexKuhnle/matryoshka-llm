import torch

from modules.mlinear import MLinear


if __name__ == "__main__":
    batch_size = 3
    context_length = 5
    input_sizes = [7, 9, 11]
    output_sizes = [13, 15, 17]
    bias = True

    mlinear = MLinear(input_sizes, output_sizes, bias=bias)
    x = torch.rand(size=((batch_size, context_length, input_sizes[-1])))
    y = mlinear(x)
    print("Matryoshka checks:")
    with torch.no_grad():
        targets = list()
        for index, (input_size, output_size) in enumerate(zip(input_sizes, output_sizes)):

            equivalent = MLinear.get_non_matryoshka_module()(input_size, output_size, bias=bias)
            mlinear.init_nested_module(index, equivalent)
            targets.append(equivalent(x[..., :input_size]))

            equivalent = MLinear(input_sizes[:index + 1], output_sizes[:index + 1], bias=bias)
            mlinear.init_nested_module(index, equivalent)
            targets.append(equivalent(x[..., :input_size]))

            for target in targets:
                assert torch.allclose(y[..., :target.size(-1)], target)

            print(f"  {input_size}/{output_size}: check")
