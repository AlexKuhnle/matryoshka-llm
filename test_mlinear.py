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
        for n, (input_size, output_size) in enumerate(zip(input_sizes, output_sizes)):
            linear = torch.nn.Linear(input_size, output_size, bias=bias)
            linear.weight.copy_(mlinear.weight[:input_size, :output_size].transpose(0, 1))
            if bias:
                linear.bias.copy_(mlinear.bias[:output_size])
            targets.append(linear(x[..., :input_size]))
            if n > 0:
                linear = MLinear(input_sizes[:n + 1], output_sizes[:n + 1], bias=bias)
                for block1, block2 in zip(linear.weight_blocks, mlinear.weight_blocks):
                    block1.copy_(block2)
                if bias:
                    linear.bias.copy_(mlinear.bias[:output_size])
                targets.append(linear(x[..., :input_size]))
            for target in targets:
                assert torch.allclose(y[..., :target.size(-1)], target)

            print(f"  {input_size}/{output_size}: check")
