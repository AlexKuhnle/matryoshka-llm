import torch

from modules.mrms_norm import MRMSNorm


if __name__ == "__main__":
    batch_size = 3
    context_length = 5
    sizes = [7, 9, 11]

    mrms_norm = MRMSNorm(sizes)
    x = torch.rand(size=((batch_size, context_length, sizes[-1])))
    y = mrms_norm(x)
    print("Matryoshka checks:")
    with torch.no_grad():
        targets = list()
        for index, size in enumerate(sizes):
            if index == 0:
                equivalent = MRMSNorm.get_non_matryoshka_module()(size)
            else:
                equivalent = MRMSNorm(sizes[:index + 1])
            mrms_norm.init_nested_module(index, equivalent)
            targets.append(equivalent(x[..., :size]))
            for target in targets:
                assert torch.allclose(y[..., :target.size(-1)], target)
            print(f"  {size}: check")
