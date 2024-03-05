import torch

from modules.mrms_norm import MRMSNorm
from modules.rms_norm import RMSNorm


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
        for n, size in enumerate(sizes):
            if n == 0:
                rms_norm = RMSNorm(size)
            else:
                rms_norm = MRMSNorm(sizes[:n + 1])
            rms_norm.scale.copy_(mrms_norm.scale[:size])
            targets.append(rms_norm(x[..., :size]))
            for target in targets:
                assert torch.allclose(y[..., :target.size(-1)], target)
            print(f"  {size}: check")
