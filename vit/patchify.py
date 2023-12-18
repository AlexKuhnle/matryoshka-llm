import torch


class Patchify(torch.nn.Module):

    def __init__(
        self,
        patch_size: int,
    ):
        super().__init__()

        assert type(patch_size) == int and patch_size >= 1
        self.patch_size = patch_size

    def forward(self, x):
        padding = (-x.size(2) % self.patch_size, -x.size(3) % self.patch_size)
        x = torch.nn.functional.unfold(
            x,
            kernel_size=self.patch_size,
            dilation=1,
            padding=padding,
            stride=self.patch_size,
        )
        x = x.transpose(1, 2)
        return x
