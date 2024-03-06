import torch

from modules.position import init_position_scheme


if __name__ == "__main__":
    scheme = "rope"
    context_length = 83

    trafo_sizes = [8, 16, 32, 64]
    target_fn, target = init_position_scheme(
        scheme="rope", context_length=31, trafo_sizes=trafo_sizes,
    )
    test_fn, test = init_position_scheme(
        scheme="rope", context_length=31, trafo_size=trafo_sizes[0],
    )
    assert (test == target[..., :trafo_sizes[0] // 2]).all()
    for index in range(len(trafo_sizes) - 1):
        test_fn, test = init_position_scheme(
            scheme="rope", context_length=31, trafo_sizes=trafo_sizes[:index + 1],
        )
        assert (test == target[..., :trafo_sizes[index] // 2]).all()

    trafo_sizes = [6, 12]
    target_fn, target = init_position_scheme(
        scheme="rope", context_length=31, trafo_sizes=trafo_sizes,
    )
    test_fn, test = init_position_scheme(
        scheme="rope", context_length=31, trafo_size=trafo_sizes[0],
    )
    assert (test == target[..., :trafo_sizes[0] // 2]).all()
    for index in range(len(trafo_sizes) - 1):
        test_fn, test = init_position_scheme(
            scheme="rope", context_length=31, trafo_sizes=trafo_sizes[:index + 1],
        )
        assert torch.allclose(test, target[..., :trafo_sizes[index] // 2])
