import math
import torch


def init_position_scheme(scheme, context_length, trafo_size=None, trafo_sizes=None):
    assert trafo_size is None or trafo_sizes is None
    if trafo_sizes is not None:
        assert all(trafo_sizes[n] < trafo_sizes[n + 1] for n in range(len(trafo_sizes) - 1))
        trafo_size = trafo_sizes[-1]
    elif trafo_size is not None:
        trafo_sizes = [trafo_size]

    if scheme == "none":
        pos_embeddings = None

        def fn_apply_pos(x, start=0):
            return x

    elif scheme == "learned-add":
        initial_value = torch.randn(context_length, trafo_size) * 0.01
        pos_embeddings = torch.nn.Parameter(initial_value, requires_grad=True)

        def fn_apply_pos(x, start=0):
            assert x.size(-2) <= context_length
            return x + pos_embeddings[:x.size(-2), :x.size(-1)].expand(x.size())

    elif scheme == "learned-mult":
        initial_value = torch.randn(context_length, trafo_size) * 0.01
        pos_embeddings = torch.nn.Parameter(initial_value, requires_grad=True)

        def fn_apply_pos(x, start=0):
            assert x.size(-2) <= context_length
            return x * pos_embeddings[:x.size(-2), :x.size(-1)].expand(x.size())

    elif scheme == "rope":
        assert trafo_size % 2 == 0

        # https://github.com/facebookresearch/llama/blob/main/llama/model.py
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, trafo_size, 2)[: (trafo_size // 2)] / trafo_size))
        t = torch.arange(context_length)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        if len(trafo_sizes) > 1:
            assert trafo_sizes[0] % 2 == 0
            assert all(_is_power_of_two(trafo_sizes[n + 1] / trafo_sizes[n]) for n in range(len(trafo_sizes) - 1))
            reorder = _compute_rope_reorder_indices(trafo_sizes)
            freqs_cis = freqs_cis[:, reorder]
        pos_embeddings = torch.nn.Parameter(freqs_cis, requires_grad=False)

        def fn_apply_pos(x, start=0):
            *shape, context, size = x.size()
            assert start + context <= context_length and size in trafo_sizes
            x = torch.view_as_complex(x.reshape(*shape, context, size // 2, 2))
            x = x * pos_embeddings[start: start + context, :x.size(-1)].expand(x.size())
            return torch.view_as_real(x).reshape(*shape, context, size)

    else:
        raise NotImplementedError
        
    return fn_apply_pos, pos_embeddings


def _is_power_of_two(x):
    return math.log2(x) % 1.0 == 0.0


def _compute_rope_reorder_indices(trafo_sizes):
    reorder = torch.arange(0, trafo_sizes[-1] // 2, trafo_sizes[-1] // trafo_sizes[0])
    for size in trafo_sizes[1:]:
        indices = torch.arange(0, trafo_sizes[-1] // 2, trafo_sizes[-1] // size)
        indices = indices[torch.isin(indices, reorder, invert=True)]
        reorder = torch.cat([reorder, indices])
    assert (reorder.sort()[0] == torch.arange(trafo_sizes[-1] // 2)).all()
    return reorder
