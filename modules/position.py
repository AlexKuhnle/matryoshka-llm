import torch


def init_position_scheme(scheme, context_length, trafo_size=None, trafo_sizes=None):
    assert trafo_size is None or trafo_sizes is None
    if trafo_sizes is not None:
        assert all(trafo_sizes[n] < trafo_sizes[n + 1] for n in range(len(trafo_sizes) - 1))
        trafo_size = trafo_sizes[-1]

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
        # https://github.com/facebookresearch/llama/blob/main/llama/model.py
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, trafo_size, 2)[: (trafo_size // 2)] / trafo_size))
        t = torch.arange(context_length)
        freqs = torch.outer(t, freqs)
        pos_embeddings = torch.polar(torch.ones_like(freqs), freqs)

        def fn_apply_pos(x, start=0):
            *shape, context, size = x.size()
            assert start + context <= context_length and size % 2 == 0
            x = torch.view_as_complex(x.reshape(*shape, context, size // 2, 2))
            x = x * pos_embeddings[start: start + context, :x.size(-1)].expand(x.size())
            return torch.view_as_real(x).reshape(*shape, context, size)

    else:
        raise NotImplementedError
        
    return fn_apply_pos, pos_embeddings
