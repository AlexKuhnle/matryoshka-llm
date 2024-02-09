import torch


def init_position_scheme(scheme, context_length, trafo_size):

    if scheme == "none":
        pos_embeddings = None

        def fn_apply_pos(x):
            return x

    elif scheme == "learned-add":
        initial_value = torch.randn(1, context_length, trafo_size) * 0.01
        pos_embeddings = torch.nn.Parameter(initial_value, requires_grad=True)

        def fn_apply_pos(x):
            assert x.size(1) <= context_length
            return x + pos_embeddings[:, :x.size(1)]

    elif scheme == "learned-mult":
        initial_value = torch.randn(1, context_length, trafo_size) * 0.01
        pos_embeddings = torch.nn.Parameter(initial_value, requires_grad=True)

        def fn_apply_pos(x):
            assert x.size(1) <= context_length
            return x * pos_embeddings[:, :x.size(1)]

    elif scheme == "rope":
        # https://github.com/facebookresearch/llama/blob/main/llama/model.py
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, trafo_size, 2)[: (trafo_size // 2)] / trafo_size))
        t = torch.arange(context_length)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        pos_embeddings = torch.nn.Parameter(freqs_cis.unsqueeze(0), requires_grad=False)

        def fn_apply_pos(x):
            batch_size, context, size = x.size()
            assert context <= context_length and size % 2 == 0
            x = torch.view_as_complex(x.reshape(batch_size, context, size // 2, 2))
            x = x * pos_embeddings[:, :context]
            return torch.view_as_real(x).reshape(batch_size, context, size)

    else:
        raise NotImplementedError
        
    return fn_apply_pos, pos_embeddings
