# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp

from ..modules.model import sinusoidal_embedding_1d
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs, offset=0):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][offset : f + offset].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[
            (sp_rank * s_per_rank) : ((sp_rank + 1) * s_per_rank), :, :
        ]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    train_mode=False,
    y=None,
    reference_latent=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == "i2v":
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    DOES_PATCH_EMBEDDING_ANCHOR_EXIST = False
    if hasattr(self, "patch_embedding_anchor"):
        DOES_PATCH_EMBEDDING_ANCHOR_EXIST = True

    PATCH_EMBEDDING_REFERENCE_EXIST = False
    if hasattr(self, "patch_embedding_reference"):
        PATCH_EMBEDDING_REFERENCE_EXIST = True

    if y is not None and not DOES_PATCH_EMBEDDING_ANCHOR_EXIST:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    if DOES_PATCH_EMBEDDING_ANCHOR_EXIST:
        y_anchor = [self.patch_embedding_anchor(u.unsqueeze(0)) for u in y]
        x = [u + v for u, v in zip(x, y_anchor)]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        if reference_latent is None:
            t = t.expand(t.size(0), seq_len)
        else:
            t = t.expand(t.size(0), seq_len * 2)
    elif t.dim() == 2 and reference_latent is not None:
        t = torch.cat([t, t], dim=1)
    else:
        raise ValueError(f"t.dim() is {t.dim()}")

    with torch.amp.autocast("cuda", dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        if reference_latent is None:
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t)
                .unflatten(0, (bt, seq_len))
                .float()
            )
        else:
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t)
                .unflatten(0, (bt, seq_len * 2))
                .float()
            )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )

    # reference latent processing
    vref = None
    if reference_latent is not None:
        if PATCH_EMBEDDING_REFERENCE_EXIST:
            vref = [torch.cat([v[4:]], dim=0) for v in reference_latent]
            vref = [self.patch_embedding_reference(u.unsqueeze(0)) for u in vref]
        else:
            vref = [torch.cat([v[4:], v], dim=0) for v in reference_latent]
            vref = [self.patch_embedding(u.unsqueeze(0)) for u in vref]

        vref = [u.flatten(2).transpose(1, 2) for u in vref]
        seq_lens = torch.tensor([u.size(1) for u in vref], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        vref = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in vref
        ])
        # 2x the seq_len to account for the temporal dimension
        seq_lens = seq_lens * 2
        grid_sizes[:, 0] = grid_sizes[:, 0] * 2

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    if reference_latent is not None:
        vref = torch.chunk(vref, get_world_size(), dim=1)[get_rank()]
        x = torch.cat([x, vref], dim=1)
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        add_ref_video=True if reference_latent is not None else False,
    )

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    if not train_mode:
        if reference_latent is not None:
            x = x.chunk(2, dim=1)[0]
            x = gather_forward(x, dim=1).contiguous()
            tmp_grid_sizes = grid_sizes.clone()
            tmp_grid_sizes[:, 0] = tmp_grid_sizes[:, 0] // 2
            x = self.unpatchify(x, tmp_grid_sizes)
        else:
            x = gather_forward(x, dim=1).contiguous()
            x = self.unpatchify(x, grid_sizes)
    else:
        sharded_grid_sizes = grid_sizes.clone()
        sharded_grid_sizes[:, 0] = sharded_grid_sizes[:, 0] // get_world_size()
        x = self.unpatchify(x, sharded_grid_sizes)
        # Convert list of tensors into batch dimension
        x = torch.stack(x, dim=0)
        if reference_latent is not None:
            x = x.chunk(2, dim=2)
            x = x[0]
        return x

    return [u.float() for u in x]


def sp_attn_forward(
    self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, add_ref_video=False
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    if add_ref_video:
        grid_sizes = grid_sizes.clone()
        grid_sizes[:, 0] = grid_sizes[:, 0] // 2
        q_in, q_ref = q.chunk(2, dim=1)
        k_in, k_ref = k.chunk(2, dim=1)

        q_in = rope_apply(q_in, grid_sizes, freqs, offset=0)
        k_in = rope_apply(k_in, grid_sizes, freqs, offset=0)
        q_ref = rope_apply(q_ref, grid_sizes, freqs, offset=50)
        k_ref = rope_apply(k_ref, grid_sizes, freqs, offset=50)

        q = torch.cat([q_in, q_ref], dim=1)
        k = torch.cat([k_in, k_ref], dim=1)
    else:
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


def sp_dit_forward_5b(
    self,
    x,
    t,
    context,
    seq_len,
    train_mode=False,
    y=None,
    reference_latent=None,
    reference_latent_2=None,
    return_reference_latent=False,
):
    """
    x:              A list of videos each with shape [C, T, H, W]. This is the noisy video.
    t:              [B]. This is the timesteps.
    context:        A list of text embeddings each with shape [L, C]. This is the text embedding.
    seq_len:        The sequence length of the video.
    train_mode:     Whether to train the model.
    y:              Whatever needs to be added to the noisy channels.
    reference_latent: Whatever needs to be patchified into additional tokens along the frame dimension.
    reference_latent_2: Whatever needs to be patchified into additional tokens along the frame dimension.
    return_reference_latent: Whether to return the reference latent.
    """
    if self.model_type == "i2v":
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    DOES_PATCH_EMBEDDING_ANCHOR_EXIST = False
    if hasattr(self, "patch_embedding_anchor"):
        DOES_PATCH_EMBEDDING_ANCHOR_EXIST = True

    PATCH_EMBEDDING_REFERENCE_EXIST = False
    if hasattr(self, "patch_embedding_reference"):
        PATCH_EMBEDDING_REFERENCE_EXIST = True

    PATCH_EMBEDDING_REFERENCE_2_EXIST = False
    if hasattr(self, "patch_embedding_reference_2"):
        PATCH_EMBEDDING_REFERENCE_2_EXIST = True

    if y is not None and not DOES_PATCH_EMBEDDING_ANCHOR_EXIST:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    if DOES_PATCH_EMBEDDING_ANCHOR_EXIST:
        y_anchor = [self.patch_embedding_anchor(u.unsqueeze(0)) for u in y]
        x = [u + v for u, v in zip(x, y_anchor)]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        if reference_latent is None:
            t = t.expand(t.size(0), seq_len)
        else:
            if reference_latent_2 is None:
                t = t.expand(t.size(0), seq_len * 2)
            else:
                t = t.expand(t.size(0), seq_len * 3)
    elif t.dim() == 2 and reference_latent is not None:
        if reference_latent_2 is None:
            t = torch.cat([t, t], dim=1)
        else:
            t = torch.cat([t, t, t], dim=1)
    else:
        raise ValueError(f"t.dim() is {t.dim()}")

    with torch.amp.autocast("cuda", dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        if reference_latent is None:
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t)
                .unflatten(0, (bt, seq_len))
                .float()
            )
        else:
            if reference_latent_2 is None:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t)
                    .unflatten(0, (bt, seq_len * 2))
                    .float()
                )
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t)
                    .unflatten(0, (bt, seq_len * 3))
                    .float()
                )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )

    # reference latent processing
    vref = None
    vref_2 = None
    SEQ_LEN_MULTIPLICATIVE_FACTOR = 1
    if reference_latent is not None:
        if PATCH_EMBEDDING_REFERENCE_EXIST:
            vref = [torch.cat([v[4:]], dim=0) for v in reference_latent]
            vref = [self.patch_embedding_reference(u.unsqueeze(0)) for u in vref]
        else:
            raise ValueError(
                "A reference_latent is provided, but no patch_embedding_reference parameter exists. Since the 5b model does not have extra patch embedding, this should not happen."
            )

        vref = [u.flatten(2).transpose(1, 2) for u in vref]
        seq_lens = torch.tensor([u.size(1) for u in vref], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        vref = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in vref
        ])
        SEQ_LEN_MULTIPLICATIVE_FACTOR = 2

    if reference_latent_2 is not None:
        assert reference_latent is not None, (
            "reference_latent_2 is provided, but reference_latent is not provided"
        )
        if PATCH_EMBEDDING_REFERENCE_2_EXIST:
            vref_2 = [torch.cat([v[4:]], dim=0) for v in reference_latent_2]
            vref_2 = [self.patch_embedding_reference_2(u.unsqueeze(0)) for u in vref_2]
        else:
            raise ValueError(
                "A reference_latent_2 is provided, but no patch_embedding_reference_2 param exists. Since the 5b model does not have extra patch embedding, this should not happen."
            )
        vref_2 = [u.flatten(2).transpose(1, 2) for u in vref_2]
        seq_lens = torch.tensor([u.size(1) for u in vref_2], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        vref_2 = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in vref_2
        ])
        # 3x the seq_len to account for the temporal dimension - we divide by 2 for unitary method - we'd already doubled
        SEQ_LEN_MULTIPLICATIVE_FACTOR = 3

    seq_lens = seq_lens * SEQ_LEN_MULTIPLICATIVE_FACTOR
    grid_sizes[:, 0] = grid_sizes[:, 0] * SEQ_LEN_MULTIPLICATIVE_FACTOR

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    if reference_latent is not None:
        vref = torch.chunk(vref, get_world_size(), dim=1)[get_rank()]
        x = torch.cat([x, vref], dim=1)
    if reference_latent_2 is not None:
        vref_2 = torch.chunk(vref_2, get_world_size(), dim=1)[get_rank()]
        x = torch.cat([x, vref_2], dim=1)
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        add_ref_video=True if reference_latent is not None else False,
        add_ref_video_2=True if reference_latent_2 is not None else False,
    )

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    if not train_mode:
        if reference_latent is not None and reference_latent_2 is None:
            x = x.chunk(SEQ_LEN_MULTIPLICATIVE_FACTOR, dim=1)[0]
            x = gather_forward(x, dim=1).contiguous()
            tmp_grid_sizes = grid_sizes.clone()
            tmp_grid_sizes[:, 0] = tmp_grid_sizes[:, 0] // SEQ_LEN_MULTIPLICATIVE_FACTOR
            x = self.unpatchify(x, tmp_grid_sizes)
        elif reference_latent is not None and reference_latent_2 is not None:
            x = x.chunk(SEQ_LEN_MULTIPLICATIVE_FACTOR, dim=1)[0]
            x = gather_forward(x, dim=1).contiguous()
            tmp_grid_sizes = grid_sizes.clone()
            tmp_grid_sizes[:, 0] = tmp_grid_sizes[:, 0] // SEQ_LEN_MULTIPLICATIVE_FACTOR
            x = self.unpatchify(x, tmp_grid_sizes)
        else:
            x = gather_forward(x, dim=1).contiguous()
            x = self.unpatchify(x, grid_sizes)
    else:
        sharded_grid_sizes = grid_sizes.clone()
        sharded_grid_sizes[:, 0] = sharded_grid_sizes[:, 0] // get_world_size()
        x = self.unpatchify(x, sharded_grid_sizes)
        # Convert list of tensors into batch dimension
        x = torch.stack(x, dim=0)
        if reference_latent is not None and reference_latent_2 is None:
            x = x.chunk(2, dim=2)
            if return_reference_latent:
                return x[0], x[-1]
            x = x[0]
        if reference_latent is not None and reference_latent_2 is not None:
            x = x.chunk(3, dim=2)
            if return_reference_latent:
                return x[0], x[-1]
            x = x[0]
        return x

    return [u.float() for u in x]


def sp_attn_forward_5b(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    dtype=torch.bfloat16,
    add_ref_video=False,
    add_ref_video_2=False,
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    if add_ref_video and not add_ref_video_2:
        grid_sizes = grid_sizes.clone()
        grid_sizes[:, 0] = grid_sizes[:, 0] // 2
        q_in, q_ref = q.chunk(2, dim=1)
        k_in, k_ref = k.chunk(2, dim=1)

        q_in = rope_apply(q_in, grid_sizes, freqs, offset=0)
        k_in = rope_apply(k_in, grid_sizes, freqs, offset=0)
        q_ref = rope_apply(q_ref, grid_sizes, freqs, offset=50)
        k_ref = rope_apply(k_ref, grid_sizes, freqs, offset=50)

        q = torch.cat([q_in, q_ref], dim=1)
        k = torch.cat([k_in, k_ref], dim=1)
    elif add_ref_video_2 and add_ref_video:
        grid_sizes = grid_sizes.clone()
        grid_sizes[:, 0] = grid_sizes[:, 0] // 3
        q_in, q_ref, q_ref_2 = q.chunk(3, dim=1)
        k_in, k_ref, k_ref_2 = k.chunk(3, dim=1)

        q_in = rope_apply(q_in, grid_sizes, freqs, offset=0)
        k_in = rope_apply(k_in, grid_sizes, freqs, offset=0)
        q_ref = rope_apply(q_ref, grid_sizes, freqs, offset=50)
        k_ref = rope_apply(k_ref, grid_sizes, freqs, offset=50)
        q_ref_2 = rope_apply(q_ref_2, grid_sizes, freqs, offset=100)
        k_ref_2 = rope_apply(k_ref_2, grid_sizes, freqs, offset=100)

        q = torch.cat([q_in, q_ref, q_ref_2], dim=1)
        k = torch.cat([k_in, k_ref, k_ref_2], dim=1)
    else:
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
