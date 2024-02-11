from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from beartype import beartype
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, einsum, nn
from zeta.nn import SwiGLU, threed_to_text

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


# sinusoidal positions


def posemb_sincos_1d(
    seq, dim, temperature=10000, device=None, dtype=torch.float32
):
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)


# helper classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond_fn=None):
        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)


# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.context_norm = (
            nn.LayerNorm(context_dim)
            if norm_context
            else nn.Identity()
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = (
            nn.Sequential(
                nn.Linear(dim, ff_inner_dim * 2, bias=False),
                SwiGLU(),
                nn.Linear(ff_inner_dim, dim, bias=False),
            )
            if parallel_ff
            else None
        )

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge and combine heads

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out


# MBConv


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = (
            torch.FloatTensor(
                (x.shape[0], 1, 1, 1), device=device
            ).uniform_()
            > self.prob
        )
        return x * keep_mask / (1 - self.prob)


def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate=4,
    shrinkage_rate=0.25,
    dropout=0.0,
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
        ),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out),
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.0,
        window_size=7,
        num_mem_kv=4,
    ):
        super().__init__()
        assert (
            dim % dim_head
        ) == 0, "dimension should be divisible by dimension per head"

        self.norm = LayerNorm(dim)

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.mem_kv = nn.Parameter(
            torch.randn(2, self.heads, num_mem_kv, dim_head)
        )

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1), nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding(
            (2 * window_size - 1) ** 2, self.heads
        )

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(
            grid, "j ... -> 1 j ..."
        )
        rel_pos += window_size - 1
        rel_pos_indices = (
            rel_pos * torch.tensor([2 * window_size - 1, 1])
        ).sum(dim=-1)

        self.register_buffer(
            "rel_pos_indices", rel_pos_indices, persistent=False
        )

    def forward(self, x):
        (
            batch,
            height,
            width,
            window_height,
            window_width,
            _,
            device,
            h,
        ) = (
            *x.shape,
            x.device,
            self.heads,
        )

        x = self.norm(x)

        # flatten

        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            (q, k, v),
        )

        # scale

        q = q * self.scale

        # null / memory / register kv

        mk, mv = map(
            lambda t: repeat(t, "h n d -> b h n d", b=q.shape[0]),
            self.mem_kv,
        )
        num_mem = mk.shape[-2]

        k = torch.cat((mk, k), dim=-2)
        v = torch.cat((mv, v), dim=-2)

        # sim

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)

        bias = F.pad(bias, (0, 0, num_mem, 0), value=0.0)

        sim = sim + rearrange(bias, "i j h -> h i j")

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(
            out,
            "b h (w1 w2) d -> b w1 w2 (h d)",
            w1=window_height,
            w2=window_width,
        )

        # combine heads out

        out = self.to_out(out)
        return rearrange(
            out, "(b x y) ... -> b x y ...", x=height, y=width
        )


class MaxViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head=32,
        dim_conv_stem=None,
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
        channels=3,
    ):
        super().__init__()
        assert isinstance(depth, tuple), (
            "depth needs to be tuple if integers indicating number of"
            " transformer blocks at that stage"
        )

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(
                channels, dim_conv_stem, 3, stride=2, padding=1
            ),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding=1),
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2**i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        cond_hidden_dims = []

        for ind, (
            (layer_dim_in, layer_dim),
            layer_depth,
        ) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                cond_hidden_dims.append(stage_dim_in)

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample=is_first,
                        expansion_rate=mbconv_expansion_rate,
                        shrinkage_rate=mbconv_shrinkage_rate,
                    ),
                    Rearrange(
                        "b d (x w1) (y w2) -> b x y w1 w2 d",
                        w1=w,
                        w2=w,
                    ),  # block-like attention
                    Residual(
                        Attention(
                            dim=layer_dim,
                            dim_head=dim_head,
                            dropout=dropout,
                            window_size=w,
                        )
                    ),
                    Residual(
                        FeedForward(dim=layer_dim, dropout=dropout)
                    ),
                    Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
                    Rearrange(
                        "b d (w1 x) (w2 y) -> b x y w1 w2 d",
                        w1=w,
                        w2=w,
                    ),  # grid-like attention
                    Residual(
                        Attention(
                            dim=layer_dim,
                            dim_head=dim_head,
                            dropout=dropout,
                            window_size=w,
                        )
                    ),
                    Residual(
                        FeedForward(dim=layer_dim, dropout=dropout)
                    ),
                    Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
                )

                self.layers.append(block)

        embed_dim = dims[-1]
        self.embed_dim = dims[-1]

        self.cond_hidden_dims = cond_hidden_dims

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce("b d h w -> b d", "mean"),
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    @beartype
    def forward(
        self,
        x,
        texts: Optional[List[str]] = None,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        cond_drop_prob=0.0,
        return_embeddings=False,
    ):
        x = self.conv_stem(x)

        cond_fns = iter(default(cond_fns, []))

        for stage in self.layers:
            cond_fn = next(cond_fns, None)

            if exists(cond_fn):
                x = cond_fn(x)

            x = stage(x)

        if return_embeddings:
            return x

        return self.mlp_head(x)


class MCViT(nn.Module):
    """
    Multi-Chunk Vision Transformer (MCViT) model.

    Args:
        dim (int): Dimension of the input features.
        dim_head (int): Dimension of each attention head.
        dropout (float): Dropout rate.
        chunks (int): Number of chunks to divide the input into.
        depth (int): Number of transformer layers.
        cross_attn_heads (int): Number of attention heads for cross attention.

    Examples:
    >>> import torch
    >>> from mcvit.model import MCViT
    >>>
    >>> # Initialize the MCViT model
    >>> mcvit = MCViT(
    ...     dim=512,
    ...     attn_seq_len=256,
    ...     dim_head=64,
    ...     dropout=0.1,
    ...     chunks=16,
    ...     depth=12,
    ...     cross_attn_heads=8,
    ... )
    >>>
    >>> # Create a random tensor to represent a video
    >>> x = torch.randn(
    ...     1, 3, 256, 256, 256
    ... )  # (batch, channels, frames, height, width)
    >>>
    >>> # Pass the tensor through the model
    >>> output = mcvit(x)
    >>>
    >>> print(
    ...     output.shape
    ... )  # Outputs the shape of the tensor after passing through the model


    """

    def __init__(
        self,
        dim: int,
        attn_seq_len: int,
        dim_head: int,
        dropout: float,
        chunks: int,
        depth: int,
        cross_attn_heads: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.attn_seq_len = attn_seq_len
        self.dim_head = dim_head
        self.dropout = dropout
        self.chunks = chunks
        self.depth = depth
        self.cross_attn_heads = cross_attn_heads

        self.video_proj = nn.Linear(dim, dim)

        self.norm = LayerNorm(dim)

        # Attn
        self.attn = Attention(
            dim,
            dim_head,
            dropout,
        )

        # Cross Attention
        self.cross_attn = CrossAttention(
            dim,
            context_dim=dim,
            dim_head=dim_head,
            heads=cross_attn_heads,
            *args,
            **kwargs,
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        # Memory consolidation
        self.memory_consolidation = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MCViT model.

        Args:
            x (Tensor): Input tensor of shape (B, T, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, T, C, H, W).

        """
        B, T, C, H, W = x.shape
        # embedding = self.video_proj(x)
        # print(embedding.shape)

        # Chunk the video into chunks using einops
        chunked_video = rearrange(x, "b t c h w -> (b t) c h w")
        print(chunked_video.shape)

        # Memory = None
        memory = torch.zeros(1, 1, 1, 1, 1)

        # Empty zs list
        zs = []

        # Loop through the chunks
        for z in chunked_video:
            z = self.norm(z)
            print(z.shape)
            for _ in range(self.depth):
                if memory is not None:
                    y = threed_to_text(
                        z, self.attn_seq_len, self.dim, True
                    )
                    print(y.shape)
                    y, _ = self.attn(y) + y
                else:
                    # Concat the norm and memory
                    kv = torch.cat([z, memory], dim=1)
                    y = threed_to_text(
                        y, self.attn_seq_len, self.dim, True
                    )
                    print(y.shape)
                    y = self.cross_attn(y, kv)

                # norm
                y_norm = self.norm(y)

                # MLP
                z = self.mlp(y_norm) + y

                # Memory consolidation that takes in memory, z, num_mem, and mc_method
                # memory = memory_consolidation(memory, z, num_mem, mc_method)

                # Normalize the memory
                memory = self.norm(memory)

                # Append the zs list
                zs.append(z)

        return zs
        # # Concatenate the zs list
        # zs = torch.cat(zs, dim=1)

        # return zs
