from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """Compact EEGNet-style binary classifier.

    Input shape: (batch, channels, times).
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        padding = kernel_length // 2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, padding), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout2d(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout2d(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_times)
            n_features = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(n_features, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._forward_features(x))


class ChebGraphConv(nn.Module):
    """Chebyshev graph convolution using a precomputed normalized Laplacian."""

    def __init__(self, in_features: int, out_features: int, K: int = 3) -> None:
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.empty(K, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        # x: (batch*time, nodes, features)
        supports = [x]
        if self.K > 1:
            supports.append(torch.einsum("ij,bjf->bif", laplacian, x))
        for _ in range(2, self.K):
            supports.append(2 * torch.einsum("ij,bjf->bif", laplacian, supports[-1]) - supports[-2])
        out = 0
        for k, support in enumerate(supports):
            out = out + torch.einsum("bnf,fo->bno", support, self.weight[k])
        return out + self.bias


def scaled_laplacian_from_adjacency(adj: torch.Tensor) -> torch.Tensor:
    degree = torch.sum(adj, dim=1)
    inv_sqrt = torch.pow(torch.clamp(degree, min=1e-6), -0.5)
    norm_adj = inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    lap = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype) - norm_adj
    return lap - torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)


class STGNNChebConv(nn.Module):
    """ChebConv spatial block followed by temporal 1D CNN and global node pooling.

    Input shape: (batch, windows, nodes, node_features).
    """

    def __init__(
        self,
        n_nodes: int,
        in_features: int,
        adjacency,
        n_classes: int = 2,
        cheb_hidden: int = 32,
        temporal_hidden: int = 64,
        K: int = 3,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        adj = torch.as_tensor(adjacency, dtype=torch.float32)
        self.register_buffer("laplacian", scaled_laplacian_from_adjacency(adj))
        self.cheb = ChebGraphConv(in_features, cheb_hidden, K=K)
        self.bn_graph = nn.BatchNorm1d(n_nodes * cheb_hidden)
        self.temporal = nn.Sequential(
            nn.Conv1d(cheb_hidden, temporal_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(temporal_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(temporal_hidden, temporal_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(temporal_hidden),
            nn.ELU(),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(temporal_hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, windows, nodes, features = x.shape
        x_flat = x.reshape(batch * windows, nodes, features)
        x_flat = F.elu(self.cheb(x_flat, self.laplacian))
        x_flat = self.bn_graph(x_flat.reshape(batch * windows, -1)).reshape(batch, windows, nodes, -1)
        pooled_nodes = x_flat.mean(dim=2)
        temporal_in = pooled_nodes.transpose(1, 2)
        temporal_out = self.temporal(temporal_in)
        graph_embedding = temporal_out.mean(dim=-1)
        return self.classifier(graph_embedding)


class LoRALinear(nn.Module):
    """Linear layer with optional low-rank LoRA update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        r: int = 0,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.r = int(r)
        self.scaling = alpha / r if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout)
        if self.r > 0:
            self.lora_a = nn.Linear(in_features, r, bias=False)
            self.lora_b = nn.Linear(r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        r: int = 0,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
        )
        layer.base.load_state_dict(linear.state_dict())
        return layer

    @property
    def weight(self) -> torch.Tensor:
        try:
            if self.r > 0:
                # Compute effective weight matrix dynamically
                return self.base.weight + (self.lora_b.weight @ self.lora_a.weight) * self.scaling
            return self.base.weight
        except Exception as e:
            import traceback
            print("--- DEBUG: Exception in LoRALinear.weight getter ---")
            traceback.print_exc()
            print("-----------------------------------------------------")
            raise e

    @property
    def bias(self) -> torch.Tensor | None:
        try:
            return self.base.bias
        except Exception as e:
            import traceback
            print("--- DEBUG: Exception in LoRALinear.bias getter ---")
            traceback.print_exc()
            print("---------------------------------------------------")
            raise e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            out = out + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return out


class LoRAMultiheadSelfAttention(nn.Module):
    """Batch-first self-attention with LoRA on q and v projections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = LoRALinear(embed_dim, embed_dim, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = LoRALinear(embed_dim, embed_dim, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        q = self.q_proj(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        out = out.transpose(1, 2).reshape(batch, tokens, self.embed_dim)
        return self.out_proj(out)


class LoRATransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LoRAMultiheadSelfAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def freeze_non_lora_parameters(
    model: nn.Module,
    *,
    train_classifier: bool = True,
    train_tokenizer: bool = True,
) -> None:
    """Freeze base weights and leave LoRA adapters plus optional classifier and tokenizer trainable."""

    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        is_head = train_classifier and ("classifier" in name or "head" in name or "final" in name)
        is_tok = train_tokenizer and ("patch_embed" in name or "tokenizer" in name)
        param.requires_grad = is_lora or is_head or is_tok


def inject_lora_into_linears(
    model: nn.Module,
    *,
    include_substrings: tuple[str, ...] = ("attn",),
    exclude_substrings: tuple[str, ...] = ("head", "classifier", "decoder", "patch_embed", "tokenizer"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> nn.Module:
    """Replace matching nn.Linear modules with LoRA-augmented versions.

    The replacement is done recursively in-place. The default target set focuses
    on attention blocks while leaving classifier heads and patch/tokenizer modules
    intact.
    """

    def should_replace(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if include_substrings and not any(s in path for s in include_substrings):
            return False
        if any(s in path for s in exclude_substrings):
            return False
        return True

    def recurse(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if should_replace(path, child):
                setattr(parent, name, LoRALinear.from_linear(child, r=r, alpha=alpha, dropout=dropout))
            else:
                recurse(child, path)

    recurse(model)
    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


class STEEGFormerLoRA(nn.Module):
    """Spatio-temporal EEG Transformer with LoRA adapters.

    Input shape: (batch, channels, times). The model creates temporal patches per
    channel, adds channel/time embeddings, then applies transformer blocks.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        patch_size: int = 32,
        patch_stride: int = 16,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_chans = n_channels
        self.n_times = n_times
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_stride)
        n_patches = 1 + max(0, (n_times - patch_size) // patch_stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.channel_embed = nn.Parameter(torch.zeros(1, n_channels, 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, 1, n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                LoRATransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        if freeze_base:
            freeze_non_lora_parameters(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, times = x.shape
        x = x.reshape(batch * channels, 1, times)
        tokens = self.patch_embed(x).transpose(1, 2)
        n_patches = tokens.shape[1]
        tokens = tokens.reshape(batch, channels, n_patches, -1)
        tokens = tokens + self.channel_embed[:, :channels, :, :] + self.time_embed[:, :, :n_patches, :]
        tokens = tokens.reshape(batch, channels * n_patches, -1)
        cls = self.cls_token.expand(batch, -1, -1)
        x = self.pos_drop(torch.cat([cls, tokens], dim=1))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        return self.classifier(x)


class LaBraMPPLoRA(nn.Module):
    """LaBraM++-style channel-aware EEG patch transformer with LoRA.

    This implementation provides the downstream adaptation shape: neural patch
    tokenizer, channel/time embeddings, transformer encoder, and binary head.
    Pretrained LaBraM++ weights can later be loaded into matching components.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        patch_size: int = 64,
        patch_stride: int = 32,
        embed_dim: int = 200,
        depth: int = 6,
        num_heads: int = 5,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_chans = n_channels
        self.n_times = n_times
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.tokenizer = nn.Sequential(
            nn.Conv1d(1, embed_dim // 2, kernel_size=patch_size // 2, stride=max(1, patch_size // 4), padding=patch_size // 4),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_times)
            n_patches = self.tokenizer(dummy).shape[-1]
        self.summary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.channel_embed = nn.Parameter(torch.zeros(1, n_channels, 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, 1, n_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                LoRATransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, n_classes))
        nn.init.trunc_normal_(self.summary_token, std=0.02)
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        if freeze_base:
            freeze_non_lora_parameters(self)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, times = x.shape
        x = x.reshape(batch * channels, 1, times)
        tokens = self.tokenizer(x).transpose(1, 2)
        n_patches = tokens.shape[1]
        tokens = tokens.reshape(batch, channels, n_patches, -1)
        tokens = tokens + self.channel_embed[:, :channels, :, :] + self.time_embed[:, :, :n_patches, :]
        tokens = tokens.reshape(batch, channels * n_patches, -1)
        summary = self.summary_token.expand(batch, -1, -1)
        x = torch.cat([summary, tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
