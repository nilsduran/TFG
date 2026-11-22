from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


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
    ) -> LoRALinear:
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
