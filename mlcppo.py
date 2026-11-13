"""Lightweight multi-layer value critic for autoregressive language models."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def resolve_layers(spec: str, num_hidden_layers: int) -> list[int]:
    """Resolve comma-separated fractions (0.5, 0.75, 1.0) or layer indices."""
    result: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 1.0:
            layer = round(value * num_hidden_layers)
        else:
            layer = int(value)
        if not 1 <= layer <= num_hidden_layers:
            raise ValueError(
                f"Critic layer {item!r} resolves to {layer}; expected 1..{num_hidden_layers}"
            )
        result.append(layer)
    if not result:
        raise ValueError("At least one critic layer is required")
    if len(set(result)) != len(result):
        raise ValueError(f"Critic layers must be unique: {result}")
    return result


class MultiLayerCritic(nn.Module):
    """Detached-feature value heads and a learned softmax layer gate."""

    def __init__(self, hidden_size: int, layers: Sequence[int], head_size: int = 256,
                 aggregation: str = "gate"):
        super().__init__()
        if aggregation not in {"gate", "mean", "min"}:
            raise ValueError(f"Unknown critic aggregation: {aggregation}")
        self.layers = list(layers)
        self.aggregation = aggregation
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in layers)
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, head_size),
                nn.SiLU(),
                nn.Linear(head_size, 1),
            )
            for _ in layers
        )
        self.gates = nn.ModuleList(nn.Linear(hidden_size, 1, bias=False) for _ in layers)
        # Start from a neutral 0.5 value estimate instead of injecting arbitrary
        # state-dependent advantages before the first critic update.
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        for gate in self.gates:
            nn.init.zeros_(gate.weight)

    def forward(self, layer_features: Sequence[torch.Tensor]):
        if len(layer_features) != len(self.layers):
            raise ValueError("Number of layer feature tensors does not match critic heads")
        # [batch, time, critics]
        logits = torch.stack(
            [head(norm(feature.detach().float())).squeeze(-1)
             for feature, norm, head in zip(layer_features, self.norms, self.heads)],
            dim=-1,
        )
        values = torch.sigmoid(logits)
        if self.aggregation == "gate":
            gate_logits = torch.stack(
                [gate(norm(feature.detach().float())).squeeze(-1)
                 for feature, norm, gate in zip(layer_features, self.norms, self.gates)],
                dim=-1,
            )
            weights = torch.softmax(gate_logits, dim=-1)
            baseline = (weights * values).sum(dim=-1)
        elif self.aggregation == "mean":
            weights = torch.full_like(values, 1.0 / len(self.layers))
            baseline = values.mean(dim=-1)
        else:
            baseline, index = values.min(dim=-1)
            weights = torch.nn.functional.one_hot(index, num_classes=len(self.layers)).to(values.dtype)
        return baseline, values, weights


def critic_loss(baseline, head_values, returns, mask, auxiliary_weight: float):
    """Masked binary cross-entropy, averaged per response then per batch."""
    eps = 1e-6
    baseline = baseline.clamp(eps, 1.0 - eps)
    head_values = head_values.clamp(eps, 1.0 - eps)
    targets = returns.unsqueeze(-1).expand_as(baseline)
    per_token = torch.nn.functional.binary_cross_entropy(baseline, targets, reduction="none")
    main = _masked_sequence_mean(per_token, mask)

    head_targets = returns[:, None, None].expand_as(head_values)
    head_loss = torch.nn.functional.binary_cross_entropy(head_values, head_targets, reduction="none")
    # Average heads before sequence normalization, so adding heads doesn't scale the loss.
    head_loss = head_loss.mean(dim=-1)
    auxiliary = _masked_sequence_mean(head_loss, mask)
    return main + auxiliary_weight * auxiliary


def _masked_sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    per_sequence = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return per_sequence.mean()
