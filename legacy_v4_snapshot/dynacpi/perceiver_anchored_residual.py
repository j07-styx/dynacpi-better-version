"""Baseline-preserving residual head for a future DynaCPI experiment.

Research hypothesis: instead of replacing a strong frozen PerceiverCPI
predictor, learn a *bounded* additive refinement from independently trained
features. The head is identity-exact at initialization and whenever the
experimental feature is unavailable. This file neither loads data nor trains.
All actual affinity training requires the approved scientific cohort gate.
"""
from __future__ import annotations
import math

import torch
from torch import nn


class PerceiverAnchoredResidual(nn.Module):
    """An opt-in residual on a FROZEN PerceiverCPI prediction.

    Args:
      feature_width: width of precomputed DynaCPI residual features.
      cap: maximum possible absolute change in the pKd prediction.
      hidden: residual MLP hidden width.

    Zero last layer implies exact parity with frozen baseline at initialization.
    The baseline tensor is detached to prevent unplanned baseline fine-tuning.
    """

    def __init__(self, feature_width: int, hidden: int = 64,
                 cap: float = 0.5):
        super().__init__()
        if not isinstance(feature_width, int) or feature_width < 1:
            raise ValueError("Feature width must be positive integer")
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError("Hidden width must be positive integer")
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("Residual cap must be finite positive")
        self.feature_width = feature_width
        self.cap = float(cap)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, perceiver_pkd: torch.Tensor,
                residual_features: torch.Tensor,
                availability: torch.Tensor | None = None,
                return_delta: bool = False):
        if not isinstance(perceiver_pkd, torch.Tensor) or not isinstance(residual_features, torch.Tensor):
            raise TypeError("Expected frozen Perceiver predictions and residual features as tensors")
        if perceiver_pkd.ndim != 1 or residual_features.ndim != 2:
            raise ValueError("Expected Perceiver [B] and residual features [B,F]")
        if residual_features.shape != (perceiver_pkd.numel(), self.feature_width):
            raise ValueError("Batch size or residual feature dimension mismatch")
        if perceiver_pkd.device != residual_features.device or perceiver_pkd.dtype != residual_features.dtype:
            raise ValueError("Frozen prediction and feature dtype/device mismatch")
        if not (torch.isfinite(perceiver_pkd).all() and torch.isfinite(residual_features).all()):
            raise ValueError("Non-finite input is not allowed")
        if availability is None:
            gate = None
        else:
            if not isinstance(availability, torch.Tensor) or availability.shape != perceiver_pkd.shape:
                raise ValueError("Availability must be [B] tensor")
            if availability.dtype != perceiver_pkd.dtype or availability.device != perceiver_pkd.device:
                raise ValueError("Availability device/dtype mismatch")
            if not torch.isfinite(availability).all() or torch.any(availability < 0) or torch.any(availability > 1):
                raise ValueError("Availability must be finite within [0,1]")
            gate = availability
        baseline = perceiver_pkd.detach()
        raw = self.head(residual_features).squeeze(-1)
        delta = self.cap * torch.tanh(raw)
        if gate is not None:
            delta = delta * gate
        prediction = baseline + delta
        return (prediction, delta) if return_delta else prediction
