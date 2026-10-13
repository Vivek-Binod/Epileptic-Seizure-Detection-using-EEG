"""Compact 1D Convolutional Neural Network for Multi-Channel EEG Seizure Detection.

Designed specifically for 18-channel longitudinal bipolar EEG windows:
Input shape: (batch_size, 18, 1024) -> 18 channels, 4 seconds at 256 Hz.

Architectural Principles:
1. Temporal Filter Bank: Captures multi-frequency oscillatory dynamics (delta, theta, alpha, beta).
2. Spatial Mixing: Cross-channel aggregation across anatomical scalp chains.
3. Lightweight: Parameter count < 50k to prevent overfitting and stay comfortably within
   4 GB VRAM on NVIDIA RTX 3050 Laptop GPU.
4. Numerical Stability: Batch normalization, ELU activations, and dropout regularization.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CompactEEGCNN(nn.Module):
        """Compact 1D Temporal-Spatial Convolutional Neural Network."""

        def __init__(
            self,
            in_channels: int = 18,
            seq_len: int = 1024,
            base_filters: int = 16,
            dropout_rate: float = 0.25,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.seq_len = seq_len
            self.base_filters = base_filters
            self.dropout_rate = dropout_rate

            # Block 1: Temporal Multi-scale Filtering
            # Kernel 64 at 256 Hz = 0.25 seconds (covers theta/delta oscillation cycles)
            self.temporal_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=base_filters,
                kernel_size=64,
                stride=2,
                padding=31,
                bias=False,
            )
            self.bn1 = nn.BatchNorm1d(base_filters)
            self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # 512 -> 128
            self.drop1 = nn.Dropout(dropout_rate)

            # Block 2: Mid-level Feature Representation
            self.conv2 = nn.Conv1d(
                in_channels=base_filters,
                out_channels=base_filters * 2,
                kernel_size=16,
                stride=1,
                padding=8,
                bias=False,
            )
            self.bn2 = nn.BatchNorm1d(base_filters * 2)
            self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)  # 128 -> 32
            self.drop2 = nn.Dropout(dropout_rate)

            # Block 3: High-level Pattern Aggregation
            self.conv3 = nn.Conv1d(
                in_channels=base_filters * 2,
                out_channels=base_filters * 4,
                kernel_size=8,
                stride=1,
                padding=4,
                bias=False,
            )
            self.bn3 = nn.BatchNorm1d(base_filters * 4)
            self.gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling -> (B, base_filters * 4, 1)
            self.drop3 = nn.Dropout(dropout_rate * 1.2)

            # Classification Head
            dense_dim = base_filters * 4
            self.fc1 = nn.Linear(dense_dim, 32)
            self.fc_bn = nn.BatchNorm1d(32)
            self.drop_fc = nn.Dropout(dropout_rate)
            self.fc_out = nn.Linear(32, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (batch_size, 18, 1024)

            Returns:
                Logits tensor of shape (batch_size, 1)
            """
            # Block 1
            h = self.temporal_conv(x)
            h = self.bn1(h)
            h = F.elu(h)
            h = self.pool1(h)
            h = self.drop1(h)

            # Block 2
            h = self.conv2(h)
            h = self.bn2(h)
            h = F.elu(h)
            h = self.pool2(h)
            h = self.drop2(h)

            # Block 3
            h = self.conv3(h)
            h = self.bn3(h)
            h = F.elu(h)
            h = self.gap(h)
            h = self.drop3(h)

            # Flatten
            h = h.squeeze(-1)

            # Dense head
            h = self.fc1(h)
            h = self.fc_bn(h)
            h = F.elu(h)
            h = self.drop_fc(h)
            logits = self.fc_out(h)

            return logits

        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """Compute seizure probability via sigmoid."""
            logits = self.forward(x)
            return torch.sigmoid(logits)

        def count_parameters(self) -> int:
            """Count total trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class WeightedFocalLoss(nn.Module):
        """Focal Loss with positive class weighting for extreme imbalance."""

        def __init__(self, gamma: float = 2.0, alpha: float = 0.75, pos_weight: Optional[float] = None):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.pos_weight = pos_weight

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            probs = torch.sigmoid(logits)
            targets = targets.view_as(logits)

            # Standard BCE loss per element
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none",
                pos_weight=torch.tensor([self.pos_weight], device=logits.device) if self.pos_weight else None
            )

            # Modulating factor: (1 - p_t)^gamma
            p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
            focal_weight = (1.0 - p_t) ** self.gamma

            if self.alpha is not None:
                alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
                focal_loss = alpha_t * focal_weight * bce_loss
            else:
                focal_loss = focal_weight * bce_loss

            return focal_loss.mean()

else:
    # Fallback stubs if PyTorch is not yet installed
    class CompactEEGCNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for CompactEEGCNN.")

    class WeightedFocalLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for WeightedFocalLoss.")
