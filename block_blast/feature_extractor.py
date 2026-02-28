"""Custom CNN feature extractor for the 8×8 Block Blast grid.

The standard SB3 NatureCNN expects images ≥ 36×36.  Our observation is a
compact (C, 8, 8) tensor so we use a lightweight residual-style CNN that
preserves full spatial resolution through 3×3 convolutions, then compresses
via a 1×1 bottleneck before the fully-connected head.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResBlock(nn.Module):
    """Pre-activation residual block (keeps spatial dims unchanged)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BlockBlastCNN(BaseFeaturesExtractor):
    """CNN tailored for (C, 8, 8) Block Blast observations.

    Architecture
    ------------
      Conv 3×3  (C → 64)  →  3 × ResBlock(64)  →  Conv 1×1 (64 → 32)
      →  Flatten (32·8·8 = 2048)  →  Linear → ReLU  →  features_dim
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        n_res_blocks: int = 3,
        channels: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)
        in_ch = observation_space.shape[0]
        h, w = observation_space.shape[1], observation_space.shape[2]

        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, channels, 3, padding=1),
            nn.ReLU(),
        ]
        for _ in range(n_res_blocks):
            layers.append(ResBlock(channels))

        bottleneck_ch = 32
        layers += [
            nn.Conv2d(channels, bottleneck_ch, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(bottleneck_ch * h * w, features_dim),
            nn.ReLU(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
