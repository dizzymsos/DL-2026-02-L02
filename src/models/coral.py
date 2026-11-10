"""
Experimento 2: Conv1D + MLP con cabeza ordinal CORAL.

El tronco Conv1D + MLP (Conv1DBackbone) es el mismo del experimento Softmax:
solo cambia la cabeza de salida.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
# Adam mueve cada parametro cerca de lr por paso. Con lr = 0.001 y unos 560
# pasos, los sesgos no alcanzan a separarse. La escala acelera su aprendizaje.
BIAS_SCALE = 10.0

from src.models.blocks import Conv1DBackbone


class CoralLayer(nn.Module):
    """
    Capa de salida CORAL.

    - puntaje latente compartido s = w^T h (sin sesgo),
    - sesgos ordenados b_0 >= b_1 >= ... >= b_{K-2}; softplus > 0 fuerza el orden,
    - logits z_k = s + b_k.

    Con K = 2 hay un solo umbral y self.bias_gaps queda vacio (largo 0).

    Formas:
    - x: (batch_size, input_size)
    - salida: (batch_size, num_classes - 1)
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.score = nn.Linear(input_size, 1, bias=False)
        self.first_bias = nn.Parameter(torch.zeros(1))
        self.bias_gaps = nn.Parameter(torch.zeros(num_classes - 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first_bias = BIAS_SCALE * self.first_bias
        decrements = torch.cumsum(F.softplus(BIAS_SCALE * self.bias_gaps), dim=0)
        biases = torch.cat([first_bias, first_bias - decrements])
        return self.score(inputs) + biases  # (B, 1) + (K-1,) -> (B, K-1)


class Conv1DCoralNet(nn.Module):
    """
    Conv1D + MLP con cabeza CORAL.

        (B, 15) -> Conv1DBackbone -> (B, hidden_dim)
                -> CoralLayer(hidden_dim, K) -> (B, K-1)

    Devuelve logits: no se aplica sigmoide en el forward.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Conv1DBackbone(
            num_features=num_features,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.head = CoralLayer(self.backbone.output_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(inputs))