import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim, dropout=0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.res_proj = None

        if input_dim != hidden_dim:
            self.res_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        if self.res_proj is None:
            hidden = hidden + input_data
        else:
            hidden = hidden + self.res_proj(input_data)

        return hidden
