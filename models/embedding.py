import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, steps_per_day=288, tod_emb_dim=16, dow_emb_dim=16):
        super().__init__()

        self.steps_per_day = steps_per_day
        self.tod_embedding = nn.Embedding(steps_per_day, tod_emb_dim)
        self.dow_embedding = nn.Embedding(7, dow_emb_dim)

    def forward(self, tod, dow):
        # x: (batch_size, in_steps, num_nodes, feat_dim)
        # tod: (batch_size, in_steps, num_nodes)
        # dow: (batch_size, in_steps, num_nodes)

        # (batch_size, in_steps, num_nodes, tod_emb_dim)
        tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
        # (batch_size, in_steps, num_nodes, dow_emb_dim)
        dow_emb = self.dow_embedding(dow.long())
        # (batch_size, in_steps, num_nodes, tod_emb_dim+dow_emb_dim)
        time_emb = torch.cat([tod_emb, dow_emb], dim=-1)

        return time_emb


class PositionEmbedding(nn.Module):
    def __init__(self, num_nodes, position_dim, method='learnable'):
        """
        :param num_nodes: Number of nodes (e.g., sensors or road nodes in traffic flow)
        :param position_dim: Embedding dimension for each node
        :param method: 'learnable' or 'sinusoidal', the method for position encoding
        """
        super(PositionEmbedding, self).__init__()

        self.num_nodes = num_nodes
        self.position_dim = position_dim
        self.method = method

        if method == 'learnable':
            # Learnable position embeddings
            self.position_embeddings = nn.Embedding(num_nodes, position_dim)
        elif method == 'sinusoidal':
            # Sinusoidal position encoding
            self.position_embeddings = None
            self._init_sinusoidal_embeddings()
        else:
            raise ValueError("Invalid method. Choose 'learnable' or 'sinusoidal'.")

    def _init_sinusoidal_embeddings(self):
        """Initialize sinusoidal position embeddings."""
        # Generate a fixed sinusoidal encoding matrix
        position_enc = torch.zeros(self.num_nodes, self.position_dim)
        for pos in range(self.num_nodes):
            for i in range(0, self.position_dim, 2):
                position_enc[pos, i] = math.sin(pos / (10000 ** (i / self.position_dim)))
                position_enc[pos, i + 1] = math.cos(pos / (10000 ** (i / self.position_dim)))
        # Make it a non-learnable parameter
        self.position_enc = nn.Parameter(position_enc, requires_grad=False)

    def forward(self, x):
        """
        :param x: Input data of shape (batch_size, in_steps, num_nodes, feat_dim)
        :return: Output with position embeddings of shape (batch_size, in_steps, num_nodes, feat_dim + position_dim)
        """
        batch_size, in_steps, num_nodes, feat_dim = x.shape

        if self.method == 'learnable':
            # Use learnable position embeddings
            position_enc = self.position_embeddings(torch.arange(num_nodes, device=x.device))  # (num_nodes, position_dim)
            position_enc = position_enc.unsqueeze(0).unsqueeze(1)  # (1, 1, num_nodes, position_dim)
            position_enc = position_enc.expand(batch_size, in_steps, -1, -1)  # (batch_size, in_steps, num_nodes, position_dim)
        elif self.method == 'sinusoidal':
            # Use sinusoidal position encoding
            position_enc = self.position_enc.unsqueeze(0).unsqueeze(1)  # (1, 1, num_nodes, position_dim)
            position_enc = position_enc.expand(batch_size, in_steps, -1, -1)  # (batch_size, in_steps, num_nodes, position_dim)

        return position_enc


if __name__ == "__main__":
    x = torch.ones(1, 2, 2, 3)
    pos_emb = PositionEmbedding(2, 3)
    x = pos_emb(x)
    print(x)