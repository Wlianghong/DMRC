import torch
import torch.nn as nn
from torchinfo import summary
from .transformer import SelfAttentionLayer
from .embedding import TimeEmbedding, PositionEmbedding
from .mask import mask_nodes
from .norm import SNorm, TNorm
from .mask import FeatureMask


class STAttnBlock(nn.Module):
    def __init__(self, model_dim, feed_forward_dim, num_heads, dropout):
        super().__init__()
        self.t_attn = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
        self.s_attn = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, model_dim)
        x = self.t_attn(x, dim=1)
        x = self.s_attn(x, dim=2)
        return x


class Predictor(nn.Module):
    def __init__(
            self,
            model_dim,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=1,
            dropout=0.1,
            output_dim=1,
            out_steps=12,
            in_steps=12
    ):
        super().__init__()

        self.output_dim = output_dim
        self.out_steps = out_steps

        self.attn_layers = nn.ModuleList(
            [STAttnBlock(model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.output_proj = nn.Linear(in_steps * model_dim, out_steps * output_dim)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, model_dim)
        batch_size, in_steps, num_nodes, model_dim = x.shape

        for attn in self.attn_layers:
            x = attn(x)

        out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
        out = out.reshape(
            batch_size, num_nodes, in_steps * model_dim
        )
        out = self.output_proj(out).view(
            batch_size, num_nodes, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        return out


class DMRCFormer(nn.Module):
    def __init__(
            self,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=1,
            output_dim=1,
            input_embedding_dim=24,
            temporal_embedding_dim=48,
            spatial_embedding_dim=24,
            adaptive_embedding_dim=24,
            add_norm=False,
            mask_ratio=0.15,
            use_recon=True,
            feed_forward_dim=256,
            num_heads=4,
            num_shared_layers=2,
            num_branch_layers=1,
            dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.add_norm = add_norm
        self.mask_ratio = mask_ratio
        self.use_recon = use_recon
        self.model_dim = (
            input_embedding_dim
            + temporal_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        masks_dim = [input_embedding_dim, temporal_embedding_dim]

        # ####################################### embedding module #######################################
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        if self.add_norm:
            self.s_norm = SNorm(input_embedding_dim)
            self.t_norm = TNorm(input_embedding_dim)

        self.time_embedding = TimeEmbedding(
            steps_per_day,
            temporal_embedding_dim // 2,
            temporal_embedding_dim // 2,
        )

        if self.spatial_embedding_dim > 0:
            self.pos_embedding = PositionEmbedding(
                num_nodes,
                spatial_embedding_dim,
                # "learnable"
                "sinusoidal"
            )
            masks_dim.append(spatial_embedding_dim)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
            masks_dim.append(adaptive_embedding_dim)

        # ####################################### masking module #######################################
        self.mask = FeatureMask(masks_dim)

        # ####################################### Reconstruction-Constrained module #######################################
        self.shared_attn_layers = nn.ModuleList(
            [STAttnBlock(self.model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_shared_layers)]
        )

        self.x_predictor = Predictor(
            self.model_dim, feed_forward_dim, num_heads, num_branch_layers, dropout, output_dim, out_steps, in_steps)

        self.y_predictor = Predictor(
            self.model_dim, feed_forward_dim, num_heads, num_branch_layers, dropout, output_dim, out_steps, in_steps)

    def forward(self, x, is_train=False):
        # x: (batch_size, in_steps, num_nodes, data+tod+dow=input_dim)
        batch_size, in_steps, num_nodes, model_dim = x.shape
        origin_x = x

        tod = x[..., 1]
        dow = x[..., 2]
        x = x[..., :self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)

        if self.add_norm:
            x_snorm = self.s_norm(x)
            x_tnorm = self.t_norm(x)
            x = x + x_tnorm + x_snorm

        time_emb = self.time_embedding(tod, dow)

        features = [x, time_emb]

        if self.spatial_embedding_dim > 0:
            pos_emb = self.pos_embedding(x)
            features.append(pos_emb)

        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        res = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        # add mask
        if is_train:
            features, masks_indices = self.mask(features, self.mask_ratio)
            mask_indices = masks_indices[0]
            mask_labels = origin_x[..., :1][mask_indices]

        x = torch.cat(features, dim=-1)

        for attn in self.shared_attn_layers:
            x = attn(x)

        y = self.y_predictor(x + res)
        if is_train and self.use_recon:
            x_hat = self.x_predictor(x)
            x_hat = x_hat[mask_indices]

        return {
            "y": y,
            "x_hat": None if not (is_train and self.use_recon) else x_hat,
            "mask_labels": None if not (is_train and self.use_recon) else mask_labels
        }

if __name__ == "__main__":
    model = DMRCFormer(207, 12, 12)
    summary(model, [64, 12, 207, 3])
