import torch
import torch.nn as nn

def mask_nodes(x, mask_ratio=0.1, mask_value=0.0, mask_type='learnable', mask_token=None):
    """
    Randomly mask some nodes with a more advanced strategy.

    Args:
        x (torch.Tensor): Input tensor with shape (batch_size, in_steps, num_nodes, feature).
        mask_ratio (float): Proportion of nodes to mask (0-1), e.g., 0.1 means masking 10% of nodes.
        mask_value (float): The value to set for masked nodes when `mask_type='zero'`.
        mask_type (str): Type of masking - 'zero', 'learnable', 'random'.
            'zero': Masked elements are set to mask_value (e.g., 0.0).
            'learnable': Masked elements are set to a learnable mask token.
            'random': Masked elements are replaced with random values.
        mask_token (torch.Tensor or None): A learnable tensor used for masking if `mask_type='learnable'`.

    Returns:
        x_masked (torch.Tensor): Tensor after masking, with the same shape as the input.
        mask_indices (torch.Tensor): Boolean tensor indicating masked positions.
    """

    # Clone the input tensor to avoid in-place modification
    x_masked = x.clone()

    batch_size, in_steps, num_nodes, feature = x.shape
    # Create a mask of the shape (batch_size, in_steps, num_nodes)
    mask_indices = torch.rand(batch_size, in_steps, num_nodes) < mask_ratio

    # Apply the mask based on the specified masking type
    if mask_type == 'zero':
        x_masked[mask_indices] = mask_value
    elif mask_type == 'learnable' and mask_token is not None:
        # Replace the entire feature dimension with mask_token for each masked node
        x_masked[mask_indices] = mask_token  # mask_token is of shape (1, feature)
    elif mask_type == 'random':
        random_values = torch.randn_like(x_masked)
        x_masked[mask_indices] = random_values[mask_indices]

    return x_masked, mask_indices

class FeatureMask(nn.Module):
    def __init__(self, feats_dim: list):
        super().__init__()

        self.masks = nn.ParameterList([
            nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, dim))) for dim in feats_dim
        ])

    def forward(self, feats: list, mask_ratio=0.15):
        masked_feats, masks_indices = [], []
        for i, feat in enumerate(feats):
            masked_feat, mask_indices = mask_nodes(
                feat,
                mask_ratio,
                mask_type="learnable",
                mask_token=self.masks[i]
            )
            masked_feats.append(masked_feat)
            masks_indices.append(mask_indices)

        return masked_feats, masks_indices