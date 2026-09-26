import torch
import torch.nn.functional as F


def unet_time_embedding(timesteps: list, embed_dim: int, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor, max_period: int = 10000) -> torch.Tensor:
    """
    Compute time embeddings for a diffusion model U-Net.

    Args:
        timesteps: list or 1D array of shape (B,) with timestep values
        embed_dim: dimension of sinusoidal embedding (must be even)
        W1: weight matrix of first linear layer, shape (embed_dim, hidden_dim)
        b1: bias of first linear layer, shape (hidden_dim,)
        W2: weight matrix of second linear layer, shape (hidden_dim, output_dim)
        b2: bias of second linear layer, shape (output_dim,)
        max_period: controls the frequency range for sinusoidal embedding

    Returns:
        torch.Tensor of shape (B, output_dim) with time embeddings
    """
    half_dim = embed_dim//2
    freq = torch.arange(half_dim, dtype=torch.float64)
    timesteps = torch.tensor(timesteps, dtype=freq.dtype)
    freq = torch.exp(-torch.log(torch.tensor(max_period)) * freq/half_dim)
    args = timesteps.unsqueeze(-1) @ freq.unsqueeze(0)

    sin_pos = torch.sin(args)
    cos_pos = torch.cos(args)
    pos_emb = torch.cat([sin_pos, cos_pos], dim=-1)

    h1 = pos_emb @ W1 + b1
    a1 = (1.0/(1.0 + torch.exp(-h1))) * h1

    return a1 @ W2 + b2
