import torch
import torch.nn as nn

def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    # TODO: return a Sequential of Linear -> ReLU -> Linear
    l1 = nn.Linear(in_dim, hidden_dim)
    relu = nn.ReLU()
    l2 = nn.Linear(hidden_dim, out_dim)

    return nn.Sequential(l1, relu, l2)
