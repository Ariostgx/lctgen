import torch
from torch import nn

import math

class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        
        layers = []
        for i in range(len(dims) - 1):
          layers.append(nn.Linear(dims[i], dims[i + 1]))
          if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)        
        
    def forward(self, x):
        x = self.mlp(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe

def pos2posemb(pos, num_pos_feats=128, temperature=10000):
  """
  Copied from https://github.com/OpenDriveLab/UniAD/blob/main/projects/mmdet3d_plugin/models/utils/functional.py
  Convert 2D position into positional embeddings.

  Args:
      pos (torch.Tensor): Input N-D position tensor.
      num_pos_feats (int, optional): Number of positional features. Default is 128.
      temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

  Returns:
      torch.Tensor: Positional embeddings tensor.
  """
  scale = 2 * math.pi
  pos = pos * scale
  dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
  dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
  
  D = pos.shape[-1]
  pos_dims = []
  for i in range(D):
    pos_dim_i = pos[..., i, None] / dim_t
    pos_dim_i = torch.stack((pos_dim_i[..., 0::2].sin(), pos_dim_i[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_dims.append(pos_dim_i)
  posemb = torch.cat(pos_dims, dim=-1)
  return posemb