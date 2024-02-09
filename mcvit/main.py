import torch 
from torch import nn, Tensor


class MCViT(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: Tensor) -> Tensor:
        return x 