import math
import torch
from torch import nn
from transformers.activations import get_activation


class Adapter(nn.Module):
    def __init__(self, dim, r, act):
        super().__init__()
        self.adapter_A = nn.Linear(dim, r)
        self.act = get_activation(act)
        self.adapter_B = nn.Linear(r, dim)

    def forward(self, x, residual):
        result = self.adapter_A(x)
        result = self.act(result)
        result = self.adapter_B(result)
        return result + residual

class Hydra(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
    ):
        super().__init__()

        self.hydra_A = nn.Linear(in_features, r, bias=False)
        self.hydra_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.hydra_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.hydra_B.weight)

    def forward(self, x):
        result = self.hydra_A(x)
        result = self.hydra_B(result)
        return result