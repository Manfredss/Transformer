import math
import torch
from torch import nn
from torch.nn import functional as F

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs) -> None:
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        dim = Q.shape[-1]
        scores = torch.bmm(Q, K.T) / math.sqrt(dim)
        self.weights = F.softmax(scores)
        return torch.bmm(self.dropout(self.weights), V)
