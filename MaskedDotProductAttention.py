import math
import torch
from torch import nn
from torch.nn import functional as F
from utils import masked_softmax

class MaskedDotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs) -> None:
        super(MaskedDotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, val_lens=None):
        dim = Q.shape[-1]
        scores = torch.bmm(Q, K.T) / math.sqrt(dim)
        self.weights = masked_softmax(scores, val_lens)
        return torch.bmm(self.dropout(self.weights), V)
