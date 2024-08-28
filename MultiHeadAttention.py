import DotProductAttention
from torch import nn
from torch.nn import functional as F
from utils import transpose_output, transpose_qkv



class MultiHeadAttention(nn.Module):
    def __init__(self, QSize, KSize, VSize, 
                 numHeads, numHiddens, dropout,
                 bias=False, **kwargs) -> None:
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.numHeads = numHeads
        self.attention = DotProductAttention(dropout)
        self.W_Q = nn.Linear(QSize, numHiddens, bias=bias)
        self.W_K = nn.Linear(KSize, numHiddens, bias=bias)
        self.W_V = nn.Linear(VSize, numHiddens, bias=bias)
        self.W_O = nn.Linear(numHiddens, numHiddens, bias=bias)

    def forward(self, Q, K, V):
        Q = transpose_qkv(self.W_Q(Q), self.numHeads)
        K = transpose_qkv(self.W_K(K), self.numHeads)
        V = transpose_qkv(self.W_V(V), self.numHeads)
        output = transpose_output(self.attention(Q, K, V), self.numHeads)
        return self.W_O(output)
