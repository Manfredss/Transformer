from torch import nn
from torch.nn import functional as F
import MultiHeadAttention, AddNorm, PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, QSize, KSize, VSize, normShape,
                 numHeads, numHiddens, dropout,  
                 ffnNumInputs, ffnNumHiddens, bias=False, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(QSize, KSize, VSize, 
                                            numHeads, numHiddens, dropout, bias)
        self.addnorm1 = AddNorm(normShape, dropout)
        self.ffn = PositionalEncoding(ffnNumInputs, ffnNumHiddens, ffnNumHiddens)
        self.addnorm2 = AddNorm(normShape, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X, X, X))
        return self.addnorm2(Y + self.ffn(Y))
