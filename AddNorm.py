from torch import nn

class AddNorm(nn.Module):
    def __init__(self, normShape, dropout, **kwargs) -> None:
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(normShape)

    def forward(self, X, Y):
        return self.layernorm(self.dropout(Y) + X)
