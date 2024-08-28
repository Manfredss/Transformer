# import utils
import torch
from torch import nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, numHiddens, dropout, maxLength=512) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # batchSize = 1
        self.P = torch.zeros((1, maxLength, numHiddens))
        # pos / 10000^(2i/d)
        X = torch.arange(maxLength, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, numHiddens, 2, dtype=torch.float32) / numHiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    

# pe = PositionalEncoding(256, 0)
# pe.eval()
# X = pe(torch.zeros((1, 60, 256)))
# P = pe.P[:, :X.shape[1], :]
# utils.plot(torch.arange(60), P[0, :, 6:10].T, xlabel='Row (position)',
#          figsize=(6, 2.5), legend=['Col %d' %d for d in torch.arange(6, 10)])