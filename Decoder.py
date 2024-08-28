import torch
from torch import nn
from torch.nn import functional as F

import AddNorm, MaskedMultiHeadAttention, MultiHeadAttention, PositionWiseFFN

class Decoder(nn.Module):
    def __init__(self, QSize, KSize, VSize, normShape,
                 numHeads, numHiddens, dropout, 
                 ffnNumInputs, ffnNumHiddens, i, **kwargs) -> None:
        super(Decoder, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MaskedMultiHeadAttention(QSize, KSize, VSize, numHeads, numHiddens, dropout)
        self.addnorm1 = AddNorm(normShape, dropout)

        self.attention2 = MultiHeadAttention(QSize, KSize, VSize, numHeads, numHiddens, dropout)
        self.addnorm2 = AddNorm(normShape, dropout)

        self.ffn = PositionWiseFFN(ffnNumInputs, ffnNumHiddens, ffnNumHiddens)
        self.addnorm3 = AddNorm(normShape, dropout)

    def forward(self, X, state):
        encoderOutput, encoderValidLen= state[0], state[1]
        """ 
        训练阶段输出序列的所有词元都在同一时间处理
        因此state[2][self.i]初始化为None
        预测阶段输出序列是通过词元一个接一个解码的
        因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        """
        # 第一层，还没接受到 encoder 输入 
        if state[2][self.i] is None:
            keyVal = X
        else:
            keyVal = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = keyVal

        if self.training:
            batchSize, numSteps, _ = X.shape
            decValidLens = torch.arange(1, numSteps + 1, device=X.device).repeat(batchSize, 1)
        else:
            decValidLens = None

        X1 = self.attention1(X, keyVal, keyVal, decValidLens)
        Y = self.addnorm1(X, X1)
        Y1 = self.attention2(Y, encoderOutput, encoderOutput, encoderValidLen)
        Y2 = self.addnorm2(Y, Y1)
        return self.addnorm3(Y, self.ffn(Y2)), state
