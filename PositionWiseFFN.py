from torch import nn

class PositionWiseFFN(nn.Module):
    def __init__(self, ffnNumInputs, ffnNumHiddens, ffnNumOutputs, **kargs) -> None:
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffnNumInputs, ffnNumHiddens)
        self.glue = nn.GELU()
        self.dense2 = nn.Linear(ffnNumHiddens, ffnNumOutputs)
    
    def forward(self, X):
        return self.dense2(self.glue(self.dense1(X)))
