import math
from torch import nn
from torch.nn import functional as F

import Encoder, Decoder, PositionalEncoding

# Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, vocabSize, QSize, KSize, VSize, 
                       ffnNumInputs, ffnNumHiddens, dropout,
                       numHeads, numHiddens, numLayers, normShape,
                       bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.numHiddens = numHiddens
        self.embedding = nn.Embedding(vocabSize, numHiddens)
        self.posEncoding = PositionalEncoding(numHiddens, dropout)
        self.encoderBlk = nn.Sequential()
        for i in range(numLayers):
            self.encoderBlk.add_module('block' + str(i),
                                       Encoder(QSize, KSize, VSize, normShape,
                                               numHeads, numHiddens, dropout,
                                               ffnNumInputs, ffnNumHiddens, bias))
        
    def forward(self, X, validLens, *args):
        X = self.posEncoding(self.embedding(X) * math.sqrt(self.numHiddens))
        self.attentionWeights = [None] * len(self.encoderBlk)
        for i, blk in enumerate(self.encoderBlk):
            X = blk(X, validLens)
            self.attentionWeights[i] = blk.attention.attention.weights
        return X

# Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocabSize, QSize, KSize, VSize, 
                       ffnNumInputs, ffnNumHiddens, dropout,
                       numHeads, numHiddens, numLayers, normShape, 
                       **kwargs) -> None:
        super(TransformerDecoder, self).__init__(**kwargs)
        self.numHiddens = numHiddens
        self.numLayers = numLayers
        self.embedding = nn.Embedding(vocabSize, numHiddens)
        self.posEncoding = PositionalEncoding(numHiddens, dropout)
        self.decoderBlk = nn.Sequential()
        for i in range(numLayers):
            self.decoderBlk.add_module('block'+str(i),
                                       Decoder(QSize, KSize, VSize, normShape,
                                               numHeads, numHiddens, dropout, 
                                               ffnNumInputs, ffnNumHiddens, i))
        self.dense = nn.Linear(numHiddens, vocabSize)

    def init_state(self, encoderOutputs, encoderValidLens, *args):
        return [encoderOutputs, encoderValidLens, [None] * self.numLayers]

    def forward(self, X, state):
        X = self.posEncoding(self.embedding(X) * math.sqrt(self.numHiddens))
        self.attention_weights = [[None] * len(self.decoderBlk) for _ in range(2)]
        for i, blk in enumerate(self.decoderBlk):
            X, state = blk(X, state)
            self.attention_weights[0][i] = blk.attention1.attention.weights
            self.attention_weights[1][i] = blk.attention2.attention.weights
        return self.dense(X), state

# Whole
class Transformer:
    def __init__(self, encoder, decoder) -> None:
        self.encoder = encoder
        self.decoder = decoder

    def forward(self):
        pass