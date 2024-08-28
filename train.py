import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import Transformer

print(torch.cuda.is_available())

device = torch.device('cuda')
model = Transformer()
batch_size = 128
epochs = 10000
criterion = nn.loss
optimizer = optim.Adam(model.parameters(), lr=.01)

model = model.to(device)  # Train on GPU
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model()
    loss = criterion()
    loss.sum().backward()
    optimizer.step()

    if epoch % 50 == 0:
        print()