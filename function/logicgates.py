# function:logicgates.py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class Gate(nn.Module):
  def __init__(self):
    super(Gate, self).__init__()
    self.layer1 = nn.Linear(2,4)
    self.layer2 = nn.Linear(4,1)
  def forward(self,x):
    x = torch.nn.functional.relu(self.layer1(x))
    x = torch.nn.functional.sigmoid(self.layer2(x))
    return x

def Train(gate, input, output, learning_rate = 1e-4, loss_threshold = 1e-2):
  stop = False
  loss_list = []
  stop = False
  lossfunc = nn.BCELoss(reduction = 'sum')
  optimizer = torch.optim.SGD(gate.parameters(), lr = learning_rate)
  iter = 0
  while ((stop == False) and (iter < 1e+5)):
    ybar = gate(input)
    optimizer.zero_grad()
    loss = lossfunc(ybar, output)
    loss.backward()
    optimizer.step()
    if (loss < loss_threshold):
      stop = True
    iter = iter + 1
  return iter

# OR gate
x = torch.tensor([[0, 0], [0,1], [1,0], [1,1]], dtype=torch.float32, requires_grad=True)
yor = torch.tensor([[0], [1], [1], [1]], dtype=torch.float32)
orgate = Gate()
iter = Train(orgate, x, yor, 1e-2)
print(orgate(x))
print("Training takes " + str(iter) + " iterations")

# AND gate
yand = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
andgate = Gate()
iter = Train(andgate, x, yand, 1e-2)
print(andgate(x))
print("Training takes " + str(iter) + " iterations")

# NAND gate
ynand = torch.tensor([[1], [1], [1], [0]], dtype=torch.float32)
nandgate = Gate()
iter = Train(nandgate, x, ynand, 1e-2)
print(nandgate(x))
print("Training takes " + str(iter) + " iterations")

# XOR gate
yxor = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
xorgate = Gate()
iter = Train(xorgate, x, yxor, 1e-2)
print(xorgate(x))
print("Training takes " + str(iter) + " iterations")
