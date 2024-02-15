# function:activation.py
import torch
import numpy as np
import matplotlib.pyplot as plt

def plotxy(x, y):
  plt.plot(x.numpy(), y.numpy(), 'r', label='y')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid('True', color='y')
  plt.show()

x = torch.arange(-5, 5, 0.1).view(-1, 1)
yrelu = torch.nn.functional.relu(x)
plotxy(x, yrelu)

ysig = torch.nn.functional.sigmoid(x)
plotxy(x, ysig)
