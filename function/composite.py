# function:composite.ipynb

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

def poly1(x):
  return 0.003 * (x * 3) - 0.02 * (x ** 2) + 0.5 * x + 0.7
def poly2(x):
  return -0.01 * (x ** 2) + 0.4 * x + 0.2
def func(x):
  return torch.sin(x * poly1(x)) + torch.cos(poly2(x))

x = torch.arange(-3, 3, 0.05, dtype = torch.float32).view(-1,1)
y = func(x)

class CompFunc(nn.Module):
  def __init__(self, degree):
    super(CompFunc, self).__init__()
    self.degree = degree
    self.linear1 = torch.nn.Linear(4, self.degree)
    self.linear2 = torch.nn.Linear(self.degree, 8)
    self.linear3 = torch.nn.Linear(8, 1)
  def forward(self,x):
    v = torch.cat([x ** i for i in range(1, self.degree)], dim=-1)
    y = self.linear1(v)
    y = torch.relu(y)
    y = self.linear2(y)
    y = torch.relu(y)
    y = self.linear3(y)
    return y

def Train(model, input, output, learning_rate = 1e-4, loss_threshold = 1e-2, max_iterations = 1e+5):
  stop = False
  loss_list = []
  stop = False
  lossfunc = nn.MSELoss(reduction='sum')
  optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
  iter = 0
  while ((stop == False) and (iter < max_iterations)):
    ybar = model(input)
    optimizer.zero_grad()
    loss = lossfunc(ybar, output)
    loss.backward()
    optimizer.step()
    if (loss < loss_threshold):
      stop = True
    iter = iter + 1
  return iter, loss

compf = CompFunc(5)
iterations, loss = Train(compf, x, y, loss_threshold= 0.1)
print(iterations)
print(loss)

yh = compf(x)
plt.plot(x.numpy(), yh.detach().numpy(), 'b+', label='yh')
plt.plot(x.numpy(), y.numpy(), 'r', label='y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid('True', color='y')
plt.show()
