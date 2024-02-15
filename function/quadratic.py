# function:quadratic.py
import torch
import numpy as np
import matplotlib.pyplot as plt

class QuadraticModel(torch.nn.Module):
  def __init__(self):
    super(QuadraticModel, self).__init__()
    self.linear = torch.nn.Linear(2,1)
  def forward(self, x):
    v = torch.cat([x* x,  x], dim = -1)
    out = self.linear(v)
    return out
quadratic_model = QuadraticModel()
print(quadratic_model.linear.weight)
print(quadratic_model.linear.bias)

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = 3 * x * x - 4 * x + 6
yb = y + torch.randn(x.size())

plt.plot(x.numpy(), yb.numpy(), 'b+', label = 'yb')
plt.plot(x.numpy(), y.numpy(), 'r', label = 'y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid('True', color='y')
plt.show()

learning_rate = 1e-5
lossfunc = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(quadratic_model.parameters(), lr = learning_rate)

from math import nan
stop = False
loss_threshold = 1e-6
loss_list = []
last_loss = nan
while (stop == False):
    optimizer.zero_grad()
    yh = quadratic_model(x)
    loss = lossfunc(yh, yb)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())
    difference = torch.abs(loss - last_loss)
    if (difference < loss_threshold):
      stop = True
    last_loss = loss
print(quadratic_model.linear.weight)
print(quadratic_model.linear.bias)

plt.plot(loss_list, 'r')
plt.tight_layout()
plt.grid('True', color='y')
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")
plt.yscale("log")
plt.show()

plt.plot(x.numpy(), yh.detach().numpy(), 'b+', label = 'yh')
plt.plot(x.numpy(), y.numpy(), 'r', label = 'y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid('True', color='y')
plt.show()
