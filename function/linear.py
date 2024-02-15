# function:linear.py
import torch
import numpy as np
import matplotlib.pyplot as plt

class LinearModel(torch.nn.Module):
  def __init__(self):
    super(LinearModel, self).__init__()
    self.linear = torch.nn.Linear(1,1)
  def forward(self, x):
    out = self.linear(x)
    return out
line_model = LinearModel()
print(line_model.linear.weight.item())
print(line_model.linear.bias.item())

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 4
yb = y + 0.8 * torch.randn(x.size())

plt.plot(x.numpy(), yb.numpy(), 'b+', label='yb')
plt.plot(x.numpy(), y.numpy(), 'r', label='y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid('True', color='y')
plt.show()

learning_rate = 0.001
lossfunc = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(line_model.parameters(), lr = learning_rate)

from math import nan
stop = False
loss_threshold = 0.001
loss_list = []
last_loss = nan
while (stop == False):
    optimizer.zero_grad()
    yh = line_model(x)
    loss = lossfunc(yh, yb)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())
    difference = loss - last_loss
    # print(last_loss, loss)
    # print(difference)
    if (torch.abs(difference) < loss_threshold):
      stop = True
    last_loss = loss
print(line_model.linear.weight.item(), line_model.linear.bias.item())

plt.plot(loss_list, 'r')
plt.tight_layout()
plt.grid('True', color='y')
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")
plt.yscale("log")
plt.show()
