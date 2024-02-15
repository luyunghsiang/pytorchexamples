# function:highdegree.py
import torch
import numpy as np
import matplotlib.pyplot as plt

class PolyModel(torch.nn.Module):
  def __init__(self, degree):
    super(PolyModel, self).__init__()
    self.degree = degree
    self.linear = torch.nn.Linear(self.degree, 1)
  def forward(self, x):
    vec = torch.cat([x ** i for i in range(1, self.degree + 1)], dim=-1)
    out = self.linear(vec)
    return out

x = torch.arange(-5, 5, 0.2, dtype=torch.float32).view(-1, 1)
def cubic(x):
  return 6 - 4 * x + 3 * (x ** 2) + 2.5 * (x ** 3) 
y3 = cubic(x)
yb3 = y3 + torch.randn(x.size())
print(x.shape, y3.shape)

cubic_model = PolyModel(3)
print(cubic_model.linear.weight)
print(cubic_model.linear.bias)

learning_rate = 1e-6
lossfunc = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(cubic_model.parameters(), lr = learning_rate)
num_iteration = 10000
loss_list = []
for i in range (num_iteration):
    yh3 = cubic_model(x)
    optimizer.zero_grad()
    loss = lossfunc(yh3, yb3)
    loss.backward()
    optimizer.step()
    if ((i % 100) == 10):
      loss_list.append(loss.item())
print(cubic_model.linear.bias)
print(cubic_model.linear.weight)

plt.plot(loss_list, 'r')
plt.tight_layout()
plt.grid('True', color='y')
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")
plt.yscale("log")
plt.show()

plt.plot(x.numpy(), y3.numpy(), 'r', label='y')
with torch.no_grad():
  yh3 = cubic_model(x)
plt.plot(x.numpy(), yh3.numpy(), 'k+', label='yh')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid('True', color='y')
plt.show()
