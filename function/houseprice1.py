# function:houseprice1.ipynb
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import csv

def ReadCSV(path):
  lines = []  # list of lists
  with open(path, mode ='r') as file:
    fhd = csv.reader(file)
    for line in fhd:
      lineval = [eval(i) for i in line] # convert string to integer
      lines.append(lineval)
  return np.array(lines) # convert to numpy array

from google.colab import drive
drive.mount('/content/drive/')
path = "/content/drive/My Drive/Colab Notebooks/machine learning/Functions/House Prices/"
# trainname = path + "training-small.csv"
trainname = path + "training.csv"
validname = path + "validation.csv"
testname = path + "test.csv"
trainfile = ReadCSV(trainname)
validfile = ReadCSV(validname)
testfile  = ReadCSV(testname)

numrec, numcol = trainfile.shape
train_factors = trainfile[:, 0:numcol - 1]
train_prices = trainfile[:, numcol - 1: numcol]

print(trainfile[0:5])
print(train_factors[0:5])
print(train_prices[0:5])

train_factors = torch.tensor(np.array(train_factors), dtype = torch.float32)
train_prices = torch.tensor(np.array(train_prices), dtype = torch.float32)

class HousePrice(nn.Module):
  def __init__(self, numcol, width1 = 8, width2 = 8):
    super(HousePrice, self).__init__()
    self.degree = numcol
    self.linear1 = torch.nn.Linear(self.degree, width1)
    self.linear2 = torch.nn.Linear(width1, width2)
    self.linear3 = torch.nn.Linear(width2, 1)
  def forward(self, x):
    x = self.linear1(x)
    x = torch.relu(x)
    x = self.linear2(x)
    x = torch.relu(x)
    x = self.linear3(x)
    x = torch.abs(x)
    return x

def Train(model, input, output, learning_rate = 1e-6, loss_threshold = 1e-2, max_iterations = 1e+5):
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
    if((iter % 1000) == 0):
      loss_list.append(loss.item())
    loss.backward()
    optimizer.step()
    if (loss < loss_threshold):
      stop = True
    iter = iter + 1
  return iter, loss_list

pricemodel = HousePrice(numcol - 1)
iterations, loss_list = Train(pricemodel, train_factors, train_prices, loss_threshold= 0.1)
print(iterations)
print(max(loss_list), min(loss_list))

def DataTransform(filepath):
  with open(filepath, mode ='r') as file:
    trainfile = csv.reader(file)
    firstrow = next(trainfile)
    numcol = len(firstrow)
    factors = []
    prices = []
    for line in trainfile:
      inputs = line[0: numcol - 1]
      output = line[numcol - 1: numcol]
      '''
      values = [eval(inputs[0])]
      values.append(eval(inputs[1]))
      values.append(1e-2 * eval(inputs[2]))
      values.append(1e-2 * eval(inputs[3]))
      values.append(1e-3 * eval(inputs[4]))
      '''
      values = [eval(i) for i in inputs] # convert list of strings to list of values
      normalize = [1, 1, 1e-2, 1e-2, 1e-3]
      values = np.multiply(values, normalize)
      factors.append(values)
      value = [1e-6 * eval(i) for i in output] # unit of price: million
      prices.append(value)
  torchfactors = torch.tensor(np.array(factors), dtype = torch.float32)
  torchprices = torch.tensor(np.array(prices), dtype = torch.float32)
  return numcol, torchfactors, torchprices

# draw histogram to check whether there any extreme value
'''
pricelist = train_prices.view(-1, 1)
print(pricelist.shape)
print(max(pricelist), min(pricelist), len(pricelist))
import statistics
print(statistics.stdev(pricelist.view(-1)))
plt.hist(pricelist, 30)
plt.show()
'''

plt.plot(loss_list, 'r')
plt.tight_layout()
plt.grid('True', color='y')
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")
plt.yscale("log")
plt.show()

# print(train_factors)
'''
sampletrain_prices = train_prices[1:1000]
samplemodelprices = pricemodel(train_factors)[1:1000]
# print(validfactors[1:10])
vprices = pricemodel(validfactors)
samplevalidmodel = validprices[1:100]
samplevalidprices = vprices[1:100]

print(sampletrain_prices)
print(samplemodelprices)
print(samplevalidprices)
print(samplevalidmodel)
'''

'''
plt.plot(sampletrain_prices, 'r+')
plt.plot(samplemodelprices.detach().numpy(), 'b+')
plt.plot(samplevalidmodel, 'ko')
plt.plot(samplevalidprices.detach().numpy(), 'mo')
plt.show()
'''

# x-y plot
'''
plt.plot(samplevalidmodel, samplevalidprices.detach().numpy(), 'ro')
plt.plot(sampletrain_prices, samplemodelprices.detach().numpy(), 'b+')
plt.show()
'''

'''
with open(name, mode ='r') as file:
  trainfile = csv.reader(file)
  firstrow = next(trainfile)
  numcol = len(firstrow)
  factors = []
  prices = []
  for line in trainfile:
    inputs = line[0: numcol - 1]
    output = line[numcol - 1: numcol]
    # print(inputs)
    # print(output)
    values = [eval(i) for i in inputs] # convert list of strings to list of values
    factors.append(values)
    value = [1e-6 * eval(i) for i in output] # unit of price: million
    prices.append(value)
# print(firstrow)
# print(type(factors))
# print(factors[0])
# print(prices[0])
train_factors = torch.tensor(np.array(factors), dtype = torch.float32)
train_prices = torch.tensor(np.array(prices), dtype = torch.float32)
# print(type(train_factors))
'''
