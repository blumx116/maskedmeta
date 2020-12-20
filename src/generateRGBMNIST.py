import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim, autograd
import matplotlib.pyplot as plt
import torchvision
from time import time


sizetransforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

train_mnist = datasets.MNIST('~/datasets/mnist', transform=sizetransforms, train=True, download=True)
test_mnist = datasets.MNIST('~/datasets/mnist', transform=sizetransforms, train=False, download=True)

color = [0, 0, 0]  # background color replacement
colorName = 'black'
# Note: all r-g-b colors should be defined from [0,1]
print("processing images to background color "+colorName)

coloredData = []
for im in range(len(train_mnist)):
    x, y = train_mnist[im]

    origIm = x[0].numpy()

    converted = np.zeros((32, 32, 3))
    for c in range(3):
        converted[:, :, c] = origIm[:, :] + (1 - origIm[:, :]) * color[c]

    converted = torch.from_numpy(converted).permute(2, 0, 1)
    coloredData.append((converted, y))

print('done processing training data')

coloredTest = []
for im in range(len(test_mnist)):
    x, y = test_mnist[im]

    origIm = x[0].numpy()

    converted = np.zeros((32, 32, 3))
    for c in range(3):
        converted[:, :, c] = origIm[:, :] + (1 - origIm[:, :]) * color[c]

    converted = torch.from_numpy(converted).permute(2, 0, 1)
    coloredTest.append((converted, y))

print('done processing test data')

train_x: torch.Tensor = torch.stack([x for x, y in coloredData], dim=0)
train_y: torch.Tensor = torch.stack([torch.from_numpy(np.asarray(y)) for x, y in coloredData], dim=0).type(torch.LongTensor)

test_x: torch.Tensor = torch.stack([x for x, y in coloredTest], dim=0)
test_y: torch.Tensor = torch.stack([torch.from_numpy(np.asarray(y)) for x, y in coloredTest], dim=0).type(torch.LongTensor)

print('done stacking all data')

x = train_x[7777]
plt.figure()
plt.imshow(np.transpose(x.numpy(), (1, 2, 0)))
plt.show()
print(train_y[7777])

print(train_x.size())
print(train_y.size())
print(test_x.size())
print(test_y.size())

torch.save(train_x, 'train_'+colorName+'_x.pt')
torch.save(train_y, 'train_'+colorName+'_y.pt')
torch.save(test_x, 'test_'+colorName+'_x.pt')
torch.save(test_x, 'test_'+colorName+'_y.pt')