import torch.nn as nn
from collections import OrderedDict

from .GatedLinear import GatedLinear
from .GatedConv2d import GatedConv2d


class C1(nn.Module):
    def __init__(self, gated: bool = False):
        super(C1, self).__init__()

        Conv = GatedConv2d if gated else nn.Conv2d

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', Conv(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self, gated: bool = False):
        super(C2, self).__init__()

        Conv = GatedConv2d if gated else nn.Conv2d

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', Conv(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self, gated: bool = False):
        super(C3, self).__init__()

        Conv = GatedConv2d if gated else nn.Conv2d

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', Conv(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4(nn.Module):
    def __init__(self, gated: bool = False):
        super(F4, self).__init__()

        Linear = GatedLinear if gated else nn.Linear

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self, gated: bool = False):
        super(F5, self).__init__()

        Linear = GatedLinear if gated else nn.Linear

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self, gated: bool = False):
        super(LeNet5, self).__init__()

        self.gated: bool = gated
        self.c1 = C1(gated)
        self.c2_1 = C2(gated)
        self.c2_2 = C2(gated)
        self.c3 = C3(gated)
        self.f4 = F4(gated)
        self.f5 = F5(gated)

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output