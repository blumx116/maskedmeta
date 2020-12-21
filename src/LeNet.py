import torch.nn as nn
from collections import OrderedDict

import torch

from .GatedLinearNew import GatedLinear
from .GatedConv2dNew import GatedConv2d

class transform_grayscale_to_RGB:
    def __init__(self, color):
        self.color = color
    def __call__(self, pic):
        return torch.stack([pic[0]+(1-pic[0])*self.color[i] for i in range(3)])
    def __repr__(self):
        return 'grayscale_to_RGB()'

class C1(nn.Module):
    def __init__(self, gated: bool = False, n_tasks: int = 1):
        super(C1, self).__init__()

        Conv = GatedConv2d if gated else nn.Conv2d
        if gated:
            self.c1 = nn.Sequential(OrderedDict([
                ('c1', Conv(3, 6, kernel_size=(5, 5), n_tasks=n_tasks)),
                ('relu1', nn.ReLU()),
                ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            ]))
        else:
            self.c1 = nn.Sequential(OrderedDict([
                ('c1', Conv(3, 6, kernel_size=(5, 5))),
                ('relu1', nn.ReLU()),
                ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            ]))

    def forward(self, img):
        output = self.c1(img)
        return output

    def set_task(self, task_idx: int) -> None:
        self.c1[0].set_task(task_idx)

    def shared_parameters(self):
        return self.c1[0].shared_parameters()

    def task_parameters_for(self, task_idx):
        return self.c1[0].task_parameters_for(task_idx)


class C2(nn.Module):
    def __init__(self, gated: bool = False, n_tasks: int = 1):
        super(C2, self).__init__()

        Conv = GatedConv2d if gated else nn.Conv2d

        if gated:
            self.c2 = nn.Sequential(OrderedDict([
                ('c2', Conv(6, 16, kernel_size=(5, 5), n_tasks=n_tasks)),
                ('relu2', nn.ReLU()),
                ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            ]))
        else:
            self.c2 = nn.Sequential(OrderedDict([
                ('c2', Conv(6, 16, kernel_size=(5, 5))),
                ('relu2', nn.ReLU()),
                ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            ]))

    def forward(self, img):
        output = self.c2(img)
        return output

    def set_task(self, task_idx: int) -> None:
        self.c2[0].set_task(task_idx)

    def shared_parameters(self):
        return self.c2[0].shared_parameters()

    def task_parameters_for(self, task_idx):
        return self.c2[0].task_parameters_for(task_idx)


class C3(nn.Module):
    def __init__(self, gated: bool = False, n_tasks: int = 1):
        super(C3, self).__init__()

        Conv = GatedConv2d if gated else nn.Conv2d

        if gated:
            self.c3 = nn.Sequential(OrderedDict([
                ('c3', Conv(16, 120, kernel_size=(5, 5), n_tasks=n_tasks)),
                ('relu3', nn.ReLU())
            ]))
        else:
            self.c3 = nn.Sequential(OrderedDict([
                ('c3', Conv(16, 120, kernel_size=(5, 5))),
                ('relu3', nn.ReLU())
            ]))

    def forward(self, img):
        output = self.c3(img)
        return output

    def set_task(self, task_idx: int) -> None:
        self.c3[0].set_task(task_idx)

    def shared_parameters(self):
        return self.c3[0].shared_parameters()

    def task_parameters_for(self, task_idx):
        return self.c3[0].task_parameters_for(task_idx)

class F4(nn.Module):
    def __init__(self, gated: bool = False, n_tasks: int = 1):
        super(F4, self).__init__()

        Linear = GatedLinear if gated else nn.Linear

        if gated:
            self.f4 = nn.Sequential(OrderedDict([
                ('f4', Linear(120, 84, n_tasks=n_tasks)),
                ('relu4', nn.ReLU())
            ]))
        else:
            self.f4 = nn.Sequential(OrderedDict([
                ('f4', Linear(120, 84)),
                ('relu4', nn.ReLU())
            ]))

    def forward(self, img):
        output = self.f4(img)
        return output

    def set_task(self, task_idx: int) -> None:
        self.f4[0].set_task(task_idx)

    def shared_parameters(self):
        return self.f4[0].shared_parameters()

    def task_parameters_for(self, task_idx):
        return self.f4[0].task_parameters_for(task_idx)

class F5(nn.Module):
    def __init__(self, gated: bool = False, n_tasks: int = 1):
        super(F5, self).__init__()

        Linear = GatedLinear if gated else nn.Linear

        if gated:
            self.f5 = nn.Sequential(OrderedDict([
                ('f5', Linear(84, 10, n_tasks=n_tasks)),
                ('sig5', nn.LogSoftmax(dim=-1))
            ]))
        else:
            self.f5 = nn.Sequential(OrderedDict([
                ('f5', Linear(84, 10)),
                ('sig5', nn.LogSoftmax(dim=-1))
            ]))

    def forward(self, img):
        output = self.f5(img)
        return output

    def set_task(self, task_idx: int) -> None:
        self.f5[0].set_task(task_idx)

    def shared_parameters(self):
        return self.f5[0].shared_parameters()

    def task_parameters_for(self, task_idx):
        return self.f5[0].task_parameters_for(task_idx)


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self, gated: bool = False, n_tasks: int = 1):
        super(LeNet5, self).__init__()

        self.gated: bool = gated
        self.c1 = C1(gated, n_tasks)
        self.c2_1 = C2(gated, n_tasks)
        self.c2_2 = C2(gated, n_tasks)
        self.c3 = C3(gated, n_tasks)
        self.f4 = F4(gated, n_tasks)
        self.f5 = F5(gated, n_tasks)

        self.n_tasks = n_tasks
        self.cur_task_idx: Optional[int] = None

    def isGated(self):
        return self.gated

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

    def set_task(self,
                 task_idx: int) -> None:
        assert 0 <= task_idx < self.n_tasks
        self.cur_task_idx = task_idx
        if(self.gated):
            self.c1.set_task(task_idx)
            self.c2_1.set_task(task_idx)
            self.c2_2.set_task(task_idx)
            self.c3.set_task(task_idx)
            self.f4.set_task(task_idx)
            self.f5.set_task(task_idx)

    def shared_parameters(self) -> nn.ParameterList:
        result: nn.ParameterList = nn.ParameterList()
        if(self.gated):
            result.extend(self.c1.shared_parameters())
            result.extend(self.c2_1.shared_parameters())
            result.extend(self.c2_2.shared_parameters())
            result.extend(self.c3.shared_parameters())
            result.extend(self.f4.shared_parameters())
            result.extend(self.f5.shared_parameters())
        return result

    def task_parameters_for(self,
                            task_idx: int) -> nn.ParameterList:
        assert task_idx < self.n_tasks
        result: nn.ParameterList = nn.ParameterList()
        if(self.gated):
            result.extend(self.c1.task_parameters_for(task_idx))
            result.extend(self.c2_1.task_parameters_for(task_idx))
            result.extend(self.c2_2.task_parameters_for(task_idx))
            result.extend(self.c3.task_parameters_for(task_idx))
            result.extend(self.f4.task_parameters_for(task_idx))
            result.extend(self.f5.task_parameters_for(task_idx))
        return result