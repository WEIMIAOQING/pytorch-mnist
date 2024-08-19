import torch
from torch import nn 
import torch.nn.functional as F

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.s1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1, padding=1)
        self.c4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,stride=1, padding=1)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1, padding=1)
        self.c6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.c9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.c10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.c12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.c13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.f14 = nn.Linear(7*7*512,4096)
        self.f15 = nn.Linear(4096,4096)
        self.f16 = nn.Linear(4096,1000)
        self.output = nn.Linear(1000,10)

    def forward(self, x):
        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(self.s1(x))

        x = self.c3(x)
        x = F.relu(x)
        x = self.c4(x)
        x = F.relu(self.s2(x))

        x = self.c5(x)
        x = F.relu(x)
        x = self.c6(x)
        x = F.relu(x)
        x = self.c7(x)
        x = F.relu(self.s3(x))

        x = self.c8(x)
        x = F.relu(x)
        x = self.c9(x)
        x = F.relu(x)
        x = self.c10(x)
        x = F.relu(self.s4(x))

        x = self.c11(x)
        x = F.relu(x)
        x = self.c12(x)
        x = F.relu(x)
        x = self.c13(x)
        x = F.relu(self.s5(x))

        x = self.flatten(x)
        x = self.f14(x)
        x = F.relu(x)
        x = self.f15(x)
        x = F.relu(x)
        x = self.f16(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)
        return x
    





        

        



