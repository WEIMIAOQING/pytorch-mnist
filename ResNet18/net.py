import torch
from torch import nn 
import torch.nn.functional as F

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        # self.c1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.c1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.c4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.c5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.c6_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn6_1 = nn.BatchNorm2d(128)
        self.c7_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0) #虚线部分
        self.bn7 = nn.BatchNorm2d(128)

        self.c8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.c9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(128)

        self.c10_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn10_1 = nn.BatchNorm2d(256)
        self.c11_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c11 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0) #虚线部分
        self.bn11 = nn.BatchNorm2d(256)

        self.c12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.c13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(256)

        self.c14_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn14_1 = nn.BatchNorm2d(512)
        self.c15_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.c15 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0) #虚线部分
        self.bn15 = nn.BatchNorm2d(512)

        self.c16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(512)
        self.c17 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn17 = nn.BatchNorm2d(512)

        self.s2 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0) #平均池化层
        self.flatten = nn.Flatten()
        self.f18 = nn.Linear(512,1000)
        self.output = nn.Linear(1000,10)

    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.s1(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.c4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.c5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x6_1 = self.c6_1(x)
        x6_1 = F.relu(x6_1)
        x7_1 = self.c7_1(x6_1)
        x7 = self.c7(x)
        x = x7 + x7_1
        x = self.bn7(x)
        x = F.relu(x)

        x = self.c8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.c9(x)
        x = self.bn9(x)
        x = F.relu(x)

        x10_1 = self.c10_1(x)
        x10_1 = F.relu(x10_1)
        x11_1 = self.c11_1(x10_1)
        x11 = self.c11(x)
        x = x11 + x11_1
        x = self.bn11(x)
        x = F.relu(x)

        x = self.c12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.c13(x)
        x = self.bn13(x)
        x = F.relu(x)

        x14_1 = self.c14_1(x)
        x14_1 = F.relu(x14_1)
        x15_1 = self.c15_1(x14_1)
        x15 = self.c15(x)
        x = x15 + x15_1
        x = self.bn15(x)
        x = F.relu(x)

        x = self.c16(x)
        x = self.bn16(x)
        x = F.relu(x)
        x = self.c17(x)
        x = self.bn17(x)
        x = F.relu(x)

        x = self.s2(x)
        x = self.flatten(x)
        x = self.f18(x)
        x = F.relu(x)

        x = self.output(x)
        x = F.softmax(x, dim=1)
        return x


    


















