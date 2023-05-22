import torch
import torch.nn as nn

"""

efficient_net.py : Model
dataloader.py: Dataloader
training.py: Training Script
"""

class MBConv(nn.Module):

    def __init__(self, size, s, channels, out_channels, layers):
        super(MBConv, self).__init__()

        pad = int(size / 2) 
        self.layers = layers
        self.layer1 = nn.Conv2d(channels, 6*channels, 1)
        self.dwise1 = nn.Conv2d(6*channels, 6*channels, size, stride=1, groups=6*channels, padding=pad)
        self.dwise = nn.Conv2d(6*channels, 6*channels, size, stride=s, groups=6*channels, padding=pad)
        self.layer3 = nn.Conv2d(6*channels, out_channels, 1)

    def forward(self, x):
        r = self.layer1(x)
        r = nn.ReLU(6)(r)
        r = self.dwise(r)
        r = nn.ReLU(6)(r)
        for i in range(self.layers - 1):
            r = self.dwise1(r)
            r = nn.ReLU(6)(r)
        r = self.layer3(r)
        print(r.shape)
        print(x.shape)
        #return r + x
        return r




class EfficientNetB0(nn.Module):

    def __init__(self):
        super(EfficientNetB0, self).__init__()

        self.c1 = nn.Conv2d(3, 32, 3, 2, padding=1)
        self.mb1 = MBConv(3, 1, 32, 16, 1)
        self.mb2 = MBConv(3, 2, 16, 24, 2)
        self.mb3 = MBConv(5, 2, 24, 40, 2)
        self.mb4 = MBConv(3, 2, 40, 80, 3)
        self.mb5 = MBConv(5, 1, 80, 112, 3)
        self.mb6 = MBConv(5, 2, 112, 192, 4)
        self.mb7 = MBConv(3, 1, 192, 320, 1)


    def forward(self, x):
        r = self.c1(x)
        r = nn.ReLU(6)(r)
        r = self.mb1(r)
        r = self.mb2(r)
        r = self.mb3(r)
        r = self.mb4(r)
        r = self.mb5(r)
        r = self.mb6(r)
        r = self.mb7(r)
        return r
