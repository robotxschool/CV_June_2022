import torch
from torch import nn
import numpy as np

class Net(nn.Module):
    def __init__(self, classes_count=2):
        super(Net, self).__init__()
        #self.quant = torch.quantization.QuantStub()
        #self.conv_quantized = torch.conv2d(self.quant, 1)
        self.classes_count = classes_count
        self.encode1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.encode2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.encode3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.encode4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.encode5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decode4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.decode5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.classifier = nn.Conv2d(16, 2, kernel_size=1)
        print(self.classifier)
        #self.dequant = torch.quantization.DeQuantStub()



    def forward(self, x):           
        out = self.encode1(x)       
        out = self.encode2(out)     
        out = self.encode3(out)     
        out = self.encode4(out)
        out = self.encode5(out) 
        out = self.decode1(out)     
        out = self.decode2(out)     
        out = self.decode3(out)     
        out = self.decode4(out)     
        out = self.decode5(out)
        out = self.classifier(out)  
        return out


if __name__ == '__main__':
    print("Shape:")
    img = torch.randn(2, 3, 224, 224)
    net = Net()
    sample = net(img)
    sample = sample.cpu().data.numpy().copy()
    print(sample)
    #print(img)
    print(sample.shape)
