import torch.nn as nn
import torch.nn.functional as F
import torch 
from torchvision import transforms
from PIL import Image
import time
#from google.colab.patches import cv2_imshow


class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()

        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, 2)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))


test_array = ["data/test/45/1655720177.742964.jpg", "data/test/90/1655720423.33968.jpg"]

model = torch.load("m.p", map_location=torch.device("cpu"))
model.eval()
#model.cuda()

transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor()])


classes = {1: "45 deg angle rotated", 0: "90 deg angle rotated"}
for path in test_array:
  img = Image.open(path)
  #display(img)
  img = transforms_test(img).unsqueeze(0)#.to(torch.device("cuda"))
  
  time_start = time.time()
  out = model(img)
  print("FPS:", 1 / (time.time() - time_start))
  _, predicted = torch.max(out.data, 1)
  print("Raw out:", predicted)
  final = classes[predicted.cpu().numpy()[0]]
  print("Processed:", final)
