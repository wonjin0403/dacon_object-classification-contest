from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from timm import create_model
import torchvision.models as models

class RESNET18(nn.Module):
    def __init__(self, out_channels):
        super(RESNET18, self).__init__()
        self.res18 = models.resnet18(pretrained=False)
        self.res18.fc = nn.Linear(in_features=self.res18.fc.in_features, out_features=out_channels)
        self.feature1 = nn.Sequential(*(list(self.res18.children())[0:8]))
        self.feature2 = nn.Sequential(list(self.res18.children())[8])
        self.feature3 = nn.Sequential(list(self.res18.children())[9])

    def forward(self, x):
        map = self.feature1(x)
        h1 = self.feature2(map)
        output = self.feature3(h1.reshape(h1.shape[0], -1))
        return output