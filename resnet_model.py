import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class EmotionResNet34(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionResNet34, self).__init__()
        self.base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
