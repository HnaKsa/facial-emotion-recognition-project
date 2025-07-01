import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights, efficientnet_b0, EfficientNet_B0_Weights

# EfficientNetB0
class EmotionEfficientNetB0(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionEfficientNetB0, self).__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)