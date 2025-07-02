import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import numpy as np
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import Counter
from torchvision import datasets
from utils import get_dataloaders , get_transforms 
from config import batch_size , epochs , lr , train_dir , val_dir
from resnet_model import EmotionResNet34
from efficientnet_model import EmotionEfficientNetB0
from model_setup import raw_dataset , labels , label_counts , num_classes , class_counts , class_weights , device , model , criterion , optimizer , scheduler , train_accuracies , train_losses , val_accuracies , val_losses 
train_loader, val_loader, class_names, train_data, val_data = get_dataloaders()
# 🔍 Predict Single Image

img = Image.open(r"/content/fer2013_data/test/sad/PrivateTest_62176553.jpg")
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])
img_tensor = transform(img).unsqueeze(0).to(device)
model.eval()

with torch.no_grad():
    output = model(img_tensor)
    pred = torch.argmax(output, 1)
    print(f"Predicted Emotion: {class_names[pred.item()]}")