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
from utils import get_dataloaders , get_transforms , train_loader , val_loader , val_data , train_data , class_names 
from config import batch_size , epochs , lr , train_dir , val_dir
from resnet_model import EmotionResNet34
from efficientnet_model import EmotionEfficientNetB0
from model_setup import raw_dataset , labels , label_counts , num_classes , class_counts , class_weights , device , model , criterion , optimizer , scheduler , train_accuracies , train_losses , val_accuracies , val_losses 

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Use the last convolutional layer name for each model
target_layers = [model.base_model.layer4[-1]]  # For ResNet34

# Prepare the image
img = Image.open("/content/fer2013_data/test/sad/PrivateTest_62176553.jpg")
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])
img_transformed = transform(img).unsqueeze(0).to(device)
rgb_img = np.stack([np.array(img.resize((224, 224)))]*3, axis=-1) / 255.0

# Get the prediction from the model
model.eval()
with torch.no_grad():
    output = model(img_transformed)
    pred = torch.argmax(output, 1)

# Apply Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=img_transformed, targets=[ClassifierOutputTarget(pred.item())])[0]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.axis('off')
plt.title("Grad-CAM Heatmap")
plt.show()