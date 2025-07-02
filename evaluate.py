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

# 📊 Training Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='green')
plt.title('Training Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curves_resnet.png")
plt.show()

# 📊 Curves for test dataset
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(val_losses, label='Test Loss')
plt.title('Testing Loss')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Test Accuracy', color='green')
plt.title('Testing Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig("testing_curves_resnet.png")
plt.show()

# 📌 Evaluation

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        output = model(imgs)
        preds = output.argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix_resnet.png")
plt.show()