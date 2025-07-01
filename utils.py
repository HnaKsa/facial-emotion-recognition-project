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
    
def get_transforms():
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    
    transform_val = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    return transform_train, transform_val

def get_dataloaders(train_dir, val_dir, batch_size):
    transform_train, transform_val = get_transforms()
    train_data = datasets.ImageFolder(train_dir, transform=transform_train)
    val_data = datasets.ImageFolder(val_dir, transform=transform_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    class_names = train_data.classes
    return train_loader, val_loader, train_data.classes
