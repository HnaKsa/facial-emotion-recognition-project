"""facial-emotion-recognition-cnn-3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AMel8mOvwGpPEcSJPozAqa7_S_HVhLJ6
"""

# from google.colab import files
# files.upload()

# import os
# import zipfile
# os.makedirs('/root/.kaggle', exist_ok=True)
# !mv kaggle.json /root/.kaggle/
# !chmod 600 /root/.kaggle/kaggle.json

# !kaggle datasets download -d msambare/fer2013

# !unzip -q fer2013.zip -d fer2013_data

"""# Transfer Learning with ResNet18
We define a custom model EmotionResNet using a pre-trained ResNet18 as the backbone.
Modifications:

The first convolutional layer is adjusted to accept grayscale images.

The final fully connected layer is replaced to output predictions for 7 emotion classes.


"""

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
# from torchvision.models import resnet18, ResNet18_Weights

# class EmotionResNet(nn.Module):
#     def __init__(self, num_classes=7):
#         super(EmotionResNet, self).__init__()
#         weights = ResNet18_Weights.DEFAULT  # مطابق با هش صحیح
#         self.base_model = resnet18(weights=weights)
#         self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # برای grayscale
#         self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.base_model(x)

from torchvision.models import resnet34, ResNet34_Weights, efficientnet_b0, EfficientNet_B0_Weights

# ResNet34
class EmotionResNet34(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionResNet34, self).__init__()
        self.base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# EfficientNetB0
class EmotionEfficientNetB0(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionEfficientNetB0, self).__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

"""# Parameters
Define key hyperparameters and dataset paths:

batch_size: Size of data batches during training.

epochs: Number of training iterations.

lr: Learning rate for the optimizer.

train_dir, val_dir: Paths to the training and validation image folders.
"""

# 🔧 Parameters

batch_size = 64
epochs = 20
lr = 0.0001
train_dir = "/content/fer2013_data/train"
val_dir = "/content/fer2013_data/test"

"""# Transforms
Image preprocessing steps:

Training transform includes:

Grayscale conversion

Resize to 224x224

Random horizontal flip

Random rotation

Normalization (mean = 0.485, std = 0.229)

Validation transform is similar but without augmentations for consistent evaluation.
"""

# 📦 Transforms

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

"""# Data Loaders
Use ImageFolder to load labeled images from directories and apply the transforms.
DataLoader wraps the datasets to:

Enable batch processing

Shuffle training data for better generalization

"""

# 📁 Data Loaders

train_data = datasets.ImageFolder(train_dir, transform=transform_train)
val_data = datasets.ImageFolder(val_dir, transform=transform_val)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
class_names = train_data.classes

"""# Model Setup
Initialize and prepare the model:

Move model to GPU (cuda) if available

Use CrossEntropyLoss for multi-class classification

Use Adam optimizer

Apply a learning rate scheduler (StepLR) to reduce the learning rate every 7 epochs
"""

from collections import Counter
import os
from torchvision import datasets


raw_dataset = datasets.ImageFolder(train_dir)
labels = [label for _, label in raw_dataset.imgs]

label_counts = Counter(labels)
num_classes = len(raw_dataset.classes)


class_counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

# 🧠 Model Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionResNet34(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for idx, class_name in enumerate(raw_dataset.classes):
    print(f"{class_name}: {label_counts[idx]} samples")

for i, (cls, weight) in enumerate(zip(raw_dataset.classes, class_weights)):
    print(f"کلاس '{cls}' (index {i}): وزن = {weight:.4f}")

print(dict(zip(raw_dataset.classes, class_counts)))

"""# Training Loop
Train the model for the defined number of epochs:

Enable training mode with model.train()

Forward pass, compute loss, backpropagate, and update weights

Track total loss and accuracy

Print epoch-wise metrics

Adjust learning rate using the scheduler


"""

for epoch in range(epochs):
    # 🔁 Phase 1: Training
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100. * correct / total
    train_losses.append(total_loss)
    train_accuracies.append(train_accuracy)

    # 📉 Phase 2: Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)

            val_loss += loss.item()
            preds = output.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100. * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    scheduler.step()

    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f}, Acc: {train_accuracy:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%")

"""# Save Model"""

# 💾 Save Model

torch.save(model.state_dict(), "emotion_model_resnet34.pth")
print("Model saved as emotion_model_resnet34.pth")

"""# Training Curves
Plot and visualize:

Training loss over epochs

Training accuracy over epochs

These curves help monitor model learning and detect overfitting or underfitting.
"""

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

# 📊 Training Curves for test dataset

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

"""# Evaluation
Evaluate the model performance on validation data:

Disable gradient computation with torch.no_grad()

Predict classes for validation set

Show a classification report (precision, recall, F1-score)

Display a confusion matrix to visualize prediction errors per class
"""

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

"""# Predict Single Image
Use a trained model to predict emotion from a single grayscale face image:

Apply the same preprocessing steps

Perform inference with the model

Output the predicted class label using argmax


"""

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
img = Image.open("/content/fer2013_data/test/sad/PrivateTest_13994896.jpg")
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

