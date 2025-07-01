from collections import Counter
import os
from torchvision import datasets
from models import resnet_model , efficientnet_model 
from utils import *
# بارگذاری dataset بدون transform برای شمردن تعداد نمونه‌ها
raw_dataset = datasets.ImageFolder(train_dir)
labels = [label for _, label in raw_dataset.imgs]

# شمارش تعداد نمونه در هر کلاس
label_counts = Counter(labels)
num_classes = len(raw_dataset.classes)

# تبدیل به Tensor و محاسبه وزن معکوس
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