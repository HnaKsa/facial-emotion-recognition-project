from models import model_setup , resnet_model , efficientnet_model
from utils import *

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