import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 설정
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
NUM_CLASSES = 27
IMG_SIZE = 224
DATA_DIR = "dataset"  # dataset/train, dataset/val 구조

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 데이터 전처리
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ✅ 모델 구성
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # 마지막 FC만 교체
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ✅ 학습 & 검증 루프
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0, 0

    for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (output.argmax(1) == y).sum().item()

    train_acc = train_correct / len(train_loader.dataset)

    # 검증
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            val_correct += (output.argmax(1) == y).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "resnet_finetuned_best.pth")

print("✅ Fine-tuning 완료. 최고 검증 정확도:", round(best_acc * 100, 2), "%")
