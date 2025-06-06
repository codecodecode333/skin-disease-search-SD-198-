import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# === 데이터셋 클래스 ===
class FusedFeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.class_to_idx = {}
        for i, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = i
            for file in os.listdir(class_path):
                if file.endswith(".npy"):
                    self.samples.append(os.path.join(class_path, file))
                    self.labels.append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature = np.load(self.samples[idx]).astype(np.float32)
        label = self.labels[idx]
        return torch.tensor(feature), label

# === 모델 정의 ===
class FusionClassifier(nn.Module):
    def __init__(self, input_dim=5120, num_classes=27):
        super(FusionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# === 학습 함수 ===
def train_model(train_dir, val_dir, num_epochs=30, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = FusedFeatureDataset(train_dir)
    val_dataset = FusedFeatureDataset(val_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FusionClassifier(num_classes=27).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 검증
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "fusion_model_best.pth")

    print(f"✅ 학습 완료! 최고 검증 정확도: {best_acc:.2f}%")

# === 실행 ===
if __name__ == "__main__":
    train_model(
        train_dir="features/fused/train",
        val_dir="features/fused/val",
        num_epochs=30,
        batch_size=32,
        lr=1e-4
    )
