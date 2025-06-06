import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 설정
IMG_SIZE = 224
BATCH_SIZE = 1
DATA_DIR = "dataset"  # dataset/train, dataset/val
OUT_DIR = "features/cnn"
MODEL_PATH = "resnet_finetuned_best.pth"
NUM_CLASSES = 27  # 클래스 수 일치하게!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 전처리 (학습 시 사용한 것과 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 데이터셋 로드
phases = ['train', 'val']
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), transform=transform) for x in phases}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=False) for x in phases}

# ✅ Fine-tuned 모델 로드 + FC 제거
class Identity(nn.Module):
    def forward(self, x):
        return x

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.fc = Identity()  # FC 레이어 제거 → 특징 추출기로 변환
model = model.to(device)
model.eval()

# ✅ 특징 추출 및 저장
with torch.no_grad():
    for phase in phases:
        for idx, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), desc=f"[{phase}] Extracting CNN features"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze().cpu().numpy()

            path, _ = image_datasets[phase].imgs[idx]
            class_name = os.path.basename(os.path.dirname(path))
            img_name = os.path.splitext(os.path.basename(path))[0]

            save_dir = os.path.join(OUT_DIR, phase, class_name)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, img_name + ".npy"), outputs)
