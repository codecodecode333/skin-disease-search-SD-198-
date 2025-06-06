import os
import cv2
import numpy as np
from tqdm import tqdm

# 설정
IMAGE_DIR = "dataset"  # dataset/train/<class>/<img>.jpg
OUT_DIR = "features/orb"
VECTOR_DIM = 1024  # 고정된 벡터 차원 수

phases = ['train', 'val']

# ORB 생성기 (기본: 500개 keypoint)
orb = cv2.ORB_create(nfeatures=500)

for phase in phases:
    for class_name in os.listdir(os.path.join(IMAGE_DIR, phase)):
        class_dir = os.path.join(IMAGE_DIR, phase, class_name)
        if not os.path.isdir(class_dir):
            continue

        save_dir = os.path.join(OUT_DIR, phase, class_name)
        os.makedirs(save_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(class_dir), desc=f"[{phase}/{class_name}] ORB 추출"):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            keypoints, descriptors = orb.detectAndCompute(image, None)

            if descriptors is None:
                descriptors = np.zeros((1, 32), dtype=np.uint8)

            # (N, 32) → 1D 벡터
            desc_vector = descriptors.flatten()

            # 고정 길이 맞추기
            if desc_vector.shape[0] < VECTOR_DIM:
                padded = np.zeros(VECTOR_DIM, dtype=np.uint8)
                padded[:desc_vector.shape[0]] = desc_vector
                desc_vector = padded
            else:
                desc_vector = desc_vector[:VECTOR_DIM]

            # 정규화 (0~1)
            desc_vector = desc_vector.astype(np.float32)
            desc_vector /= 255.0

            # 저장
            base_name = os.path.splitext(img_name)[0]
            out_path = os.path.join(save_dir, base_name + ".npy")
            np.save(out_path, desc_vector)
