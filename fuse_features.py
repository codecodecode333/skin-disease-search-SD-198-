import os
import numpy as np
from tqdm import tqdm

# 경로 설정
cnn_dir = "features/cnn"
orb_dir = "features/orb"
sift_dir = "features/sift"
out_dir = "features/fused"
phases = ['train', 'val']

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

for phase in phases:
    for class_name in os.listdir(os.path.join(cnn_dir, phase)):
        cnn_class_path = os.path.join(cnn_dir, phase, class_name)
        orb_class_path = os.path.join(orb_dir, phase, class_name)
        sift_class_path = os.path.join(sift_dir, phase, class_name)
        save_class_path = os.path.join(out_dir, phase, class_name)
        os.makedirs(save_class_path, exist_ok=True)

        for file in tqdm(os.listdir(cnn_class_path), desc=f"[{phase}/{class_name}] 병합 중"):
            base_name = os.path.splitext(file)[0]

            cnn_path = os.path.join(cnn_class_path, file)
            orb_path = os.path.join(orb_class_path, file)
            sift_path = os.path.join(sift_class_path, file)

            if not (os.path.exists(cnn_path) and os.path.exists(orb_path) and os.path.exists(sift_path)):
                continue

            cnn_vec = np.load(cnn_path).astype(np.float32)
            orb_vec = np.load(orb_path).astype(np.float32)
            sift_vec = np.load(sift_path).astype(np.float32)

            # 정규화
            cnn_vec = normalize(cnn_vec)
            orb_vec = normalize(orb_vec)
            sift_vec = normalize(sift_vec)

            # 병합
            fused_vec = np.concatenate([2*cnn_vec, orb_vec, sift_vec])

            # 저장
            save_path = os.path.join(save_class_path, base_name + ".npy")
            np.save(save_path, fused_vec)
