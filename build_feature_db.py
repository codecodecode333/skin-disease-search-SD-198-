import os
import numpy as np
import pickle
from tqdm import tqdm

# 설정
FEATURE_DIR = "features/fused/train"  # 병합 특징 벡터 디렉토리
IMAGE_DIR = "raw_images"                  # 원본 이미지 디렉토리
OUTPUT_PATH = "fused_db.pkl"         # 저장 경로

db = []
skipped = 0

for class_name in tqdm(os.listdir(FEATURE_DIR), desc="DB 생성 중"):
    feat_class_dir = os.path.join(FEATURE_DIR, class_name)
    img_class_dir = os.path.join(IMAGE_DIR, class_name)

    if not os.path.isdir(feat_class_dir):
        continue

    for fname in os.listdir(feat_class_dir):
        if not fname.endswith(".npy"):
            continue

        npy_path = os.path.join(feat_class_dir, fname)
        image_name = fname.replace(".npy", ".jpg")
        img_path = os.path.join(img_class_dir, image_name)

        if not os.path.exists(img_path):
            skipped += 1
            continue  # 매칭되는 이미지 없으면 건너뜀

        try:
            feature = np.load(npy_path).astype(np.float32)
        except Exception as e:
            print(f"❌ 오류 발생: {npy_path} → {e}")
            skipped += 1
            continue

        if feature.ndim != 1:
            print(f"❌ 벡터 차원 오류: {npy_path} → {feature.shape}")
            skipped += 1
            continue

        db.append({
            "feature": feature,
            "path": img_path,          # ✅ 여기 수정
            "class": class_name
        })

# DB 저장
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(db, f)

print(f"✅ DB 생성 완료: {len(db)}개 항목 저장됨")
print(f"🚫 누락된 항목: {skipped}개")
