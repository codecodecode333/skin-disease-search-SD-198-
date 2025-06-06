import os
import shutil
import random

SRC_DIR = "processed_images"     # 전처리된 이미지 폴더
OUT_DIR = "dataset"              # train/val 나뉘는 최종 저장 폴더
TRAIN_RATIO = 0.8                # 학습 데이터 비율

def make_dirs():
    for split in ['train', 'val']:
        split_path = os.path.join(OUT_DIR, split)
        os.makedirs(split_path, exist_ok=True)

def split_data():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    make_dirs()

    for class_name in os.listdir(SRC_DIR):
        class_dir = os.path.join(SRC_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)
        random.shuffle(images)

        split_idx = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for split, img_list in zip(['train', 'val'], [train_imgs, val_imgs]):
            split_class_dir = os.path.join(OUT_DIR, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img in img_list:
                src = os.path.join(class_dir, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy2(src, dst)

    print("✅ train/val 데이터 분할 완료! 저장 경로:", OUT_DIR)

if __name__ == "__main__":
    split_data()
