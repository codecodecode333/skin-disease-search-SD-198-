import os
import cv2
import shutil

# 경로 설정
RAW_DIR = 'raw_images'         # 원본 이미지가 있는 폴더
OUT_DIR = 'processed_images'   # 전처리 이미지 저장 폴더
IMG_SIZE = (224, 224)          # CNN 입력 사이즈

def enhance_image_opencv(img):
    """CLAHE + Blur + Resize 적용"""
    # CLAHE 적용 (LAB 공간)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Gaussian Blur (노이즈 제거)
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)

    # Resize
    img_resized = cv2.resize(img_blur, IMG_SIZE)

    return img_resized

def preprocess_dataset():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    for cls in os.listdir(RAW_DIR):
        cls_path = os.path.join(RAW_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        out_cls_path = os.path.join(OUT_DIR, cls)
        os.makedirs(out_cls_path, exist_ok=True)

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ 이미지 로드 실패: {img_path}")
                    continue
                processed = enhance_image_opencv(img)
                save_path = os.path.join(out_cls_path, img_name)
                cv2.imwrite(save_path, processed)
            except Exception as e:
                print(f"❌ 에러: {img_path} - {e}")

    print("✅ 전처리 완료! 저장 경로:", OUT_DIR)

if __name__ == "__main__":
    preprocess_dataset()
