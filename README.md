# Skin Disease Similarity Search

AI 기반 피부질환 유사 이미지 검색 시스템

입력된 피부 이미지를 분석하여 가장 유사한 피부질환 이미지를 검색하는 프로젝트입니다.

단순 분류(Classification)뿐만 아니라 이미지 검색(Content-Based Image Retrieval, CBIR)을 목표로 하였으며, CNN 특징과 전통적인 컴퓨터 비전 특징(ORB, SIFT)을 결합하여 검색 성능을 향상시켰습니다.

---

## Tech Stack

- Python
- PyTorch
- OpenCV
- ResNet50
- ORB
- SIFT
- Tkinter

---

## Features

### Image Preprocessing

입력 이미지에 대해 다음 전처리를 수행하여 노이즈를 줄이고 특징 추출 성능을 향상시켰습니다.

- CLAHE
- Gaussian Blur
- Resize (224×224)

### CNN Feature Extraction

ResNet50 Fine-Tuning을 수행하여 피부질환 이미지 특징을 학습하였습니다.

- 27개 피부질환 클래스 학습
- Transfer Learning 적용

### Local Feature Extraction

CNN 특징만으로 구분하기 어려운 시각적 패턴을 보완하기 위해 Local Feature를 추가 추출하였습니다.

- ORB
- SIFT

### Feature Fusion

다음 특징들을 결합하여 최종 특징 벡터를 생성하였습니다.

- CNN Feature (2048)
- ORB Feature (1024)
- SIFT Feature (2048)

**Final Feature Vector: 5120 Dimensions**

### Similarity Search

입력 이미지에 대해 다음 과정을 수행합니다.

1. Feature Extraction
2. Top-K Disease Prediction
3. Candidate Filtering
4. Cosine Similarity Calculation
5. Top-5 Similar Images Retrieval

---

## Pipeline

```text
Input Image
    ↓
Image Preprocessing
    ↓
CNN / ORB / SIFT Feature Extraction
    ↓
Feature Fusion
    ↓
Similarity Search
    ↓
Top-5 Similar Images
```

---

## GUI

### Query Image Selection

사용자가 검색할 피부 이미지를 선택합니다.

![image](https://github.com/user-attachments/assets/c478373e-654a-47f7-995e-1c3d17fb1349)

### Search Result

입력 이미지와 가장 유사한 피부질환 이미지 Top-5를 표시합니다.

![image](https://github.com/user-attachments/assets/92caa6e9-7be4-4af3-90f8-ef8dd3e6d2a3)

---

## Dataset

- SD-198 Dataset
- 27 Skin Disease Classes

Dataset:

https://huggingface.co/datasets/resyhgerwshshgdfghsdfgh/SD-198

---

## Troubleshooting

### 1. Low Retrieval Accuracy Using Only CNN Features

#### Problem

초기에는 ResNet50에서 추출한 CNN Feature만 사용하여 유사도 검색을 수행했습니다.

하지만 피부질환 이미지는 색상, 경계, 질감이 유사한 경우가 많아 CNN의 전역 특징(Global Feature)만으로는 세밀한 차이를 구분하기 어려웠습니다.

#### Solution

CNN Feature에 ORB, SIFT 기반 Local Feature를 추가로 결합하였습니다.

- CNN : 전역 특징 추출
- ORB : 국소 특징 추출
- SIFT : 병변 경계 및 질감 특징 추출

이를 통해 전역 특징과 국소 특징을 함께 활용하는 Feature Fusion 구조를 설계하였습니다.

```text
CNN Feature (2048)
+ ORB Feature (1024)
+ SIFT Feature (2048)

= 5120-D Feature Vector
```

---

### 2. Domain Gap of ImageNet Pretrained Model

#### Problem

초기에는 ImageNet 기반 사전학습 모델을 그대로 사용하였습니다.

일반 이미지와 피부질환 이미지는 데이터 특성이 다르기 때문에 피부 병변의 세밀한 특징을 충분히 반영하지 못하는 문제가 있었습니다.

#### Solution

SD-198 Dataset을 활용하여 ResNet50 Fine-Tuning을 수행하였습니다.

이를 통해 피부질환 이미지에 특화된 Feature Representation을 학습하도록 개선하였습니다.

---

### 3. Poor Retrieval Quality When Searching Entire Database

#### Problem

입력 이미지와 전체 데이터셋을 직접 비교하는 방식에서는 시각적으로 유사하지만 실제 질환 클래스가 다른 이미지가 상위 결과로 검색되는 문제가 발생했습니다.

#### Solution

1. Fusion Classifier를 이용하여 Top-K 질환 후보 예측
2. 후보 질환 클래스만 검색 대상으로 제한
3. Cosine Similarity 기반 유사도 계산 수행

이를 통해 검색 범위를 축소하고 질환 일관성을 향상시켰습니다.

---

## Future Improvements

### Quantitative Evaluation

다음과 같은 정량적 성능 평가 지표를 추가할 계획입니다.

- Top-1 Accuracy
- Top-5 Accuracy
- Precision@K
- Recall@K
- Mean Average Precision (mAP)

### Feature Fusion Ablation Study

각 Feature의 기여도를 분석하기 위한 실험을 수행할 수 있습니다.

- CNN Only
- ORB Only
- SIFT Only
- CNN + ORB
- CNN + SIFT
- CNN + ORB + SIFT

### Explainable AI

Grad-CAM 등을 활용하여 모델이 피부 이미지의 어떤 영역을 중요하게 판단했는지 시각화할 수 있습니다.

### Service Expansion

현재 Tkinter 기반 Desktop GUI로 구현되어 있으나 향후 Web 기반 서비스로 확장할 수 있습니다.

- Flask / FastAPI
- React / Next.js
- Image Upload & Search Service

---

## Key Contributions

- ResNet50 Fine-Tuning
- ORB/SIFT Feature Extraction
- CNN + Local Feature Fusion Architecture
- Cosine Similarity-Based Retrieval
- GUI-Based Search System

---

## Author

Kim Minjae
