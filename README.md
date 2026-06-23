# Skin Disease Similarity Search

AI 기반 피부질환 유사 이미지 검색 시스템

입력된 피부 이미지를 분석하여 가장 유사한 피부질환 이미지를 검색하는 프로젝트입니다.

단순 분류(Classification)뿐만 아니라 이미지 검색(Content-Based Image Retrieval)을 목표로 하였으며, CNN 특징과 전통적인 컴퓨터 비전 특징(ORB, SIFT)을 결합하여 검색 성능을 향상시켰습니다.

---

## Tech Stack

* Python
* PyTorch
* OpenCV
* ResNet50
* ORB
* SIFT
* Tkinter

---

## Features

### Image Preprocessing

입력 이미지에 대해

* CLAHE
* Gaussian Blur
* Resize (224x224)

전처리를 수행하여 노이즈를 줄이고 특징 추출 성능을 향상시켰습니다.

### CNN Feature Extraction

ResNet50 Fine-Tuning을 수행하여 피부질환 이미지 특징을 학습하였습니다.

* 27개 피부질환 클래스
* Transfer Learning 적용

### Local Feature Extraction

CNN 특징만으로 구분하기 어려운 시각적 패턴을 보완하기 위해

* ORB
* SIFT

특징을 추가 추출하였습니다.

### Feature Fusion

다음 특징들을 결합하여 최종 특징 벡터를 생성하였습니다.

* CNN Feature (2048)
* ORB Feature (1024)
* SIFT Feature (2048)

최종 Feature Vector : 5120 Dimension

### Similarity Search

입력 이미지에 대해

1. 특징 추출
2. Top-K 질환 예측
3. 후보 데이터 필터링
4. Cosine Similarity 계산
5. Top-5 유사 이미지 반환

과정을 수행합니다.

---

## Pipeline

Input Image

→ Image Preprocessing

→ CNN / ORB / SIFT Feature Extraction

→ Feature Fusion

→ Similarity Search

→ Top-5 Similar Images

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

* SD-198 Dataset
* 27 Skin Disease Classes

Dataset:
https://huggingface.co/datasets/resyhgerwshshgdfghsdfgh/SD-198

---

## Troubleshooting
1. CNN 단일 특징만으로 유사 이미지 검색 정확도가 낮은 문제

초기에는 ResNet50에서 추출한 CNN Feature만 사용하여 유사도 검색을 수행했습니다.
하지만 피부질환 이미지는 색상, 경계, 질감이 비슷한 경우가 많아 CNN의 전역 특징만으로는 세밀한 차이를 구분하기 어려웠습니다.

Solution

CNN Feature에 ORB, SIFT 기반 Local Feature를 추가로 결합했습니다.

CNN: 전체적인 형태, 색상, 질감 등 전역 특징 추출
ORB: 빠른 국소 특징 추출
SIFT: 병변 경계와 질감 패턴 추출

이를 통해 CNN + ORB + SIFT 기반 5120차원 Feature Vector를 구성하여 전역 특징과 국소 특징을 함께 활용하도록 개선했습니다.

2. ImageNet 사전학습 모델의 도메인 불일치 문제

초기 ResNet50은 ImageNet 기반 사전학습 모델을 그대로 사용했기 때문에 피부질환 이미지의 세밀한 패턴을 충분히 반영하지 못했습니다.

Solution

SD-198 피부질환 데이터셋을 기반으로 ResNet50 Fine-Tuning을 수행했습니다.
이를 통해 일반 이미지 분류 모델을 피부질환 이미지 도메인에 맞게 조정했습니다.

3. 전체 DB 대상 유사도 검색 시 검색 품질 저하 문제

입력 이미지와 전체 이미지 DB를 바로 비교하면 시각적으로 유사하지만 질환 클래스가 다른 이미지가 상위 결과로 나오는 문제가 있었습니다.

Solution

먼저 FusionClassifier로 Top-K 질환 후보를 예측한 뒤, 해당 후보 클래스에 속한 이미지들만 대상으로 Cosine Similarity 검색을 수행했습니다.

이를 통해 검색 범위를 줄이고, 유사 이미지 검색 결과의 질환 일관성을 높였습니다.

---

## Improvements
CNN, ORB, SIFT 각각의 단독 성능과 Feature Fusion 성능 비교 실험 추가
Top-1 / Top-5 Accuracy, Recall@K 등 정량적 검색 성능 지표 추가
클래스별 데이터 수 차이로 인한 불균형 문제 개선
Grad-CAM 등을 활용하여 모델이 병변의 어떤 영역을 참고했는지 시각화
Tkinter GUI를 Web 기반 UI로 개선
실제 서비스 적용 시 의료 진단이 아닌 참고용 이미지 검색 시스템임을 명확히 표시

---

## Key Contributions

* ResNet50 Fine-Tuning
* ORB/SIFT 특징 추출
* CNN + Local Feature Fusion 설계
* Cosine Similarity 기반 검색
* GUI 기반 검색 시스템 구현
