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

## Key Contributions

* ResNet50 Fine-Tuning
* ORB/SIFT 특징 추출
* CNN + Local Feature Fusion 설계
* Cosine Similarity 기반 검색
* GUI 기반 검색 시스템 구현
