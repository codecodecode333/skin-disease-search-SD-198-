import os
import cv2
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from model.fusion_model import FusionClassifier
from model.resnet_finetune import get_finetuned_model

# ==================== 설정 ====================
DB_PATH = "fused_db.pkl"
MODEL_PATH = "fusion_model_best.pth"
RESNET_PATH = "resnet_finetuned_best.pth"
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_DIM, ORB_DIM, SIFT_DIM = 2048, 1024, 2048
FUSED_DIM = CNN_DIM + ORB_DIM + SIFT_DIM
NUM_CLASSES = 27
TOP_CLASS_K = 3
TOP_K = 5

# ✅ 인덱스 → 클래스명 리스트
index_to_label = [
    'Acne_Vulgaris', 'Actinic_Keratosis', 'Allergic_Contact_Dermatitis', 'Angioma',
    'Atopic_Dermatitis', 'Basal_Cell_Carcinoma', 'Cellulitis', 'Cutaneous_Horn',
    'Dermatofibroma', 'Eczema', 'Herpes_Simplex_Virus', 'Hidradenitis_Suppurativa',
    'Ichthyosis', 'Keloid', 'Keratosis_Pilaris', 'Lentigo_Maligna_Melanoma',
    'Lichen_Planus', 'Malignant_Melanoma', 'Nevus_Spilus', 'Onychomycosis',
    'Pityriasis_Rosea', 'Psoriasis', 'Pyogenic_Granuloma', 'Rosacea',
    'Seborrheic_Dermatitis', 'Tinea_Corporis', 'Xerosis'
]

# ==================== ResNet 특징 추출 (fc 제외) ====================
def forward_features(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    return torch.flatten(x, 1)

# ==================== 특징 추출 함수 ====================
def extract_query_feature(image_path, cnn_model):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)

    # CNN
    with torch.no_grad():
        feat = cnn_model.forward_features(img_tensor)
        cnn_feat = feat.squeeze().cpu().numpy()

    # ORB
    img_gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()
    _, orb_desc = orb.detectAndCompute(img_gray, None)
    orb_feat = np.zeros((32, 32), dtype=np.uint8)
    if orb_desc is not None:
        orb_feat[:min(len(orb_desc), 32)] = orb_desc[:32]
    orb_feat = normalize(orb_feat.reshape(1, -1))[0]

    # SIFT
    sift = cv2.SIFT_create()
    _, sift_desc = sift.detectAndCompute(img_gray, None)
    sift_feat = np.zeros((16, 128), dtype=np.float32)
    if sift_desc is not None:
        sift_feat[:min(len(sift_desc), 16)] = sift_desc[:16]
    sift_feat = normalize(sift_feat.reshape(1, -1))[0]

    # 결합
    fused = np.concatenate([cnn_feat * 2.0, orb_feat, sift_feat])
    assert fused.shape[0] == FUSED_DIM, f"❌ 벡터 차원 오류: {fused.shape}"
    return fused

# ==================== 클래스 예측 ====================
def predict_topk_classes(feature, model, top_k=3):
    with torch.no_grad():
        input_tensor = torch.tensor(feature).unsqueeze(0).float().to(DEVICE)
        probs = F.softmax(model(input_tensor), dim=1).cpu().numpy()[0]
        return probs.argsort()[::-1][:top_k]

# ==================== 유사도 검색 ====================
def search_top_k(feature, db_items, k=5):
    feats = np.array([item["feature"] for item in db_items])
    sims = cosine_similarity([feature], feats)[0]
    top_indices = sims.argsort()[::-1][:k]
    return [(db_items[i], sims[i]) for i in top_indices]

# ==================== 검색 함수 (GUI에서 호출) ====================
def run_similarity_search(query_path, mode="topk"):
    global fusion_model, cnn_model, db

    # 특징 추출
    feature = extract_query_feature(query_path, cnn_model)

    # Top-N 기반 분류 필터링
    if mode == "topk":
        top_classes = predict_topk_classes(feature, fusion_model, top_k=TOP_CLASS_K)
        top_class_names = [index_to_label[i] for i in top_classes]
        filtered_db = [item for item in db if item["class"] in top_class_names]
    else:
        filtered_db = db

    # 유사도 검색
    results = search_top_k(feature, filtered_db, k=TOP_K)
    return results

# ==================== 모델 및 DB 로딩 ====================
# 이 부분은 모듈 import 시 1회만 로딩됨
fusion_model = FusionClassifier(input_dim=FUSED_DIM, num_classes=NUM_CLASSES).to(DEVICE)
fusion_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
fusion_model.eval()

cnn_model = get_finetuned_model(num_classes=NUM_CLASSES).to(DEVICE)
cnn_model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
cnn_model.eval()
cnn_model.forward_features = forward_features.__get__(cnn_model)

with open(DB_PATH, "rb") as f:
    db = pickle.load(f)
