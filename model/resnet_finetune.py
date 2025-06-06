# model/get_finetuned_model.py
import torch.nn as nn
import torchvision.models as models
import torch

def get_finetuned_model(weight_path=None, num_classes=27):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if weight_path:
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model
