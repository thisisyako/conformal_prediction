import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

NUM_CLASSES = 9


def load_model(device):

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features, NUM_CLASSES)
    )
    model = model.to(device)
    return model


def freeze_backbone(m, freeze=True):
    for param in m.features.parameters():
        param.requires_grad = not freeze
