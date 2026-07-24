"""
model.py

A ResNet18 backbone (ImageNet-pretrained) with its classification head
replaced by a single linear output, for continuous age regression.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model() -> nn.Module:
    """Build a ResNet18 with a single-output regression head in place of the classifier."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model