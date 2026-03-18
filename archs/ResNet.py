import torch
import torch.nn as nn

def ResNet(num_classes=8, pretrained=True):
    if pretrained:
        net = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    else:
        net = torch.hub.load("pytorch/vision", "resnet50")
    final_in_ftrs = net.fc.in_features
    net.fc = nn.Linear(final_in_ftrs, num_classes)
    return net